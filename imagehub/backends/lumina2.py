"""Lumina-Image 2.0 text-to-image backend over a local diffusers snapshot."""

from __future__ import annotations

import math
import os
import secrets
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.flux import _sigmas
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.gemma2 import Gemma2Model
from imagehub.models.lumina2_transformer import Lumina2Transformer2DModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_gemma import GemmaTokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant designed to generate superior images with the superior degree of "
    "image-text alignment based on textual prompts or user prompts."
)
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 30,
    "guidance": 4.0,
    "width": 1024,
    "height": 1024,
    "seed": None,
    "negative_prompt": "",
    "system_prompt": None,
    "cfg_trunc_ratio": 1.0,
    "cfg_normalization": True,
    "max_sequence_length": 256,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "lumina-image-2.0"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class Lumina2Components:
    tokenizer: GemmaTokenizer
    text_encoder: Gemma2Model
    transformer: Lumina2Transformer2DModel
    vae: AutoencoderKL
    scheduler_config: dict[str, Any]

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> Lumina2Components:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Lumina-Image 2.0 snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            class_name = read_json(root / "model_index.json").get("_class_name")
            if class_name != "Lumina2Pipeline":
                raise CheckpointError(f"Unsupported Lumina-Image 2.0 pipeline class {class_name!r}")
            return cls(
                tokenizer=GemmaTokenizer(root / "tokenizer"),
                text_encoder=Gemma2Model.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                transformer=Lumina2Transformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _prompt_embeds(
    components: Lumina2Components,
    text: str,
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """``Lumina2Pipeline._get_gemma_prompt_embeds`` for one text (right padding).

    The embeddings are the second-to-last hidden state of the text encoder.
    """
    input_ids, attention_mask = components.tokenizer(
        [text], max_length=max_sequence_length, padding_side="right"
    )
    attention_mask = attention_mask.to(device)
    embeds = components.text_encoder(
        input_ids.to(device), attention_mask.long(), hidden_state_index=-2
    )
    return embeds.to(dtype=dtype, device=device), attention_mask


@torch.inference_mode()
def encode_prompt(
    components: Lumina2Components,
    prompt: str,
    negative_prompt: str,
    system_prompt: str,
    max_sequence_length: int,
    device: torch.device,
    do_classifier_free_guidance: bool,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Prompt and negative embeddings with their masks; the negative prompt has no system prompt."""
    dtype = components.text_encoder.embed_tokens.weight.dtype
    embeds, mask = _prompt_embeds(
        components,
        system_prompt + " <Prompt Start> " + prompt,
        max_sequence_length,
        device,
        dtype,
    )
    if not do_classifier_free_guidance:
        return embeds, mask, None, None
    negative_embeds, negative_mask = _prompt_embeds(
        components, negative_prompt, max_sequence_length, device, dtype
    )
    return embeds, mask, negative_embeds, negative_mask


@torch.inference_mode()
def denoise(
    components: Lumina2Components,
    *,
    prompt: str,
    negative_prompt: str,
    system_prompt: str,
    steps: int,
    guidance: float,
    cfg_trunc_ratio: float,
    cfg_normalization: bool,
    max_sequence_length: int,
    latents: Tensor,
) -> Tensor:
    use_cfg = guidance > 1.0
    transformer = components.transformer
    prompt_embeds, prompt_mask, negative_embeds, negative_mask = encode_prompt(
        components,
        prompt,
        negative_prompt,
        system_prompt,
        max_sequence_length,
        latents.device,
        use_cfg,
    )
    dtype = transformer.x_embedder.weight.dtype
    prompt_embeds = prompt_embeds.to(dtype)
    prompt_mask = prompt_mask.long()
    if negative_embeds is not None and negative_mask is not None:
        negative_embeds = negative_embeds.to(dtype)
        negative_mask = negative_mask.long()
    latents = latents.to(prompt_embeds.dtype)
    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    # The published scheduler uses a static shift, so the pipeline's ``mu`` has no effect.
    scheduler.set_timesteps(steps, sigmas=_sigmas(steps))
    train_steps = scheduler.config["num_train_timesteps"]
    for index, timestep in enumerate(scheduler.timesteps):
        truncated = (index + 1) / steps > cfg_trunc_ratio
        # Lumina uses t=0 as the noise and t=1 as the image.
        model_timestep = (1 - timestep / train_steps).expand(latents.shape[0])
        noise_pred = transformer(latents.to(dtype), model_timestep, prompt_embeds, prompt_mask)
        if use_cfg and not truncated:
            assert negative_embeds is not None and negative_mask is not None
            noise_uncond = transformer(
                latents.to(dtype), model_timestep, negative_embeds, negative_mask
            )
            guided = noise_uncond + guidance * (noise_pred - noise_uncond)
            if cfg_normalization:
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                guided_norm = torch.norm(guided, dim=-1, keepdim=True)
                guided = guided * (cond_norm / guided_norm)
            noise_pred = guided
        latents = scheduler.step(-noise_pred, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: Tensor) -> Tensor:
    latents = latents / vae.scaling_factor + (vae.shift_factor or 0.0)
    image = vae.decode(latents.to(vae.decoder.conv_in.weight.dtype))
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


def _finite_number(options: dict[str, Any], name: str) -> None:
    value = options[name]
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite number >= 0")


class Lumina2Backend(TextToImageBackend):
    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        model_id: str | None = None,
    ) -> None:
        super().__init__()
        self.source = Path(model)
        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = dtype or torch.float32
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be torch.float32, torch.float16 or torch.bfloat16")
        self.components: Lumina2Components | None = None
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown Lumina-Image 2.0 options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if o["negative_prompt"] is None:
            o["negative_prompt"] = ""
        if not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        if o["system_prompt"] is None:
            o["system_prompt"] = DEFAULT_SYSTEM_PROMPT
        if not isinstance(o["system_prompt"], str):
            raise TypeError("system_prompt must be a string or None")
        if type(o["cfg_normalization"]) is not bool:
            raise TypeError("cfg_normalization must be a bool")
        for name in ("steps", "width", "height", "max_sequence_length"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        if o["max_sequence_length"] > 512:
            raise ValueError("max_sequence_length must be at most 512")
        for name in ("guidance", "cfg_trunc_ratio"):
            _finite_number(o, name)
        seed = o["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        try:
            transformer_config = read_json(self.source / "transformer" / "config.json")
        except FileNotFoundError:
            transformer_config = {}
        try:
            blocks = read_json(self.source / "vae" / "config.json").get(
                "block_out_channels", [1, 1, 1, 1]
            )
        except FileNotFoundError:
            blocks = [1, 1, 1, 1]
        vae_scale = 2 ** (len(blocks) - 1)
        patch = int(transformer_config.get("patch_size", 2))
        for name in ("width", "height"):
            if o[name] % (vae_scale * patch):
                raise ValueError(f"{name} must be a multiple of {vae_scale * patch}")
        axes_lens = transformer_config.get("axes_lens")
        if axes_lens is not None:
            if o["max_sequence_length"] >= axes_lens[0]:
                raise ValueError(f"max_sequence_length must be below {axes_lens[0]}")
            if o["height"] // (vae_scale * patch) > axes_lens[1]:
                raise ValueError(f"height must be at most {axes_lens[1] * vae_scale * patch}")
            if o["width"] // (vae_scale * patch) > axes_lens[2]:
                raise ValueError(f"width must be at most {axes_lens[2] * vae_scale * patch}")
        o["latent_shape"] = (
            1,
            int(transformer_config.get("in_channels", 16)),
            o["height"] // vae_scale,
            o["width"] // vae_scale,
        )
        return o

    def _load(self) -> None:
        self.components = Lumina2Components.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the Lumina-Image 2.0 architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        seed = o["seed"]
        if seed is None:
            seed = secrets.randbits(63)
        generator = torch.Generator("cpu").manual_seed(seed)
        latents = torch.randn(o["latent_shape"], generator=generator, dtype=torch.float32)
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
            system_prompt=o["system_prompt"],
            steps=o["steps"],
            guidance=float(o["guidance"]),
            cfg_trunc_ratio=float(o["cfg_trunc_ratio"]),
            cfg_normalization=o["cfg_normalization"],
            max_sequence_length=o["max_sequence_length"],
            latents=latents.to(self.device),
        )
        pixels = decode_latents(self.components.vae, final)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": o["steps"],
            "guidance": float(o["guidance"]),
            "width": o["width"],
            "height": o["height"],
            "negative_prompt": o["negative_prompt"],
            "cfg_trunc_ratio": float(o["cfg_trunc_ratio"]),
            "cfg_normalization": o["cfg_normalization"],
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
