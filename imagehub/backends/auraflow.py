"""AuraFlow text-to-image backend over a local diffusers-layout snapshot."""

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
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.auraflow_transformer import AuraFlowTransformer2DModel
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.umt5 import UMT5EncoderModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_auraflow import AuraFlowTokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 50,
    "guidance": 3.5,
    "width": 1024,
    "height": 1024,
    "seed": None,
    "max_sequence_length": 256,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "auraflow-v0.3"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class AuraFlowComponents:
    tokenizer: AuraFlowTokenizer
    text_encoder: UMT5EncoderModel
    transformer: AuraFlowTransformer2DModel
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
    ) -> AuraFlowComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not an AuraFlow snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            class_name = read_json(root / "model_index.json").get("_class_name")
            if class_name != "AuraFlowPipeline":
                raise CheckpointError(f"Unsupported AuraFlow pipeline class {class_name!r}")
            return cls(
                tokenizer=AuraFlowTokenizer(root / "tokenizer"),
                text_encoder=UMT5EncoderModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                transformer=AuraFlowTransformer2DModel.from_pretrained(
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
    components: AuraFlowComponents, prompt: str, max_length: int, device: torch.device
) -> Tensor:
    """Encoder states of one prompt padded to ``max_length``, zeroed where masked."""
    input_ids, attention_mask = components.tokenizer(
        [prompt], max_length=max_length, padding_side="right"
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device=device, dtype=torch.long)
    embeds = components.text_encoder(input_ids, attention_mask)[0]
    return embeds * attention_mask.unsqueeze(-1).expand(embeds.shape)


@torch.inference_mode()
def encode_prompt(
    components: AuraFlowComponents,
    prompt: str,
    negative_prompt: str | None,
    max_sequence_length: int,
    device: torch.device,
    do_classifier_free_guidance: bool,
) -> tuple[Tensor, Tensor | None]:
    """``AuraFlowPipeline.encode_prompt`` for one prompt: (prompt, negative) embeddings."""
    dtype = components.text_encoder.shared.weight.dtype
    prompt_embeds = _prompt_embeds(components, prompt, max_sequence_length, device).to(
        dtype=dtype, device=device
    )
    if not do_classifier_free_guidance:
        return prompt_embeds, None
    negative = _prompt_embeds(components, negative_prompt or "", prompt_embeds.shape[1], device).to(
        dtype=dtype, device=device
    )
    return prompt_embeds, negative


@torch.inference_mode()
def denoise(
    components: AuraFlowComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    max_sequence_length: int,
    latents: Tensor,
) -> Tensor:
    use_cfg = guidance > 1.0
    prompt_embeds, negative_embeds = encode_prompt(
        components, prompt, negative_prompt, max_sequence_length, latents.device, use_cfg
    )
    if negative_embeds is not None:
        prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    scheduler.set_timesteps(steps)
    latents = latents.to(prompt_embeds.dtype)
    for t in scheduler.timesteps:
        model_input = torch.cat([latents] * 2) if use_cfg else latents
        # AuraFlow takes timesteps in [0, 1], with 1 as noise.
        timestep = (t / 1000).reshape(1).expand(model_input.shape[0])
        timestep = timestep.to(latents.device, dtype=latents.dtype)
        noise = components.transformer(model_input, prompt_embeds, timestep)
        if use_cfg:
            noise_unconditional, noise_text = noise.chunk(2)
            noise = noise_unconditional + guidance * (noise_text - noise_unconditional)
        latents = scheduler.step(noise, t, latents)
    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: Tensor) -> Tensor:
    """Decode final latents to uint8 images of shape (batch, height, width, 3) on CPU."""
    dtype = vae.decoder.conv_in.weight.dtype
    if dtype == torch.float16 and vae.config.get("force_upcast"):
        # The VAE overflows in float16.
        vae.to(dtype=torch.float32)
        dtype = torch.float32
    image = vae.decode(latents.to(dtype) / vae.scaling_factor)
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class AuraFlowBackend(TextToImageBackend):
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
        self.components: AuraFlowComponents | None = None
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
            raise TypeError(f"Unknown AuraFlow options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if o["negative_prompt"] is not None and not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "width", "height", "max_sequence_length"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        guidance = o["guidance"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, int | float)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError("guidance must be a finite number >= 0")
        seed = o["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        try:
            transformer_config = read_json(self.source / "transformer" / "config.json")
        except FileNotFoundError:
            transformer_config = {"in_channels": 4}
        try:
            vae_config = read_json(self.source / "vae" / "config.json")
        except FileNotFoundError:
            vae_config = {"block_out_channels": [1, 1, 1, 1]}
        vae_scale = 2 ** (len(vae_config.get("block_out_channels", [1, 1, 1, 1])) - 1)
        for name in ("width", "height"):
            if o[name] % (vae_scale * 2):
                raise ValueError(f"{name} must be a multiple of {vae_scale * 2}")
        grid = math.isqrt(int(transformer_config.get("pos_embed_max_size", 0)))
        limit = grid * int(transformer_config.get("patch_size", 2)) * vae_scale
        if grid and (o["width"] > limit or o["height"] > limit):
            raise ValueError(f"width and height must be at most {limit}")
        o["latent_shape"] = (
            1,
            int(transformer_config.get("in_channels", 4)),
            o["height"] // vae_scale,
            o["width"] // vae_scale,
        )
        return o

    def _load(self) -> None:
        self.components = AuraFlowComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the AuraFlow architecture, so it was loaded",
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
        latents = torch.randn(
            o["latent_shape"],
            generator=torch.Generator("cpu").manual_seed(seed),
            dtype=torch.float32,
        )
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
            steps=o["steps"],
            guidance=float(o["guidance"]),
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
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
