"""Lumina-Next text-to-image backend over a local diffusers snapshot."""

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
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.gemma import GemmaModel
from imagehub.models.lumina_next_transformer import (
    ROTARY_TABLE_SIZE,
    LuminaNextDiT2DModel,
    rotary_frequencies,
)
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_gemma import GemmaTokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
PIPELINE_CLASSES = ("LuminaText2ImgPipeline", "LuminaPipeline")
#: The pipeline fixes the prompt length; its ``max_sequence_length`` argument is not used.
MAX_SEQUENCE_LENGTH = 256
#: Prompt lengths are padded up to a multiple of this.
PAD_MULTIPLE = 8
VAE_SCALE_FACTOR = 8
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 30,
    "guidance": 4.0,
    "width": 1024,
    "height": 1024,
    "seed": None,
    "negative_prompt": "",
    "scaling_watershed": 1.0,
    "proportional_attn": True,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "lumina-next"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class LuminaNextComponents:
    tokenizer: GemmaTokenizer
    text_encoder: GemmaModel
    transformer: LuminaNextDiT2DModel
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
    ) -> LuminaNextComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Lumina-Next snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            class_name = read_json(root / "model_index.json").get("_class_name")
            if class_name not in PIPELINE_CLASSES:
                raise CheckpointError(f"Unsupported Lumina-Next pipeline class {class_name!r}")
            return cls(
                tokenizer=GemmaTokenizer(root / "tokenizer"),
                text_encoder=GemmaModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                transformer=LuminaNextDiT2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _encode(
    components: LuminaNextComponents,
    ids: list[int],
    length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Second-to-last hidden state of one right-padded token row, and its mask."""
    tokenizer = components.tokenizer
    padding = length - len(ids)
    input_ids = torch.tensor([ids + [tokenizer.pad_id] * padding], dtype=torch.long)
    mask = torch.tensor([[1] * len(ids) + [0] * padding], dtype=torch.long)
    mask = mask.to(device)
    embeds = components.text_encoder(input_ids.to(device), mask, hidden_state_index=-2)
    return embeds.to(dtype=dtype, device=device), mask


@torch.inference_mode()
def encode_prompt(
    components: LuminaNextComponents,
    prompt: str,
    negative_prompt: str,
    device: torch.device,
    do_classifier_free_guidance: bool,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """``LuminaPipeline.encode_prompt`` for one prompt (the cleaning step is skipped).

    The prompt is lower-cased and stripped, padded to the next multiple of eight tokens, and
    truncated to 256 tokens including the start and end tokens. The negative prompt is used
    as written and padded to the prompt's length.
    """
    tokenizer = components.tokenizer
    dtype = components.text_encoder.embed_tokens.weight.dtype
    ids = tokenizer.tokenize(prompt.lower().strip(), MAX_SEQUENCE_LENGTH)
    length = -(-len(ids) // PAD_MULTIPLE) * PAD_MULTIPLE
    embeds, mask = _encode(components, ids, length, device, dtype)
    if not do_classifier_free_guidance:
        return embeds, mask, None, None
    negative_ids = tokenizer.tokenize(negative_prompt, length)
    negative_embeds, negative_mask = _encode(components, negative_ids, length, device, dtype)
    return embeds, mask, negative_embeds, negative_mask


@torch.inference_mode()
def denoise(
    components: LuminaNextComponents,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    scaling_watershed: float,
    proportional_attn: bool,
    latents: Tensor,
) -> Tensor:
    use_cfg = guidance > 1.0
    transformer = components.transformer
    device = latents.device
    prompt_embeds, prompt_mask, negative_embeds, negative_mask = encode_prompt(
        components, prompt, negative_prompt, device, use_cfg
    )
    if use_cfg:
        assert negative_embeds is not None and negative_mask is not None
        prompt_embeds = torch.cat([prompt_embeds, negative_embeds], dim=0)
        prompt_mask = torch.cat([prompt_mask, negative_mask], dim=0)
    dtype = prompt_embeds.dtype
    latents = latents.to(dtype)

    # The pipeline measures image size in pixels against the published sample size.
    default_image_size = int(transformer.config.get("sample_size", 128)) * VAE_SCALE_FACTOR
    pixels = latents.shape[2] * VAE_SCALE_FACTOR * latents.shape[3] * VAE_SCALE_FACTOR
    scaling_factor = math.sqrt(pixels / default_image_size**2)
    base_sequence_length = (default_image_size // 16) ** 2 if proportional_attn else None

    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    scheduler.set_timesteps(steps)
    train_steps = scheduler.config["num_train_timesteps"]
    rotary_cache: dict[tuple[float, float], Tensor] = {}
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents] * 2) if use_cfg else latents
        # Lumina uses t=0 as the noise and t=1 as the image.
        model_timestep = (1 - timestep.expand(model_input.shape[0]) / train_steps).to(device)
        # Time-aware scaled RoPE: stretch positions early on, rescale the base late.
        if model_timestep[0] < scaling_watershed:
            factors = (scaling_factor, 1.0)
        else:
            factors = (1.0, scaling_factor)
        if factors not in rotary_cache:
            rotary_cache[factors] = rotary_frequencies(
                transformer.head_dim,
                ROTARY_TABLE_SIZE,
                ROTARY_TABLE_SIZE,
                linear_factor=factors[0],
                ntk_factor=factors[1],
            )
        noise_pred = transformer(
            model_input,
            model_timestep,
            prompt_embeds,
            prompt_mask,
            rotary_cache[factors],
            base_sequence_length=base_sequence_length,
        )
        # The model also predicts a variance; only the first half of the channels is the update.
        noise_pred = noise_pred.chunk(2, dim=1)[0]
        if use_cfg:
            # Guidance is applied to the first three channels only, as the reference does.
            noise_eps, noise_rest = noise_pred[:, :3], noise_pred[:, 3:]
            noise_cond, noise_uncond = torch.split(noise_eps, len(noise_eps) // 2, dim=0)
            noise_half = noise_uncond + guidance * (noise_cond - noise_uncond)
            noise_eps = torch.cat([noise_half, noise_half], dim=0)
            noise_pred = torch.cat([noise_eps, noise_rest], dim=1)
            noise_pred, _ = noise_pred.chunk(2, dim=0)
        latents = scheduler.step(-noise_pred, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: Tensor) -> Tensor:
    image = vae.decode((latents / vae.scaling_factor).to(vae.decoder.conv_in.weight.dtype))
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


def _finite_number(options: dict[str, Any], name: str, *, minimum: float | None = 0) -> None:
    value = options[name]
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be a finite number{suffix}")


class LuminaNextBackend(TextToImageBackend):
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
        self.components: LuminaNextComponents | None = None
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
            raise TypeError(f"Unknown Lumina-Next options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if o["negative_prompt"] is None:
            o["negative_prompt"] = ""
        if not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        if type(o["proportional_attn"]) is not bool:
            raise TypeError("proportional_attn must be a bool")
        for name in ("steps", "width", "height"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        _finite_number(o, "guidance")
        _finite_number(o, "scaling_watershed", minimum=None)
        seed = o["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        try:
            transformer_config = read_json(self.source / "transformer" / "config.json")
        except FileNotFoundError:
            transformer_config = {}
        step = VAE_SCALE_FACTOR * int(transformer_config.get("patch_size", 2))
        for name in ("width", "height"):
            if o[name] % step:
                raise ValueError(f"{name} must be a multiple of {step}")
            if o[name] // step > ROTARY_TABLE_SIZE:
                raise ValueError(f"{name} must be at most {ROTARY_TABLE_SIZE * step}")
        o["latent_shape"] = (
            1,
            int(transformer_config.get("in_channels", 4)),
            o["height"] // VAE_SCALE_FACTOR,
            o["width"] // VAE_SCALE_FACTOR,
        )
        return o

    def _load(self) -> None:
        self.components = LuminaNextComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the Lumina-Next architecture, so it was loaded",
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
            steps=o["steps"],
            guidance=float(o["guidance"]),
            scaling_watershed=float(o["scaling_watershed"]),
            proportional_attn=o["proportional_attn"],
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
            "scaling_watershed": float(o["scaling_watershed"]),
            "proportional_attn": o["proportional_attn"],
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
