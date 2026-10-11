"""Kandinsky 3 text-to-image backend over a local diffusers-layout snapshot.

The pipeline order (T5 prompt encoding with the context cut to the longest prompt, the
all-zero negative context when no negative prompt is given, classifier-free guidance as
``(g + 1) * text - g * uncond``, stochastic DDPM steps drawing from the same generator as the
initial noise, MoVQ decoding without quantization and uint8 rounding) follows diffusers'
``Kandinsky3Pipeline``. The repository publishes float16 weights only; the loaders read the
``fp16`` variant file names and the model is run in the requested dtype (float32 by default).
"""

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
from imagehub.models.kandinsky3_unet import Kandinsky3UNet
from imagehub.models.t5 import T5EncoderModel
from imagehub.models.vq_model import VQModel
from imagehub.schedulers.ddpm import DDPMScheduler
from imagehub.tokenization_t5 import T5Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "unet", "movq", "scheduler")
MAX_SEQUENCE_LENGTH = 128
LATENT_CHANNELS = 4
MOVQ_SCALE = 8
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 25,
    "guidance": 3.0,
    "width": 1024,
    "height": 1024,
    "num_images": 1,
    "seed": None,
    "latents": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "kandinsky-3"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class Kandinsky3Components:
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel
    unet: Kandinsky3UNet
    movq: VQModel
    scheduler_config: dict[str, Any]

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> Kandinsky3Components:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Kandinsky 3 snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            scheduler_config = read_json(root / "scheduler" / "scheduler_config.json")
            if scheduler_config.get("_class_name", "DDPMScheduler") != "DDPMScheduler":
                raise ValueError(
                    f"Unsupported Kandinsky 3 scheduler {scheduler_config['_class_name']}"
                )
            DDPMScheduler(scheduler_config)
            return cls(
                tokenizer=T5Tokenizer(root / "tokenizer"),
                text_encoder=T5EncoderModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                unet=Kandinsky3UNet.from_pretrained(root / "unet", device=device, dtype=dtype),
                movq=VQModel.from_pretrained(root / "movq", device=device, dtype=dtype),
                scheduler_config=scheduler_config,
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def latent_size(size: int) -> int:
    """Latent edge for a pixel edge: rounded up to a multiple of 64 pixels, then divided by 8."""
    return math.ceil(size / MOVQ_SCALE**2) * MOVQ_SCALE


@torch.inference_mode()
def _encode(
    components: Kandinsky3Components, texts: list[str], device: torch.device
) -> tuple[Tensor, Tensor]:
    input_ids, mask = components.tokenizer(texts, max_length=MAX_SEQUENCE_LENGTH)
    mask = mask.to(device)
    embeds = components.text_encoder(input_ids.to(device), mask)[0]
    return embeds, mask


@torch.inference_mode()
def encode_prompt(
    components: Kandinsky3Components,
    prompt: str,
    negative_prompt: str | None,
    *,
    do_classifier_free_guidance: bool,
    num_images: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Context ``[(1 + cfg) * num_images, tokens, 4096]`` and its boolean mask.

    With guidance the unconditional half comes first. Without a negative prompt it is all
    zeros with an all-false mask; with one it is the encoded negative prompt cut to the
    conditional length.
    """
    embeds, mask = _encode(components, [prompt], device)
    embeds = embeds.clone()
    embeds[~mask] = 0
    length = int(mask.sum(-1).max()) + 1
    embeds, mask = embeds[:, :length], mask[:, :length]
    embeds = embeds * mask.unsqueeze(2).to(embeds.dtype)
    embeds = embeds.repeat(num_images, 1, 1)
    mask = mask.repeat(num_images, 1)
    if not do_classifier_free_guidance:
        return embeds, mask
    if negative_prompt is None:
        negative_embeds = torch.zeros_like(embeds)
        negative_mask = torch.zeros_like(mask)
    else:
        negative_embeds, negative_mask = _encode(components, [negative_prompt], device)
        negative_embeds = negative_embeds[:, :length]
        negative_mask = negative_mask[:, :length]
        negative_embeds = negative_embeds * negative_mask.unsqueeze(2).to(negative_embeds.dtype)
        negative_embeds = negative_embeds.repeat(num_images, 1, 1)
        negative_mask = negative_mask.repeat(num_images, 1)
    return torch.cat([negative_embeds, embeds]), torch.cat([negative_mask, mask])


@torch.inference_mode()
def denoise(
    components: Kandinsky3Components,
    context: Tensor,
    context_mask: Tensor,
    latents: Tensor,
    *,
    steps: int,
    guidance: float,
    generator: torch.Generator,
) -> Tensor:
    use_cfg = guidance > 1.0
    scheduler = DDPMScheduler(components.scheduler_config)
    scheduler.set_timesteps(steps)
    latents = latents * scheduler.init_noise_sigma
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents] * 2) if use_cfg else latents
        noise = components.unet(model_input, timestep, context, context_mask)
        if use_cfg:
            noise_uncond, noise_text = noise.chunk(2)
            noise = (guidance + 1.0) * noise_text - guidance * noise_uncond
        latents = scheduler.step(noise, timestep, latents, generator=generator)
    return latents


@torch.inference_mode()
def decode_latents(movq: VQModel, latents: Tensor) -> Tensor:
    """Decode final latents to uint8 images of shape (batch, height, width, 3) on CPU."""
    image = movq.decode(latents, force_not_quantize=True)
    image = (image * 0.5 + 0.5).clamp(0, 1)
    return (image.permute(0, 2, 3, 1).float().cpu() * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class Kandinsky3Backend(TextToImageBackend):
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
        self.components: Kandinsky3Components | None = None
        self.snapshot: dict[str, Any] | None = None
        model_defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | model_defaults

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown Kandinsky 3 options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if o["negative_prompt"] is not None and not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "num_images", "width", "height"):
            _positive_int(o, name)
        try:
            train_steps = int(
                read_json(self.source / "scheduler" / "scheduler_config.json")[
                    "num_train_timesteps"
                ]
            )
        except (FileNotFoundError, KeyError):
            train_steps = 1000
        if o["steps"] > train_steps:
            raise ValueError(f"steps must be at most {train_steps}")
        for name in ("width", "height"):
            if o[name] % MOVQ_SCALE:
                raise ValueError(f"{name} must be a multiple of {MOVQ_SCALE}")
        guidance = o["guidance"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, int | float)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError("guidance must be a finite number >= 0")
        seed, latents = o["seed"], o["latents"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        shape = (
            o["num_images"],
            LATENT_CHANNELS,
            latent_size(o["height"]),
            latent_size(o["width"]),
        )
        if latents is not None and (
            not isinstance(latents, Tensor)
            or not latents.is_floating_point()
            or tuple(latents.shape) != shape
        ):
            raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        return o

    def _load(self) -> None:
        self.components = Kandinsky3Components.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was "
                "selected; both share the Kandinsky 3 architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        seed, latents = o["seed"], o["latents"]
        if seed is None:
            seed = secrets.randbits(63)
        generator = torch.Generator("cpu").manual_seed(seed)
        height, width = latent_size(o["height"]), latent_size(o["width"])
        if latents is None:
            latents = torch.randn(
                (o["num_images"], LATENT_CHANNELS, height, width),
                generator=generator,
                dtype=torch.float32,
            )
        use_cfg = float(o["guidance"]) > 1.0
        context, context_mask = encode_prompt(
            self.components,
            inputs.prompt,
            o["negative_prompt"],
            do_classifier_free_guidance=use_cfg,
            num_images=o["num_images"],
            device=self.device,
        )
        final = denoise(
            self.components,
            context,
            context_mask,
            latents.to(device=self.device, dtype=self.dtype),
            steps=o["steps"],
            guidance=float(o["guidance"]),
            generator=generator,
        )
        pixels = decode_latents(self.components.movq, final)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": o["steps"],
            "guidance": float(o["guidance"]),
            "width": width * MOVQ_SCALE,
            "height": height * MOVQ_SCALE,
            "negative_prompt": o["negative_prompt"],
            "scheduler": "ddpm",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(p) for p in pixels), metadata)
