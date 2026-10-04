"""Stable Diffusion 1.x text-to-image backend over a local diffusers-layout snapshot.

The sampling order (prompt encoding, classifier-free guidance batch order, noise scaling,
PLMS steps, VAE decoding and uint8 rounding) follows diffusers' StableDiffusionPipeline.
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
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.pndm import PNDMScheduler
from imagehub.tokenization import CLIPTokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "unet", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 50,
    "guidance": 7.5,
    "width": 512,
    "height": 512,
    "num_images": 1,
    "seed": None,
    "latents": None,
}
LATENT_CHANNELS = 4
LATENT_SCALE = 8
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def resolve_device(device: str | torch.device) -> torch.device:
    available = ["cpu"]
    if torch.cuda.is_available():
        available.append("cuda")
    if torch.backends.mps.is_available():
        available.append("mps")
    try:
        selected = torch.device(device)
    except RuntimeError:
        raise ValueError(f"Unknown device {device!r}. Available: {', '.join(available)}") from None
    if selected.type not in available:
        raise ValueError(
            f"Device {str(selected)!r} is not available. Available: {', '.join(available)}"
        )
    return selected


def initial_latents(seed: int, shape: tuple[int, ...]) -> Tensor:
    """Noise drawn like diffusers with a CPU ``torch.Generator`` seeded by ``seed``."""
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def _hint(root: Path, model_id: str | None) -> str:
    return f"Download a snapshot with: python -m imagehub.pretrained {model_id or 'sd-v1-5'} {root}"


@dataclass
class StableDiffusionComponents:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    unet: UNet2DConditionModel
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
    ) -> StableDiffusionComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Stable Diffusion snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            return cls(
                tokenizer=CLIPTokenizer(root / "tokenizer"),
                text_encoder=CLIPTextModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                unet=UNet2DConditionModel.from_pretrained(
                    root / "unet", device=device, dtype=dtype
                ),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


@torch.inference_mode()
def encode_prompt(
    components: StableDiffusionComponents, prompt: str, device: torch.device
) -> Tensor:
    ids = torch.tensor([components.tokenizer(prompt)], device=device)
    return components.text_encoder(ids)


@torch.inference_mode()
def denoise(
    components: StableDiffusionComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    latents: Tensor,
) -> Tensor:
    """Run the PLMS loop from initial noise ``latents`` (on the model device); return latents."""
    dtype = components.unet.conv_in.weight.dtype
    batch = latents.shape[0]
    cond = encode_prompt(components, prompt, latents.device).to(dtype).repeat(batch, 1, 1)
    use_cfg = guidance > 1.0
    if use_cfg:
        uncond = encode_prompt(components, negative_prompt or "", latents.device)
        context = torch.cat([uncond.to(dtype).repeat(batch, 1, 1), cond])
    else:
        context = cond
    scheduler = PNDMScheduler(components.scheduler_config)
    scheduler.set_timesteps(steps)
    latents = latents.to(dtype) * scheduler.init_noise_sigma
    for t in scheduler.timesteps:
        model_input = torch.cat([latents] * 2) if use_cfg else latents
        noise = components.unet(model_input, t, context)
        if use_cfg:
            noise_uncond, noise_text = noise.chunk(2)
            noise = noise_uncond + guidance * (noise_text - noise_uncond)
        latents = scheduler.step(noise, t, latents)
    return latents


@torch.inference_mode()
def decode_latents(components: StableDiffusionComponents, latents: Tensor) -> Tensor:
    """Decode final latents to uint8 images of shape (batch, height, width, 3) on CPU."""
    vae = components.vae
    image = vae.decode(latents / vae.scaling_factor)
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    value = options[name]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")


class StableDiffusionBackend(TextToImageBackend):
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
        self.components: StableDiffusionComponents | None = None
        self.snapshot: dict[str, Any] | None = None

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown Stable Diffusion options: {', '.join(unknown)}")
        o = OPTION_DEFAULTS | merged
        if o["negative_prompt"] is not None and not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "num_images", "width", "height"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        for name in ("width", "height"):
            if o[name] % LATENT_SCALE:
                raise ValueError(f"{name} must be a multiple of {LATENT_SCALE}")
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
        if latents is not None:
            if seed is not None:
                raise ValueError("Pass either seed or latents, not both")
            shape = (
                o["num_images"],
                LATENT_CHANNELS,
                o["height"] // LATENT_SCALE,
                o["width"] // LATENT_SCALE,
            )
            if (
                not isinstance(latents, Tensor)
                or not latents.is_floating_point()
                or tuple(latents.shape) != shape
            ):
                raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        return o

    def _load(self) -> None:
        self.components = StableDiffusionComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was "
                "selected; both share the Stable Diffusion 1.x architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        seed, latents = o["seed"], o["latents"]
        if latents is None:
            if seed is None:
                seed = secrets.randbits(63)
            shape = (
                o["num_images"],
                LATENT_CHANNELS,
                o["height"] // LATENT_SCALE,
                o["width"] // LATENT_SCALE,
            )
            latents = initial_latents(seed, shape)
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
            steps=o["steps"],
            guidance=float(o["guidance"]),
            latents=latents.to(torch.float32).to(self.device),
        )
        pixels = decode_latents(self.components, final)
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
            "scheduler": "pndm",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(p) for p in pixels), metadata)
