"""Latent Diffusion Model text-to-image backend."""

from __future__ import annotations

import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.ldm_bert import LDMBertModel
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.ddim import DDIMScheduler
from imagehub.tokenization_bert import BertTokenizer
from imagehub.types import ImageResult, T2IRequest

OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 50,
    "guidance": 1.0,
    "width": 256,
    "height": 256,
    "num_images": 1,
    "seed": None,
    "latents": None,
    "eta": 0.0,
}


class LDMUNet(UNet2DConditionModel):
    """The early config predates ``cross_attention_dim`` but weights encode 1280-wide text."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__({"cross_attention_dim": 1280} | config)


@dataclass
class LDMComponents:
    tokenizer: BertTokenizer
    bert: LDMBertModel
    unet: LDMUNet
    vqvae: AutoencoderKL
    scheduler_config: dict[str, Any]

    @classmethod
    def load(cls, root: Path, device: torch.device, dtype: torch.dtype) -> LDMComponents:
        needed = ("tokenizer", "bert", "unet", "vqvae", "scheduler")
        missing = [name for name in needed if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(f"{root} is not an LDM snapshot (missing {', '.join(missing)})")
        return cls(
            BertTokenizer(root / "tokenizer"),
            LDMBertModel.from_pretrained(root / "bert", device=device, dtype=dtype),
            LDMUNet.from_pretrained(root / "unet", device=device, dtype=dtype),
            AutoencoderKL.from_pretrained(root / "vqvae", device=device, dtype=dtype),
            read_json(root / "scheduler" / "scheduler_config.json"),
        )


@torch.inference_mode()
def encode_prompt(components: LDMComponents, prompt: str, device: torch.device) -> Tensor:
    ids, _ = components.tokenizer([prompt], 77)
    return components.bert(ids.to(device))


@torch.inference_mode()
def denoise(
    components: LDMComponents, prompt: str, options: dict[str, Any], latents: Tensor
) -> Tensor:
    dtype = components.unet.conv_in.weight.dtype
    batch = latents.shape[0]
    cond = encode_prompt(components, prompt, latents.device).to(dtype).repeat(batch, 1, 1)
    scheduler = DDIMScheduler(components.scheduler_config)
    scheduler.set_timesteps(options["steps"])
    if options["guidance"] != 1.0:
        uncond = encode_prompt(components, options["negative_prompt"] or "", latents.device)
        context = torch.cat([uncond.to(dtype).repeat(batch, 1, 1), cond])
    else:
        context = cond
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents] * 2) if options["guidance"] != 1.0 else latents
        noise = components.unet(model_input, timestep, context)
        if options["guidance"] != 1.0:
            uncond, text = noise.chunk(2)
            noise = uncond + options["guidance"] * (text - uncond)
        latents = scheduler.step(noise, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(components: LDMComponents, latents: Tensor) -> Tensor:
    image = components.vqvae.decode((1 / components.vqvae.scaling_factor) * latents)
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


class LDMBackend(TextToImageBackend):
    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        model_id: str | None = None,
    ) -> None:
        super().__init__()
        self.source, self.model_id, self.device = Path(model), model_id, resolve_device(device)
        self.dtype = dtype or torch.float32
        if self.dtype is not torch.float32:
            raise ValueError("LDM supports torch.float32 only")
        self.components: LDMComponents | None = None
        self.snapshot: dict[str, Any] | None = None

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown LDM options: {', '.join(unknown)}")
        result = OPTION_DEFAULTS | merged
        for name in ("steps", "width", "height", "num_images"):
            if type(result[name]) is not int or result[name] <= 0:
                raise ValueError(f"{name} must be a positive int")
        if result["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        if result["width"] % 8 or result["height"] % 8:
            raise ValueError("width and height must be multiples of 8")
        if (
            not isinstance(result["guidance"], int | float)
            or isinstance(result["guidance"], bool)
            or not math.isfinite(result["guidance"])
        ):
            raise ValueError("guidance must be a finite number")
        if result["eta"] != 0.0:
            raise ValueError("eta must be 0.0")
        if result["negative_prompt"] is not None and not isinstance(result["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        if result["seed"] is not None and (
            type(result["seed"]) is not int or not 0 <= result["seed"] < 2**64
        ):
            raise ValueError("seed must be an int in [0, 2**64)")
        shape = (result["num_images"], 4, result["height"] // 8, result["width"] // 8)
        if result["latents"] is not None:
            if result["seed"] is not None:
                raise ValueError("Pass either seed or latents, not both")
            if (
                not isinstance(result["latents"], Tensor)
                or not result["latents"].is_floating_point()
                or tuple(result["latents"].shape) != shape
            ):
                raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        return result

    def _load(self) -> None:
        self.components = LDMComponents.load(self.source, self.device, self.dtype)
        self.snapshot = read_snapshot(self.source)

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed, latents = options["seed"], options["latents"]
        if latents is None:
            seed = secrets.randbits(63) if seed is None else seed
            generator = torch.Generator("cpu").manual_seed(seed)
            latents = torch.randn(
                (options["num_images"], 4, options["height"] // 8, options["width"] // 8),
                generator=generator,
            )
        final = denoise(
            self.components, inputs.prompt, options, latents.to(self.device, torch.float32)
        )
        pixels = decode_latents(self.components, final)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": options["steps"],
            "guidance": float(options["guidance"]),
            "width": options["width"],
            "height": options["height"],
            "negative_prompt": options["negative_prompt"],
            "scheduler": "ddim",
            "device": str(self.device),
            "dtype": "float32",
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
