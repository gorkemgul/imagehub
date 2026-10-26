"""Ovis-Image text-to-image backend for local diffusers snapshots."""

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
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.ovis_image_transformer import OvisImageTransformer2DModel
from imagehub.models.qwen3 import Qwen3Model
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
DEFAULTS: dict[str, Any] = {
    "negative_prompt": "",
    "steps": 50,
    "guidance": 5.0,
    "width": 1024,
    "height": 1024,
    "num_images": 1,
    "seed": None,
    "latents": None,
    "max_sequence_length": 256,
}


@dataclass
class OvisImageComponents:
    tokenizer: Qwen2Tokenizer
    text_encoder: Qwen3Model
    transformer: OvisImageTransformer2DModel
    vae: AutoencoderKL
    scheduler_config: dict[str, Any]

    @classmethod
    def load(
        cls, directory: str | os.PathLike[str], *, device: torch.device, dtype: torch.dtype
    ) -> OvisImageComponents:
        root = Path(directory)
        missing = [part for part in COMPONENTS if not (root / part).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not an Ovis-Image snapshot (missing {', '.join(missing)})"
            )
        return cls(
            Qwen2Tokenizer(root / "tokenizer"),
            Qwen3Model.from_pretrained(root / "text_encoder", device=device, dtype=dtype),
            OvisImageTransformer2DModel.from_pretrained(
                root / "transformer", device=device, dtype=dtype
            ),
            AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
            read_json(root / "scheduler" / "scheduler_config.json"),
        )


def _message(prompt: str) -> str:
    system = (
        "Describe the image by detailing the color, quantity, text, shape, size, texture, "
        "spatial relationships of the objects and background: "
    )
    return (
        f"<|im_start|>user\n{system}{prompt}<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


@torch.inference_mode()
def _embeds(
    parts: OvisImageComponents, prompt: str, count: int, length: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    ids, mask = parts.tokenizer(_message(prompt), max_length=length + 28)
    ids, mask = ids.to(device), mask.to(device)
    hidden = parts.text_encoder(ids, mask) * mask[..., None]
    hidden = hidden[:, 28:].repeat(1, count, 1).view(count, length, -1)
    text_ids = torch.zeros(length, 3, dtype=hidden.dtype, device=device)
    text_ids[:, 1:] = torch.arange(length, device=device, dtype=hidden.dtype)[:, None]
    return hidden, text_ids


def _pack(latents: Tensor) -> Tensor:
    batch, channels, height, width = latents.shape
    return (
        latents.view(batch, channels, height // 2, 2, width // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, -1, channels * 4)
    )


def _unpack(latents: Tensor, height: int, width: int) -> Tensor:
    batch, _, channels = latents.shape
    height, width = height // 8, width // 8
    return (
        latents.view(batch, height // 2, width // 2, channels // 4, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels // 4, height, width)
    )


@torch.inference_mode()
def denoise(
    parts: OvisImageComponents,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    latents: Tensor,
    height: int,
    width: int,
    max_sequence_length: int,
) -> Tensor:
    batch = latents.shape[0]
    prompt_embeds, text_ids = _embeds(parts, prompt, batch, max_sequence_length, latents.device)
    negative, negative_ids = _embeds(
        parts, negative_prompt, batch, max_sequence_length, latents.device
    )
    image_height, image_width = height // 16, width // 16
    image_ids = torch.zeros(
        image_height, image_width, 3, dtype=latents.dtype, device=latents.device
    )
    image_ids[..., 1] = torch.arange(image_height, device=latents.device)[:, None]
    image_ids[..., 2] = torch.arange(image_width, device=latents.device)[None]
    image_ids = image_ids.view(-1, 3)
    scheduler = FlowMatchEulerDiscreteScheduler(parts.scheduler_config)
    sigmas = torch.linspace(1.0, 1 / steps, steps).tolist()
    seq_len = latents.shape[1]
    cfg = parts.scheduler_config
    mu = (
        seq_len
        * (float(cfg["max_shift"]) - float(cfg["base_shift"]))
        / (float(cfg["max_image_seq_len"]) - float(cfg["base_image_seq_len"]))
        + float(cfg["base_shift"])
        - float(cfg["base_image_seq_len"])
        * (float(cfg["max_shift"]) - float(cfg["base_shift"]))
        / (float(cfg["max_image_seq_len"]) - float(cfg["base_image_seq_len"]))
    )
    scheduler.set_timesteps(steps, sigmas=sigmas, mu=mu)
    scheduler.set_begin_index(0)
    for timestep in scheduler.timesteps:
        step = timestep.expand(batch).to(latents.dtype) / 1000
        predicted = parts.transformer(
            latents, prompt_embeds.to(latents.dtype), step, image_ids, text_ids
        )
        if guidance > 1:
            negative_predicted = parts.transformer(
                latents, negative.to(latents.dtype), step, image_ids, negative_ids
            )
            predicted = negative_predicted + guidance * (predicted - negative_predicted)
        latents = scheduler.step(predicted, timestep, latents)
    return latents


class OvisImageBackend(TextToImageBackend):
    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        model_id: str | None = None,
    ) -> None:
        super().__init__()
        self.source, self.device, self.dtype, self.model_id = (
            Path(model),
            resolve_device(device),
            dtype or torch.float32,
            model_id,
        )
        self.components: OvisImageComponents | None = None
        self.snapshot: dict[str, Any] | None = None
        self.defaults = DEFAULTS | (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        selected = self.defaults | (dict(inputs.parameters) | options)
        unknown = set(selected) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"Unknown Ovis-Image options: {', '.join(sorted(unknown))}")
        for name in ("steps", "width", "height", "num_images", "max_sequence_length"):
            if type(selected[name]) is not int or selected[name] <= 0:
                raise ValueError(f"{name} must be a positive int")
        if selected["width"] % 16 or selected["height"] % 16:
            raise ValueError("width and height must be multiples of 16")
        if selected["max_sequence_length"] > 256:
            raise ValueError("max_sequence_length must be at most 256")
        if not isinstance(selected["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string")
        if not isinstance(selected["guidance"], int | float) or not math.isfinite(
            selected["guidance"]
        ):
            raise ValueError("guidance must be finite")
        if selected["seed"] is not None and (
            type(selected["seed"]) is not int or not 0 <= selected["seed"] < 2**64
        ):
            raise ValueError("seed must be an int in [0, 2**64)")
        return selected

    def _load(self) -> None:
        self.components = OvisImageComponents.load(
            self.source, device=self.device, dtype=self.dtype
        )
        self.snapshot = read_snapshot(self.source)

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed = options["seed"] if options["seed"] is not None else secrets.randbits(63)
        shape = (options["num_images"], 16, options["height"] // 8, options["width"] // 8)
        noise = (
            options["latents"]
            if options["latents"] is not None
            else torch.randn(
                shape, generator=torch.Generator("cpu").manual_seed(seed), dtype=self.dtype
            )
        )
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=options["negative_prompt"],
            steps=options["steps"],
            guidance=float(options["guidance"]),
            latents=_pack(noise.to(self.device)),
            height=options["height"],
            width=options["width"],
            max_sequence_length=options["max_sequence_length"],
        )
        pixels = self.components.vae.decode(
            _unpack(final, options["height"], options["width"]) / self.components.vae.scaling_factor
            + (self.components.vae.shift_factor or 0)
        ).clamp(-1, 1)
        pixels = (
            ((pixels / 2 + 0.5).permute(0, 2, 3, 1).float().cpu() * 255).round().to(torch.uint8)
        )
        return ImageResult(
            tuple(Image(pixel) for pixel in pixels),
            {
                "model_id": self.model_id,
                "revision": (self.snapshot or {}).get("revision"),
                "seed": seed,
                "steps": options["steps"],
                "guidance": float(options["guidance"]),
                "device": str(self.device),
                "dtype": str(self.dtype).removeprefix("torch."),
            },
        )
