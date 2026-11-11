"""Nucleus-Image text-to-image backend."""

# mypy: ignore-errors

from __future__ import annotations

import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.krea2 import (
    _load_text_encoder,
    calculate_shift,
    pack_latents,
    unpack_latents,
)
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl_qwen_image import AutoencoderKLQwenImage
from imagehub.models.nucleus_image_transformer import NucleusImageTransformer2DModel
from imagehub.models.qwen3_vl_text import Qwen3VLTextModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 50,
    "guidance": 4.0,
    "width": 1024,
    "height": 1024,
    "seed": None,
    "latents": None,
    "max_sequence_length": 1024,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_SYSTEM = (
    "You are an image generation assistant. Follow the user's prompt literally. "
    "Pay careful attention to spatial layout: objects described as on the left must appear on "
    "the left, on the right on the right. Match exact object counts and assign colors to the "
    "correct objects."
)


@dataclass
class NucleusImageComponents:
    tokenizer: Qwen2Tokenizer
    text_encoder: Qwen3VLTextModel
    transformer: NucleusImageTransformer2DModel
    vae: AutoencoderKLQwenImage
    scheduler_config: dict[str, Any]

    @classmethod
    def load(cls, root: Path, device: torch.device, dtype: torch.dtype) -> NucleusImageComponents:
        required = ("processor", "text_encoder", "transformer", "vae", "scheduler")
        if missing := [name for name in required if not (root / name).is_dir()]:
            raise CheckpointError(
                f"{root} is not a Nucleus-Image snapshot (missing {', '.join(missing)})"
            )
        return cls(
            Qwen2Tokenizer(root / "processor"),
            _load_text_encoder(root / "text_encoder", device=device, dtype=dtype),
            NucleusImageTransformer2DModel.from_pretrained(
                root / "transformer", device=device, dtype=dtype
            ),
            AutoencoderKLQwenImage.from_pretrained(root / "vae", device=device, dtype=dtype),
            read_json(root / "scheduler" / "scheduler_config.json"),
        )


def _format(prompt: str) -> str:
    return (
        f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n<|im_start|>user\n"
        f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )


@torch.inference_mode()
def encode_prompt(
    components: NucleusImageComponents, prompt: str, device: torch.device, maximum: int
) -> tuple[Tensor, Tensor | None]:
    formatted = _format(prompt)
    length = min(len(components.tokenizer.tokenize(formatted)), maximum)
    padded_length = (length + 7) // 8 * 8
    ids, mask = components.tokenizer(formatted, padded_length)
    positions = (mask.long().cumsum(-1) - 1).clamp(min=0).unsqueeze(0).expand(3, -1, -1).to(device)
    states = components.text_encoder(
        ids.to(device), mask.to(device), position_ids=positions, output_hidden_states=True
    )
    mask = mask.to(device)
    return states[-8], None if mask.all() else mask


def _sigmas(steps: int) -> list[float]:
    return (
        [1.0]
        if steps == 1
        else [1.0 + index * ((1 / steps - 1.0) / (steps - 1)) for index in range(steps)]
    )


@torch.inference_mode()
def denoise(
    components: NucleusImageComponents,
    prompt: str,
    negative: str | None,
    steps: int,
    guidance: float,
    latents: Tensor,
    height: int,
    width: int,
    maximum: int,
) -> Tensor:
    dtype = cast(torch.dtype, components.transformer.img_in.weight.dtype)
    cond, cond_mask = encode_prompt(components, prompt, latents.device, maximum)
    uncond = uncond_mask = None
    if guidance > 1:
        uncond, uncond_mask = encode_prompt(components, negative or "", latents.device, maximum)
    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    scheduler.set_timesteps(
        steps,
        sigmas=_sigmas(steps),
        mu=calculate_shift(
            latents.shape[1],
            int(components.scheduler_config.get("base_image_seq_len", 256)),
            int(components.scheduler_config.get("max_image_seq_len", 4096)),
            float(components.scheduler_config.get("base_shift", 0.5)),
            float(components.scheduler_config.get("max_shift", 1.15)),
        ),
    )
    scheduler.set_begin_index(0)
    shape = (1, height // 16, width // 16)
    for t in scheduler.timesteps:
        time = (t / int(scheduler.config["num_train_timesteps"])).expand(latents.shape[0]).to(dtype)
        pred = components.transformer(latents.to(dtype), cond.to(dtype), time, shape, cond_mask)
        if guidance > 1:
            assert uncond is not None
            un = components.transformer(
                latents.to(dtype), uncond.to(dtype), time, shape, uncond_mask
            )
            combined = un + guidance * (pred - un)
            pred = combined * (
                torch.norm(pred, dim=-1, keepdim=True) / torch.norm(combined, dim=-1, keepdim=True)
            )
        latents = scheduler.step(-pred, t, latents)
    return latents


class NucleusImageBackend(TextToImageBackend):
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
        self.device = resolve_device(device)
        self.dtype = dtype or torch.float32
        self.model_id = model_id
        self.components: NucleusImageComponents | None = None
        self.snapshot: dict[str, Any] | None = None
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be torch.float32, torch.float16 or torch.bfloat16")

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        unknown = set(inputs.parameters) | set(options) - set(OPTION_DEFAULTS)
        if unknown:
            raise TypeError("Unknown Nucleus-Image options: " + ", ".join(sorted(unknown)))
        result = OPTION_DEFAULTS | inputs.parameters | options
        for name in ("steps", "width", "height", "max_sequence_length"):
            if type(result[name]) is not int or result[name] <= 0:
                raise ValueError(f"{name} must be a positive int")
        if result["width"] % 16 or result["height"] % 16:
            raise ValueError("width and height must be multiples of 16")
        if (
            not isinstance(result["guidance"], int | float)
            or isinstance(result["guidance"], bool)
            or not math.isfinite(result["guidance"])
        ):
            raise ValueError("guidance must be finite")
        return result

    def _load(self) -> None:
        self.components = NucleusImageComponents.load(self.source, self.device, self.dtype)
        self.snapshot = read_snapshot(self.source)

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        seed = o["seed"] if o["seed"] is not None else secrets.randbits(63)
        raw = torch.randn(
            (1, 16, o["height"] // 8, o["width"] // 8),
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        latent = o["latents"] if o["latents"] is not None else pack_latents(raw)
        final = denoise(
            self.components,
            inputs.prompt,
            o["negative_prompt"],
            o["steps"],
            float(o["guidance"]),
            latent.to(self.device),
            o["height"],
            o["width"],
            o["max_sequence_length"],
        )
        decoded = unpack_latents(final, o["height"] // 8, o["width"] // 8)
        channels = int(self.components.vae.config["z_dim"])
        mean = torch.tensor(self.components.vae.config["latents_mean"], device=self.device).view(
            1, channels, 1, 1, 1
        )
        invstd = 1 / torch.tensor(
            self.components.vae.config["latents_std"], device=self.device
        ).view(1, channels, 1, 1, 1)
        pixels = (
            (
                (self.components.vae.decode(decoded / invstd + mean)[:, :, 0] / 2 + 0.5)
                .clamp(0, 1)
                .permute(0, 2, 3, 1)
                .float()
                .cpu()
                * 255
            )
            .round()
            .to(torch.uint8)
        )
        return ImageResult(
            tuple(Image(item) for item in pixels),
            {
                "model_id": (self.snapshot or {}).get("model_id"),
                "revision": (self.snapshot or {}).get("revision"),
                "seed": seed,
                "steps": o["steps"],
                "guidance": float(o["guidance"]),
                "device": str(self.device),
                "dtype": str(self.dtype).removeprefix("torch."),
            },
        )
