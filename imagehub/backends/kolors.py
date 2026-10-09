"""Kolors text-to-image backend over a local diffusers-layout snapshot.

The sampling order (ChatGLM prompt encoding, SDXL-style size conditioning, classifier-free
guidance batch order, Euler noise scaling, VAE decoding and uint8 rounding) follows diffusers'
``KolorsPipeline``. Kolors publishes float16 weights only; the loaders read the ``fp16``
variant file names and the model is run in the requested dtype (float32 by default).
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
from imagehub.backends.stable_diffusion import (
    LATENT_CHANNELS,
    LATENT_SCALE,
    resolve_device,
)
from imagehub.backends.stable_diffusion_xl import _pair, _positive_int
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.chatglm import ChatGLMModel
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.euler import EulerDiscreteScheduler
from imagehub.tokenization_chatglm import ChatGLMTokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "unet", "vae", "scheduler")
MAX_SEQUENCE_LENGTH = 256
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 50,
    "guidance": 5.0,
    "width": None,
    "height": None,
    "num_images": 1,
    "seed": None,
    "latents": None,
    "original_size": None,
    "crops_coords_top_left": None,
    "target_size": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "kolors"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class KolorsComponents:
    tokenizer: ChatGLMTokenizer
    text_encoder: ChatGLMModel
    unet: UNet2DConditionModel
    vae: AutoencoderKL
    scheduler_config: dict[str, Any]
    force_zeros_for_empty_prompt: bool

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> KolorsComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Kolors snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            model_index = read_json(root / "model_index.json")
            scheduler_config = read_json(root / "scheduler" / "scheduler_config.json")
            if scheduler_config.get("_class_name", "EulerDiscreteScheduler") != (
                "EulerDiscreteScheduler"
            ):
                raise ValueError(f"Unsupported Kolors scheduler {scheduler_config['_class_name']}")
            return cls(
                tokenizer=ChatGLMTokenizer(root / "tokenizer" / "tokenizer.model"),
                text_encoder=ChatGLMModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                unet=UNet2DConditionModel.from_pretrained(
                    root / "unet", device=device, dtype=dtype
                ),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=scheduler_config,
                force_zeros_for_empty_prompt=bool(
                    model_index.get("force_zeros_for_empty_prompt", False)
                ),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


@torch.inference_mode()
def encode_prompt(
    components: KolorsComponents, prompt: str, device: torch.device
) -> tuple[Tensor, Tensor]:
    """``(prompt_embeds, pooled)`` of one prompt: ``[1, 256, 4096]`` and ``[1, 4096]``."""
    ids, mask, positions = components.tokenizer([prompt], max_length=MAX_SEQUENCE_LENGTH)
    return components.text_encoder(ids.to(device), positions.to(device), mask.to(device))


@torch.inference_mode()
def denoise(
    components: KolorsComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    latents: Tensor,
    original_size: tuple[int, int],
    crops_coords_top_left: tuple[int, int],
    target_size: tuple[int, int],
) -> Tensor:
    dtype = components.unet.conv_in.weight.dtype
    device = latents.device
    batch = latents.shape[0]
    prompt_embeds, pooled = encode_prompt(components, prompt, device)
    prompt_embeds = prompt_embeds.to(dtype).repeat(batch, 1, 1)
    pooled = pooled.to(dtype).repeat(batch, 1)
    use_cfg = guidance > 1.0
    if use_cfg:
        if negative_prompt is None and components.force_zeros_for_empty_prompt:
            negative_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled = torch.zeros_like(pooled)
        else:
            negative_embeds, negative_pooled = encode_prompt(
                components, negative_prompt or "", device
            )
            negative_embeds = negative_embeds.to(dtype).repeat(batch, 1, 1)
            negative_pooled = negative_pooled.to(dtype).repeat(batch, 1)
        context = torch.cat([negative_embeds, prompt_embeds])
        text_embeds = torch.cat([negative_pooled, pooled])
    else:
        context = prompt_embeds
        text_embeds = pooled

    time_ids = torch.tensor(
        [original_size + crops_coords_top_left + target_size], dtype=dtype, device=device
    )
    if use_cfg:
        time_ids = torch.cat([time_ids, time_ids])
    time_ids = time_ids.repeat(batch, 1)

    scheduler = EulerDiscreteScheduler(components.scheduler_config)
    scheduler.set_timesteps(steps)
    latents = latents.to(dtype) * scheduler.init_noise_sigma
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents] * 2) if use_cfg else latents
        model_input = scheduler.scale_model_input(model_input, timestep)
        noise = components.unet(
            model_input, timestep, context, text_embeds=text_embeds, time_ids=time_ids
        )
        if use_cfg:
            noise_uncond, noise_text = noise.chunk(2)
            noise = noise_uncond + guidance * (noise_text - noise_uncond)
        latents = scheduler.step(noise, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(components: KolorsComponents, latents: Tensor) -> Tensor:
    """Decode final latents to uint8 images of shape (batch, height, width, 3) on CPU.

    A float16 VAE with ``force_upcast`` (the diffusers default) decodes in float32.
    """
    vae = components.vae
    dtype = next(vae.parameters()).dtype
    upcast = dtype == torch.float16 and bool(vae.config.get("force_upcast", True))
    if upcast:
        vae.to(dtype=torch.float32)
        latents = latents.to(torch.float32)
    try:
        image = vae.decode(latents / vae.scaling_factor)
    finally:
        if upcast:
            vae.to(dtype=torch.float16)
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


class KolorsBackend(TextToImageBackend):
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
        self.components: KolorsComponents | None = None
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
            raise TypeError(f"Unknown Kolors options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if o["width"] is None or o["height"] is None:
            try:
                sample_size = int(read_json(self.source / "unet" / "config.json")["sample_size"])
            except (FileNotFoundError, KeyError):
                sample_size = 128
            if o["width"] is None:
                o["width"] = sample_size * LATENT_SCALE
            if o["height"] is None:
                o["height"] = sample_size * LATENT_SCALE
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
            train_steps = 1100
        if o["steps"] > train_steps:
            raise ValueError(f"steps must be at most {train_steps}")
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
        shape = (
            o["num_images"],
            LATENT_CHANNELS,
            o["height"] // LATENT_SCALE,
            o["width"] // LATENT_SCALE,
        )
        if latents is not None:
            if seed is not None:
                raise ValueError("Pass either seed or latents, not both")
            if (
                not isinstance(latents, Tensor)
                or not latents.is_floating_point()
                or tuple(latents.shape) != shape
            ):
                raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        for name in ("original_size", "target_size"):
            o[name] = _pair(o[name], name, positive=True)
        o["crops_coords_top_left"] = _pair(
            o["crops_coords_top_left"], "crops_coords_top_left", positive=False
        )
        return o

    def _load(self) -> None:
        self.components = KolorsComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was "
                "selected; both share the Kolors architecture, so it was loaded",
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
            latents = torch.randn(
                (
                    o["num_images"],
                    LATENT_CHANNELS,
                    o["height"] // LATENT_SCALE,
                    o["width"] // LATENT_SCALE,
                ),
                generator=torch.Generator("cpu").manual_seed(seed),
                dtype=torch.float32,
            )
        size = (o["height"], o["width"])
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
            steps=o["steps"],
            guidance=float(o["guidance"]),
            latents=latents.to(torch.float32).to(self.device),
            original_size=o["original_size"] or size,
            crops_coords_top_left=o["crops_coords_top_left"] or (0, 0),
            target_size=o["target_size"] or size,
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
            "scheduler": "euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(p) for p in pixels), metadata)
