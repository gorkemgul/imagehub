"""Stable Diffusion XL text-to-image backend over a local diffusers-layout snapshot."""

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
    decode_latents,
    resolve_device,
)
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel, CLIPTextModelWithProjection
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.edm_dpmsolver import EDMDPMSolverMultistepScheduler
from imagehub.schedulers.euler import EulerDiscreteScheduler
from imagehub.schedulers.euler_ancestral import EulerAncestralDiscreteScheduler
from imagehub.tokenization import CLIPTokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = (
    "tokenizer",
    "tokenizer_2",
    "text_encoder",
    "text_encoder_2",
    "unet",
    "vae",
    "scheduler",
)
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


def _scheduler(config: dict[str, Any]) -> tuple[Any, str]:
    class_name = config.get("_class_name", "EulerDiscreteScheduler")
    if class_name == "EulerDiscreteScheduler":
        return EulerDiscreteScheduler(config), "euler"
    if class_name == "EulerAncestralDiscreteScheduler":
        return EulerAncestralDiscreteScheduler(config), "euler_ancestral"
    if class_name == "EDMDPMSolverMultistepScheduler":
        return EDMDPMSolverMultistepScheduler(config), "edm_dpmsolver_multistep"
    raise ValueError(f"Unsupported SDXL scheduler {class_name}")


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "sdxl-base-1.0"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class StableDiffusionXLComponents:
    tokenizer: CLIPTokenizer
    tokenizer_2: CLIPTokenizer
    text_encoder: CLIPTextModel
    text_encoder_2: CLIPTextModelWithProjection
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
    ) -> StableDiffusionXLComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Stable Diffusion XL snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            model_index = read_json(root / "model_index.json")
            return cls(
                tokenizer=CLIPTokenizer(root / "tokenizer"),
                tokenizer_2=CLIPTokenizer(root / "tokenizer_2"),
                text_encoder=CLIPTextModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                text_encoder_2=CLIPTextModelWithProjection.from_pretrained(
                    root / "text_encoder_2", device=device, dtype=dtype
                ),
                unet=UNet2DConditionModel.from_pretrained(
                    root / "unet", device=device, dtype=dtype
                ),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
                force_zeros_for_empty_prompt=bool(
                    model_index.get("force_zeros_for_empty_prompt", True)
                ),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


@torch.inference_mode()
def encode_prompt(
    components: StableDiffusionXLComponents, prompt: str, device: torch.device
) -> tuple[Tensor, Tensor]:
    ids_1 = torch.tensor([components.tokenizer(prompt)], device=device)
    ids_2 = torch.tensor([components.tokenizer_2(prompt)], device=device)
    _, hidden_1 = components.text_encoder(ids_1, output_hidden_states=True)
    pooled, _, hidden_2 = components.text_encoder_2(ids_2, output_hidden_states=True)
    return torch.cat([hidden_1[-2], hidden_2[-2]], dim=-1), pooled


@torch.inference_mode()
def denoise(
    components: StableDiffusionXLComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    latents: Tensor,
    original_size: tuple[int, int],
    crops_coords_top_left: tuple[int, int],
    target_size: tuple[int, int],
    generator: torch.Generator | None = None,
) -> Tensor:
    dtype = components.unet.conv_in.weight.dtype
    batch = latents.shape[0]
    prompt_embeds, pooled = encode_prompt(components, prompt, latents.device)
    prompt_embeds = prompt_embeds.to(dtype).repeat(batch, 1, 1)
    pooled = pooled.to(dtype).repeat(batch, 1)
    use_cfg = guidance > 1.0
    if use_cfg:
        if negative_prompt is None and components.force_zeros_for_empty_prompt:
            negative_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled = torch.zeros_like(pooled)
        else:
            negative_embeds, negative_pooled = encode_prompt(
                components, negative_prompt or "", latents.device
            )
            negative_embeds = negative_embeds.to(dtype).repeat(batch, 1, 1)
            negative_pooled = negative_pooled.to(dtype).repeat(batch, 1)
        context = torch.cat([negative_embeds, prompt_embeds])
        text_embeds = torch.cat([negative_pooled, pooled])
    else:
        context = prompt_embeds
        text_embeds = pooled

    time_ids = torch.tensor(
        [original_size + crops_coords_top_left + target_size], dtype=dtype, device=latents.device
    )
    if use_cfg:
        time_ids = torch.cat([time_ids, time_ids])
    time_ids = time_ids.repeat(batch, 1)

    scheduler, _ = _scheduler(components.scheduler_config)
    scheduler.set_timesteps(steps)
    latents = latents.to(dtype) * scheduler.init_noise_sigma
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents] * 2) if use_cfg else latents
        model_input = scheduler.scale_model_input(model_input, timestep)
        noise = components.unet(
            model_input,
            timestep,
            context,
            text_embeds=text_embeds,
            time_ids=time_ids,
        )
        if use_cfg:
            noise_uncond, noise_text = noise.chunk(2)
            noise = noise_uncond + guidance * (noise_text - noise_uncond)
        if isinstance(scheduler, EulerAncestralDiscreteScheduler):
            latents = scheduler.step(noise, timestep, latents, generator=generator)
        else:
            latents = scheduler.step(noise, timestep, latents)
    return latents


def _positive_int(options: dict[str, Any], name: str) -> None:
    value = options[name]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")


def _pair(value: Any, name: str, *, positive: bool) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"{name} must be a pair of {qualifier}ints")
    valid_values = all(
        type(item) is int and (item > 0 if positive else item >= 0) for item in value
    )
    if not valid_values:
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"{name} must be a pair of {qualifier}ints")
    return value


class StableDiffusionXLBackend(TextToImageBackend):
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
        self.components: StableDiffusionXLComponents | None = None
        self.snapshot: dict[str, Any] | None = None
        model_defaults = {}
        if model_id is not None and model_id in MODEL_CATALOG:
            model_defaults = dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
        self.option_defaults = OPTION_DEFAULTS | model_defaults

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown Stable Diffusion XL options: {', '.join(unknown)}")
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
        self.components = StableDiffusionXLComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was "
                "selected; both share the Stable Diffusion XL architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        seed, latents = o["seed"], o["latents"]
        generator = None
        if latents is None:
            if seed is None:
                seed = secrets.randbits(63)
            generator = torch.Generator("cpu").manual_seed(seed)
            latents = torch.randn(
                (
                    o["num_images"],
                    LATENT_CHANNELS,
                    o["height"] // LATENT_SCALE,
                    o["width"] // LATENT_SCALE,
                ),
                generator=generator,
                dtype=torch.float32,
            )
        # As in diffusers, caller-provided latents without a seed leave ancestral noise on
        # the global RNG by passing generator=None.
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
            generator=generator,
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
            "scheduler": _scheduler(self.components.scheduler_config)[1],
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(p) for p in pixels), metadata)
