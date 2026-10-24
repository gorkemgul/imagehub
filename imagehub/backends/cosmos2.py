"""Cosmos-Predict2 text-to-image backend over a local diffusers snapshot."""

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
from imagehub.models.autoencoder_kl_wan import AutoencoderKLWan
from imagehub.models.cosmos_transformer import CosmosTransformer3DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_t5 import T5Tokenizer
from imagehub.types import ImageResult, T2IRequest

DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated "
    "images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed "
    "out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, "
    "unnatural transitions, outdated special effects, fake elements, unconvincing visuals, "
    "poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)
COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 35,
    "guidance": 7.0,
    "width": 1360,
    "height": 768,
    "seed": None,
    "max_sequence_length": 512,
    "latents": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@dataclass
class Cosmos2Components:
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel
    transformer: CosmosTransformer3DModel
    vae: AutoencoderKLWan
    scheduler_config: dict[str, Any]

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> Cosmos2Components:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            model = model_id or "cosmos-predict2-2b-t2i"
            raise CheckpointError(
                f"{root} is not a Cosmos-Predict2 snapshot (missing {', '.join(missing)}). "
                f"Download one with: python -m imagehub.pretrained {model} {root}"
            )
        class_name = read_json(root / "model_index.json").get("_class_name")
        if class_name != "Cosmos2TextToImagePipeline":
            raise CheckpointError(f"Unsupported Cosmos-Predict2 pipeline class {class_name!r}")
        return cls(
            T5Tokenizer(root / "tokenizer"),
            T5EncoderModel.from_pretrained(root / "text_encoder", device=device, dtype=dtype),
            CosmosTransformer3DModel.from_pretrained(
                root / "transformer", device=device, dtype=dtype
            ),
            AutoencoderKLWan.from_pretrained(root / "vae", device=device, dtype=dtype),
            read_json(root / "scheduler" / "scheduler_config.json"),
        )


def _embeds(
    components: Cosmos2Components, prompt: str, length: int, device: torch.device
) -> Tensor:
    ids, mask = components.tokenizer([prompt], length)
    embeds = components.text_encoder(ids.to(device), mask.to(device))[0]
    return embeds.masked_fill(~mask.to(device).unsqueeze(-1), 0)


@torch.inference_mode()
def denoise(
    components: Cosmos2Components,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    max_sequence_length: int,
    height: int,
    width: int,
    latents: Tensor,
) -> Tensor:
    dtype = components.transformer.patch_embed.proj.weight.dtype
    prompt_embeds = _embeds(components, prompt, max_sequence_length, latents.device).to(dtype)
    negative_embeds = None
    if guidance > 1:
        negative_embeds = _embeds(
            components,
            negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            max_sequence_length,
            latents.device,
        ).to(dtype)
    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    sigmas = torch.linspace(0, 1, steps, dtype=torch.float64).tolist()
    scheduler.set_timesteps(steps, sigmas=sigmas)
    if scheduler.config["final_sigmas_type"] == "sigma_min":
        scheduler.sigmas[-1] = scheduler.sigmas[-2]
    padding_mask = latents.new_zeros(1, 1, height, width, dtype=dtype)
    latents = latents.to(torch.float32) * float(scheduler.config["sigma_max"])
    for index, timestep in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[index]
        current = sigma / (sigma + 1)
        c_in, c_skip, c_out = 1 - current, 1 - current, -current
        model_input = (latents * c_in).to(dtype)
        time = current.expand(latents.shape[0]).to(dtype)
        conditional = components.transformer(
            model_input, time, prompt_embeds, padding_mask=padding_mask
        )
        prediction = (c_skip * latents + c_out * conditional.float()).to(dtype)
        if negative_embeds is not None:
            unconditional = components.transformer(
                model_input, time, negative_embeds, padding_mask=padding_mask
            )
            unconditional = (c_skip * latents + c_out * unconditional.float()).to(dtype)
            prediction = prediction + guidance * (prediction - unconditional)
        velocity = (latents - prediction) / sigma
        latents = scheduler.step(velocity, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(components: Cosmos2Components, latents: Tensor) -> Tensor:
    mean = torch.tensor(components.vae.latents_mean).view(1, -1, 1, 1).to(latents)
    std = torch.tensor(components.vae.latents_std).view(1, -1, 1, 1).to(latents)
    latents = latents[:, :, 0] * std / float(components.scheduler_config["sigma_data"]) + mean
    image = components.vae.decode(latents.to(components.vae.decoder.conv_in.weight.dtype))
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).to(torch.uint8)


class Cosmos2Backend(TextToImageBackend):
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
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults
        self.components: Cosmos2Components | None = None
        self.snapshot: dict[str, Any] | None = None

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown Cosmos-Predict2 options: {', '.join(unknown)}")
        values = self.option_defaults | merged
        if values["negative_prompt"] is not None and not isinstance(values["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "width", "height", "max_sequence_length"):
            if type(values[name]) is not int or values[name] <= 0:
                raise ValueError(f"{name} must be a positive int")
        if values["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        if values["width"] % 16 or values["height"] % 16:
            raise ValueError("width and height must be multiples of 16")
        guidance = values["guidance"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, int | float)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError("guidance must be a finite number >= 0")
        seed, latents = values["seed"], values["latents"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        shape = (1, 16, 1, values["height"] // 8, values["width"] // 8)
        if latents is not None:
            if seed is not None:
                raise ValueError("Pass either seed or latents, not both")
            if (
                not isinstance(latents, Tensor)
                or not latents.is_floating_point()
                or tuple(latents.shape) != shape
            ):
                raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        values["latent_shape"] = shape
        return values

    def _load(self) -> None:
        self.components = Cosmos2Components.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r}, not {self.model_id!r}",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed, latents = options["seed"], options["latents"]
        if latents is None:
            seed = secrets.randbits(63) if seed is None else seed
            latents = torch.randn(
                options["latent_shape"],
                generator=torch.Generator("cpu").manual_seed(seed),
                dtype=torch.float32,
            )
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=options["negative_prompt"],
            steps=options["steps"],
            guidance=float(options["guidance"]),
            max_sequence_length=options["max_sequence_length"],
            height=options["height"],
            width=options["width"],
            latents=latents.to(self.device),
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
            "scheduler": "flow_match_euler_karras",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
