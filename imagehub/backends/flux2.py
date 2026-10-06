"""FLUX.2 klein text-to-image backend over a local diffusers-layout snapshot."""

from __future__ import annotations

import math
import os
import secrets
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl_flux2 import (
    AutoencoderKLFlux2,
    denormalize_latents,
    unpatchify_latents,
)
from imagehub.models.flux2_transformer import Flux2Transformer2DModel
from imagehub.models.qwen3 import Qwen3Model
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 4,
    "guidance": 1.0,
    "width": 1024,
    "height": 1024,
    "num_images": 1,
    "seed": None,
    "latents": None,
    "max_sequence_length": 512,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "flux2-klein-4b"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class Flux2Components:
    tokenizer: Qwen2Tokenizer
    text_encoder: Qwen3Model
    transformer: Flux2Transformer2DModel
    vae: AutoencoderKLFlux2
    scheduler_config: dict[str, Any]
    is_distilled: bool

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> Flux2Components:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a FLUX.2 snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            model_index = read_json(root / "model_index.json")
            return cls(
                tokenizer=Qwen2Tokenizer(root / "tokenizer"),
                text_encoder=Qwen3Model.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                transformer=Flux2Transformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKLFlux2.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
                is_distilled=bool(model_index.get("is_distilled", False)),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _text_ids(prompt_embeds: Tensor) -> Tensor:
    batch, length, _ = prompt_embeds.shape
    coordinates = torch.cartesian_prod(
        torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(length)
    )
    return coordinates.unsqueeze(0).expand(batch, -1, -1)


def _latent_ids(latents: Tensor) -> Tensor:
    batch, _, height, width = latents.shape
    coordinates = torch.cartesian_prod(
        torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(1)
    )
    return coordinates.unsqueeze(0).expand(batch, -1, -1)


@torch.inference_mode()
def encode_prompt(
    components: Flux2Components,
    prompt: str,
    device: torch.device,
    num_images: int,
    max_sequence_length: int,
) -> tuple[Tensor, Tensor]:
    text = components.tokenizer.chat_prompt(prompt)
    input_ids, attention_mask = components.tokenizer(text, max_sequence_length)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    _, hidden_states = components.text_encoder(input_ids, attention_mask, output_hidden_states=True)
    selected = torch.stack([hidden_states[index] for index in (9, 18, 27)], dim=1)
    batch_size, channels, sequence_length, hidden_size = selected.shape
    prompt_embeds = selected.permute(0, 2, 1, 3).reshape(
        batch_size, sequence_length, channels * hidden_size
    )
    prompt_embeds = prompt_embeds.repeat(1, num_images, 1).view(
        batch_size * num_images, sequence_length, -1
    )
    return prompt_embeds, _text_ids(prompt_embeds).to(device)


def _sigmas(steps: int) -> list[float]:
    if steps == 1:
        return [1.0]
    stop = 1.0 / steps
    step = (stop - 1.0) / (steps - 1)
    values = [1.0 + index * step for index in range(steps)]
    values[-1] = stop
    return values


@torch.inference_mode()
def denoise(
    components: Flux2Components,
    *,
    prompt: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    latents: Tensor,
    max_sequence_length: int,
) -> Tensor:
    dtype = cast(torch.dtype, components.transformer.x_embedder.weight.dtype)
    batch_size = latents.shape[0]
    prompt_embeds, text_ids = encode_prompt(
        components, prompt, latents.device, batch_size, max_sequence_length
    )
    prompt_embeds = prompt_embeds.to(dtype)
    use_cfg = guidance > 1 and not components.is_distilled
    negative_embeds = negative_text_ids = None
    if use_cfg:
        negative_embeds, negative_text_ids = encode_prompt(
            components, "", latents.device, batch_size, max_sequence_length
        )
        negative_embeds = negative_embeds.to(dtype)

    latent_ids = _latent_ids(latents).to(latents.device)
    latent_height, latent_width = latents.shape[-2:]
    latents = latents.reshape(batch_size, latents.shape[1], -1).permute(0, 2, 1)
    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    sigmas = _sigmas(steps)
    scheduler.set_timesteps(
        steps,
        sigmas=sigmas,
        mu=compute_empirical_mu(latents.shape[1], steps),
    )
    scheduler.set_begin_index(0)
    for timestep in scheduler.timesteps:
        model_timestep = timestep.expand(batch_size).to(latents.dtype) / 1000
        noise_pred = components.transformer(
            latents.to(dtype),
            prompt_embeds,
            model_timestep,
            latent_ids,
            text_ids,
        )
        if use_cfg:
            assert negative_embeds is not None and negative_text_ids is not None
            negative_pred = components.transformer(
                latents.to(dtype),
                negative_embeds,
                model_timestep,
                latent_ids,
                negative_text_ids,
            )
            noise_pred = negative_pred + guidance * (noise_pred - negative_pred)
        latents = scheduler.step(noise_pred, timestep, latents)

    latents = latents.permute(0, 2, 1).reshape(
        batch_size, latents.shape[-1], latent_height, latent_width
    )
    assert components.vae.bn.running_mean is not None
    assert components.vae.bn.running_var is not None
    latents = denormalize_latents(
        latents,
        components.vae.bn.running_mean,
        components.vae.bn.running_var,
        components.vae.batch_norm_eps,
    )
    return unpatchify_latents(latents)


@torch.inference_mode()
def decode_latents(vae: AutoencoderKLFlux2, latents: Tensor) -> Tensor:
    dtype = vae.decoder.conv_in.weight.dtype
    image = vae.decode(latents.to(dtype))
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class Flux2Backend(TextToImageBackend):
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
        self.components: Flux2Components | None = None
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults

    def _latent_shape(self, options: dict[str, Any]) -> tuple[int, int, int, int]:
        channels = 128
        scale = 16
        transformer_config = self.source / "transformer" / "config.json"
        vae_config = self.source / "vae" / "config.json"
        if transformer_config.is_file():
            channels = int(read_json(transformer_config).get("in_channels", channels))
        if vae_config.is_file():
            blocks = read_json(vae_config).get("block_out_channels", [1, 1, 1, 1])
            scale = 2 ** (len(blocks) - 1) * 2
        return (
            options["num_images"],
            channels,
            options["height"] // scale,
            options["width"] // scale,
        )

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown FLUX.2 options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        for name in ("steps", "num_images", "width", "height", "max_sequence_length"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        for name in ("width", "height"):
            if o[name] % 16:
                raise ValueError(f"{name} must be a multiple of 16")
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
        shape = self._latent_shape(o)
        if latents is not None:
            if seed is not None:
                raise ValueError("Pass either seed or latents, not both")
            if (
                not isinstance(latents, Tensor)
                or not latents.is_floating_point()
                or tuple(latents.shape) != shape
            ):
                raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        return o

    def _load(self) -> None:
        self.components = Flux2Components.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the FLUX.2 architecture, so it was loaded",
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
                self._latent_shape(o),
                generator=torch.Generator("cpu").manual_seed(seed),
                dtype=torch.float32,
            )
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            steps=o["steps"],
            guidance=float(o["guidance"]),
            width=o["width"],
            height=o["height"],
            latents=latents.to(torch.float32).to(self.device),
            max_sequence_length=o["max_sequence_length"],
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
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
