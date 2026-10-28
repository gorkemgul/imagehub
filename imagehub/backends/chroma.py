"""Chroma text-to-image backend over a local diffusers-layout snapshot."""

from __future__ import annotations

import math
import os
import secrets
from collections.abc import Mapping
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
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.chroma_transformer import ChromaTransformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_t5 import T5Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 35,
    "guidance": 5.0,
    "width": 1024,
    "height": 1024,
    "num_images": 1,
    "seed": None,
    "latents": None,
    "negative_prompt": "",
    "max_sequence_length": 512,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "chroma1-hd"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


def calculate_shift(
    image_seq_len: int,
    config: Mapping[str, Any] | None = None,
) -> float:
    config = config or {}
    base_seq_len = int(config.get("base_image_seq_len", 256))
    max_seq_len = int(config.get("max_image_seq_len", 4096))
    base_shift = float(config.get("base_shift", 0.5))
    max_shift = float(config.get("max_shift", 1.15))
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


@dataclass
class ChromaComponents:
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel
    transformer: ChromaTransformer2DModel
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
    ) -> ChromaComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Chroma snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            return cls(
                tokenizer=T5Tokenizer(root / "tokenizer"),
                text_encoder=T5EncoderModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                transformer=ChromaTransformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
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
    components: ChromaComponents,
    prompt: str,
    device: torch.device,
    num_images: int,
    max_sequence_length: int,
) -> tuple[Tensor, Tensor, Tensor]:
    input_ids, tokenizer_mask = components.tokenizer([prompt], max_sequence_length)
    input_ids = input_ids.to(device)
    tokenizer_mask = tokenizer_mask.to(device)
    prompt_embeds = components.text_encoder(input_ids, tokenizer_mask)[0]
    prompt_embeds = prompt_embeds.to(components.text_encoder.shared.weight.dtype)

    seq_lengths = tokenizer_mask.sum(dim=1)
    mask_indices = (
        torch.arange(tokenizer_mask.size(1), device=device)
        .unsqueeze(0)
        .expand(input_ids.shape[0], -1)
    )
    attention_mask = (mask_indices <= seq_lengths.unsqueeze(1)).to(
        dtype=prompt_embeds.dtype, device=device
    )

    prompt_embeds = prompt_embeds.repeat(1, num_images, 1)
    prompt_embeds = prompt_embeds.view(num_images, max_sequence_length, -1)
    attention_mask = attention_mask.repeat(1, num_images).view(num_images, max_sequence_length)
    text_ids = torch.zeros(max_sequence_length, 3, device=device, dtype=prompt_embeds.dtype)
    return prompt_embeds, attention_mask, text_ids


def _prepare_latent_image_ids(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    del batch_size
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    return latent_image_ids.reshape(height * width, 3).to(device=device, dtype=dtype)


def _pack_latents(
    latents: Tensor,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
) -> Tensor:
    latents = latents.view(
        batch_size,
        num_channels_latents,
        height // 2,
        2,
        width // 2,
        2,
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(
        batch_size,
        (height // 2) * (width // 2),
        num_channels_latents * 4,
    )


def _unpack_latents(latents: Tensor, height: int, width: int, vae_scale_factor: int) -> Tensor:
    batch_size, _, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, height, width)


def _prepare_attention_mask(
    attention_mask: Tensor,
    batch_size: int,
    sequence_length: int,
) -> Tensor:
    image_ones = torch.ones(
        batch_size, sequence_length, device=attention_mask.device, dtype=attention_mask.dtype
    )
    return torch.cat([attention_mask, image_ones], dim=1)


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
    components: ChromaComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    guidance: float,
    steps: int,
    width: int,
    height: int,
    latents: Tensor,
    max_sequence_length: int,
) -> Tensor:
    dtype = cast(torch.dtype, components.transformer.x_embedder.weight.dtype)
    batch_size, channels, latent_height, latent_width = latents.shape
    prompt_embeds, attention_mask, text_ids = encode_prompt(
        components, prompt, latents.device, batch_size, max_sequence_length
    )
    prompt_embeds = prompt_embeds.to(dtype)
    attention_mask = attention_mask.to(dtype)
    text_ids = text_ids.to(dtype)

    do_cfg = guidance > 1 and negative_prompt is not None
    negative_embeds = negative_attention_mask = negative_text_ids = None
    if do_cfg:
        negative_embeds, negative_attention_mask, negative_text_ids = encode_prompt(
            components, negative_prompt or "", latents.device, batch_size, max_sequence_length
        )
        negative_embeds = negative_embeds.to(dtype)
        negative_attention_mask = negative_attention_mask.to(dtype)
        negative_text_ids = negative_text_ids.to(dtype)

    latents = _pack_latents(latents, batch_size, channels, latent_height, latent_width)
    latent_image_ids = _prepare_latent_image_ids(
        batch_size, latent_height // 2, latent_width // 2, latents.device, latents.dtype
    )
    image_seq_len = latents.shape[1]
    attention_mask = _prepare_attention_mask(attention_mask, batch_size, image_seq_len)
    if do_cfg:
        assert negative_attention_mask is not None
        negative_attention_mask = _prepare_attention_mask(
            negative_attention_mask, batch_size, image_seq_len
        )

    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    scheduler.set_timesteps(
        steps,
        sigmas=_sigmas(steps),
        mu=calculate_shift(image_seq_len, components.scheduler_config),
    )
    scheduler.set_begin_index(0)

    for timestep in scheduler.timesteps:
        model_timestep = timestep.expand(batch_size).to(latents.dtype) / 1000
        noise_pred = components.transformer(
            latents.to(dtype),
            prompt_embeds,
            model_timestep,
            latent_image_ids,
            text_ids,
            attention_mask,
        )
        if do_cfg:
            assert (
                negative_embeds is not None
                and negative_attention_mask is not None
                and negative_text_ids is not None
            )
            negative_pred = components.transformer(
                latents.to(dtype),
                negative_embeds,
                model_timestep,
                latent_image_ids,
                negative_text_ids,
                negative_attention_mask,
            )
            noise_pred = negative_pred + guidance * (noise_pred - negative_pred)
        latents = scheduler.step(noise_pred, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: Tensor, height: int, width: int) -> Tensor:
    vae_scale_factor = 2 ** (len(vae.config["block_out_channels"]) - 1)
    latents = _unpack_latents(latents, height, width, vae_scale_factor)
    latents = latents / vae.scaling_factor + (vae.shift_factor or 0.0)
    image = vae.decode(latents.to(vae.decoder.conv_in.weight.dtype))
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class ChromaBackend(TextToImageBackend):
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
        self.components: ChromaComponents | None = None
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults

    def _latent_shape(self, options: dict[str, Any]) -> tuple[int, int, int, int]:
        packed_channels = 64
        vae_scale = 8
        transformer_config = self.source / "transformer" / "config.json"
        vae_config = self.source / "vae" / "config.json"
        if transformer_config.is_file():
            packed_channels = int(read_json(transformer_config).get("in_channels", packed_channels))
        if vae_config.is_file():
            blocks = read_json(vae_config).get("block_out_channels", [1, 1, 1, 1])
            vae_scale = 2 ** (len(blocks) - 1)
        return (
            options["num_images"],
            packed_channels // 4,
            options["height"] // vae_scale,
            options["width"] // vae_scale,
        )

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown Chroma options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        for name in ("steps", "num_images", "width", "height", "max_sequence_length"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        if o["max_sequence_length"] > 512:
            raise ValueError("max_sequence_length must be at most 512")
        for name in ("width", "height"):
            if o[name] % 16:
                raise ValueError(f"{name} must be a multiple of 16")
        value = o["guidance"]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("guidance must be a finite number >= 0")
        if o["negative_prompt"] is not None and not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
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
        o["latent_shape"] = shape
        return o

    def _load(self) -> None:
        self.components = ChromaComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            raise CheckpointError(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected"
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
                o["latent_shape"],
                generator=torch.Generator("cpu").manual_seed(seed),
                dtype=torch.float32,
            )
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
            guidance=float(o["guidance"]),
            steps=o["steps"],
            width=o["width"],
            height=o["height"],
            latents=latents.to(torch.float32).to(self.device),
            max_sequence_length=o["max_sequence_length"],
        )
        pixels = decode_latents(self.components.vae, final, o["height"], o["width"])
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": o["steps"],
            "guidance": float(o["guidance"]),
            "negative_prompt": o["negative_prompt"],
            "width": o["width"],
            "height": o["height"],
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
