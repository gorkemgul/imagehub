"""CogView4 text-to-image backend over a local diffusers-layout snapshot."""

from __future__ import annotations

import math
import os
import secrets
import warnings
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
from imagehub.models.cogview4_transformer import CogView4Transformer2DModel
from imagehub.models.glm import GlmModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_glm import GlmTokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 50,
    "guidance": 5.0,
    "width": 1024,
    "height": 1024,
    "seed": None,
    "negative_prompt": None,
    "max_sequence_length": 1024,
    "latents": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def calculate_shift(
    image_seq_len: int,
    config: Mapping[str, Any] | None = None,
) -> float:
    config = config or {}
    base_seq_len = int(config.get("base_image_seq_len", 256))
    base_shift = float(config.get("base_shift", 0.25))
    max_shift = float(config.get("max_shift", 0.75))
    m = (image_seq_len / base_seq_len) ** 0.5
    return m * max_shift + base_shift


def cogview4_sigmas(steps: int, num_train_timesteps: int) -> list[float]:
    if steps == 1:
        timesteps = [num_train_timesteps]
    else:
        step = (1.0 - num_train_timesteps) / (steps - 1)
        timesteps = [int(num_train_timesteps + index * step) for index in range(steps)]
        timesteps[-1] = 1
    return (torch.tensor(timesteps, dtype=torch.float32) / num_train_timesteps).tolist()


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "cogview4-6b"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class CogView4Components:
    tokenizer: GlmTokenizer
    text_encoder: GlmModel
    transformer: CogView4Transformer2DModel
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
    ) -> CogView4Components:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a CogView4 snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            return cls(
                tokenizer=GlmTokenizer(root / "tokenizer"),
                text_encoder=GlmModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                transformer=CogView4Transformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _glm_embeds(
    components: CogView4Components,
    prompt: str,
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    input_ids, _ = components.tokenizer([prompt], max_sequence_length)
    return components.text_encoder(input_ids.to(device)).to(dtype=dtype, device=device)


@torch.inference_mode()
def encode_prompt(
    components: CogView4Components,
    prompt: str,
    negative_prompt: str | None,
    max_sequence_length: int,
    device: torch.device,
    do_classifier_free_guidance: bool,
) -> tuple[Tensor, Tensor | None]:
    dtype = cast(torch.dtype, components.text_encoder.embed_tokens.weight.dtype)
    prompt_embeds = _glm_embeds(components, prompt, max_sequence_length, device, dtype)
    negative_prompt_embeds = None
    if do_classifier_free_guidance:
        negative_prompt_embeds = _glm_embeds(
            components, negative_prompt or "", max_sequence_length, device, dtype
        )
    return prompt_embeds, negative_prompt_embeds


@torch.inference_mode()
def denoise(
    components: CogView4Components,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    latents: Tensor,
    max_sequence_length: int,
) -> Tensor:
    use_cfg = guidance > 1
    prompt_embeds, negative_prompt_embeds = encode_prompt(
        components,
        prompt,
        negative_prompt,
        max_sequence_length,
        latents.device,
        use_cfg,
    )
    original_size = torch.tensor(
        [[height, width]], dtype=prompt_embeds.dtype, device=latents.device
    )
    target_size = torch.tensor([[height, width]], dtype=prompt_embeds.dtype, device=latents.device)
    crop_coords = torch.tensor([[0, 0]], dtype=prompt_embeds.dtype, device=latents.device)
    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    num_train_timesteps = int(scheduler.config["num_train_timesteps"])
    sigmas = cogview4_sigmas(steps, num_train_timesteps)
    vae_scale_factor = 2 ** (len(components.vae.config["block_out_channels"]) - 1)
    patch_size = components.transformer.patch_size
    image_seq_len = ((height // vae_scale_factor) * (width // vae_scale_factor)) // (patch_size**2)
    scheduler.set_timesteps(
        steps,
        sigmas=sigmas,
        mu=calculate_shift(image_seq_len, components.scheduler_config),
    )
    transformer_dtype = cast(torch.dtype, components.transformer.patch_embed.proj.weight.dtype)
    for timestep in scheduler.timesteps:
        latent_model_input = latents.to(transformer_dtype)
        current_timestep = timestep.expand(latents.shape[0])
        noise_pred_cond = components.transformer(
            latent_model_input,
            prompt_embeds,
            current_timestep,
            original_size,
            target_size,
            crop_coords,
        )
        if use_cfg:
            assert negative_prompt_embeds is not None
            noise_pred_uncond = components.transformer(
                latent_model_input,
                negative_prompt_embeds,
                current_timestep,
                original_size,
                target_size,
                crop_coords,
            )
            noise_pred = noise_pred_uncond + guidance * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond
        latents = scheduler.step(noise_pred, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: Tensor) -> Tensor:
    latents = latents.to(vae.decoder.conv_in.weight.dtype) / vae.scaling_factor
    image = vae.decode(latents)
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class CogView4Backend(TextToImageBackend):
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
        self.components: CogView4Components | None = None
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults

    def _latent_shape(self, options: dict[str, Any]) -> tuple[int, int, int, int]:
        channels = 16
        vae_scale = 8
        transformer_config = self.source / "transformer" / "config.json"
        vae_config = self.source / "vae" / "config.json"
        if transformer_config.is_file():
            channels = int(read_json(transformer_config).get("in_channels", channels))
        if vae_config.is_file():
            blocks = read_json(vae_config).get("block_out_channels", [1, 1, 1, 1])
            vae_scale = 2 ** (len(blocks) - 1)
        return (1, channels, options["height"] // vae_scale, options["width"] // vae_scale)

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown CogView4 options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if o["negative_prompt"] is not None and not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "width", "height", "max_sequence_length"):
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
        o["latent_shape"] = shape
        return o

    def _load(self) -> None:
        self.components = CogView4Components.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the CogView4 architecture, so it was loaded",
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
                o["latent_shape"],
                generator=torch.Generator("cpu").manual_seed(seed),
                dtype=torch.float32,
            )
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
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
            "negative_prompt": o["negative_prompt"],
            "width": o["width"],
            "height": o["height"],
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
