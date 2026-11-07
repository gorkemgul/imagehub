"""Ideogram 4 text-to-image backend over a local diffusers-layout snapshot."""

from __future__ import annotations

import math
import os
import secrets
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.caption_verifier import CaptionVerifier
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import load_weights, read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.lora_formats import LoRAOption, apply_lora, parse_lora_option
from imagehub.models.autoencoder_ideogram4 import AutoencoderIdeogram4, get_latent_norm
from imagehub.models.fp8 import load_fp8_state_dict, swap_linears_to_fp8
from imagehub.models.ideogram4_text import build_text_inputs, encode_text
from imagehub.models.ideogram4_transformer import Ideogram4Transformer
from imagehub.models.qwen3_vl_text import Qwen3VLTextModel
from imagehub.schedulers.ideogram4 import (
    PRESETS,
    get_schedule_for_resolution,
    ideogram4_step,
    make_step_intervals,
)
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = (
    "tokenizer",
    "text_encoder",
    "transformer",
    "unconditional_transformer",
    "vae",
    "scheduler",
)
OPTION_DEFAULTS: dict[str, Any] = {
    "preset": "V4_QUALITY_48",
    "steps": None,
    "guidance": None,
    "guidance_schedule": None,
    "width": 1024,
    "height": 1024,
    "seed": None,
    "latents": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "ideogram4-fp8"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


def _load_fp8_transformer(
    directory: Path, device: torch.device, dtype: torch.dtype
) -> Ideogram4Transformer:
    state = load_weights(directory)
    with torch.device("meta"):
        model = Ideogram4Transformer(read_json(directory / "config.json"))
        swap_linears_to_fp8(model, state, compute_dtype=dtype)
    model.init_non_persistent_buffers()
    load_fp8_state_dict(model, state, device, dtype, assign=True)
    return model.eval()


def _load_fp8_text_encoder(
    directory: Path, device: torch.device, dtype: torch.dtype
) -> Qwen3VLTextModel:
    state = load_weights(directory)
    config = read_json(directory / "config.json")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, Mapping):
        raise CheckpointError(f"Expected text_config in {directory / 'config.json'}")
    with torch.device("meta"):
        model = Qwen3VLTextModel(text_config)
        prepared = model.prepare_state_dict(state)
        swap_linears_to_fp8(model, prepared, compute_dtype=dtype)
    model.init_non_persistent_buffers()
    load_fp8_state_dict(model, prepared, device, dtype, assign=True)
    return model.eval()


@dataclass
class Ideogram4Components:
    tokenizer: Qwen2Tokenizer
    text_encoder: Qwen3VLTextModel
    conditional_transformer: Ideogram4Transformer
    unconditional_transformer: Ideogram4Transformer
    vae: AutoencoderIdeogram4
    latent_shift: Tensor
    latent_scale: Tensor
    patch_size: int = 2
    ae_scale_factor: int = 8
    max_text_tokens: int = 2048

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> Ideogram4Components:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not an Ideogram 4 snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            shift, scale = get_latent_norm()
            return cls(
                tokenizer=Qwen2Tokenizer(root / "tokenizer"),
                text_encoder=_load_fp8_text_encoder(root / "text_encoder", device, dtype),
                conditional_transformer=_load_fp8_transformer(root / "transformer", device, dtype),
                unconditional_transformer=_load_fp8_transformer(
                    root / "unconditional_transformer", device, dtype
                ),
                vae=AutoencoderIdeogram4.from_pretrained(root / "vae", device=device, dtype=dtype),
                latent_shift=shift.to(device),
                latent_scale=scale.to(device),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _verify_prompt(prompt: str) -> None:
    issues = CaptionVerifier().verify_raw(prompt)
    if issues:
        raise ValueError("caption verifier flagged prompt[0]:\n" + "\n".join(issues))


@torch.inference_mode()
def denoise(
    components: Ideogram4Components,
    *,
    prompt: str,
    steps: int,
    guidance_schedule: Sequence[float],
    mu: float,
    std: float,
    width: int,
    height: int,
    latents: Tensor,
) -> Tensor:
    _verify_prompt(prompt)
    inputs = build_text_inputs(
        components.tokenizer,
        [prompt],
        height,
        width,
        device=latents.device,
        patch_size=components.patch_size,
        ae_scale_factor=components.ae_scale_factor,
        max_text_tokens=components.max_text_tokens,
    )
    token_ids = cast(Tensor, inputs["token_ids"])
    text_position_ids = cast(Tensor, inputs["text_position_ids"])
    position_ids = cast(Tensor, inputs["position_ids"])
    segment_ids = cast(Tensor, inputs["segment_ids"])
    indicator = cast(Tensor, inputs["indicator"])
    llm_features = encode_text(
        components.text_encoder,
        token_ids,
        text_position_ids,
        indicator,
    )
    max_text_tokens = int(inputs["max_text_tokens"])
    num_image_tokens = int(inputs["num_image_tokens"])
    negative_position_ids = position_ids[:, max_text_tokens:]
    negative_segment_ids = segment_ids[:, max_text_tokens:]
    negative_indicator = indicator[:, max_text_tokens:]
    negative_features = torch.zeros(
        latents.shape[0],
        num_image_tokens,
        llm_features.shape[-1],
        dtype=llm_features.dtype,
        device=latents.device,
    )
    text_padding = torch.zeros(
        latents.shape[0],
        max_text_tokens,
        latents.shape[-1],
        dtype=torch.float32,
        device=latents.device,
    )
    schedule = get_schedule_for_resolution((height, width), known_mean=mu, std=std)
    intervals = make_step_intervals(steps).to(latents.device)
    for index in range(steps - 1, -1, -1):
        t_value = float(schedule(intervals[index + 1].unsqueeze(0)).item())
        s_value = float(schedule(intervals[index].unsqueeze(0)).item())
        timestep = torch.full(
            (latents.shape[0],), t_value, dtype=torch.float32, device=latents.device
        )
        positive = components.conditional_transformer(
            llm_features=llm_features,
            x=torch.cat([text_padding, latents], dim=1),
            t=timestep,
            position_ids=position_ids,
            segment_ids=segment_ids,
            indicator=indicator,
        )[:, max_text_tokens:]
        negative = components.unconditional_transformer(
            llm_features=negative_features,
            x=latents,
            t=timestep,
            position_ids=negative_position_ids,
            segment_ids=negative_segment_ids,
            indicator=negative_indicator,
        )
        latents = ideogram4_step(
            latents,
            positive,
            negative,
            guidance_schedule[index],
            t_value,
            s_value,
        )
    return latents


@torch.inference_mode()
def decode_latents(
    components: Ideogram4Components, latents: Tensor, height: int, width: int
) -> Tensor:
    batch_size = latents.shape[0]
    patch = components.patch_size
    grid_height = height // (components.ae_scale_factor * patch)
    grid_width = width // (components.ae_scale_factor * patch)
    latents = latents * components.latent_scale + components.latent_shift
    channels = latents.shape[-1] // (patch * patch)
    latents = latents.view(batch_size, grid_height, grid_width, patch, patch, channels)
    latents = latents.permute(0, 5, 1, 3, 2, 4).contiguous()
    latents = latents.view(batch_size, channels, grid_height * patch, grid_width * patch)
    dtype = components.vae.decoder.conv_in.weight.dtype
    image = components.vae.decode(latents.to(dtype)).float().clamp(-1.0, 1.0)
    image = ((image + 1.0) * 127.5).round().to(torch.uint8)
    return image.permute(0, 2, 3, 1).cpu()


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


class Ideogram4Backend(TextToImageBackend):
    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        model_id: str | None = None,
        lora: LoRAOption = None,
        lora_scale: float = 1.0,
        lora_fuse: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(lora_fuse, bool):
            raise TypeError("lora_fuse must be a bool")
        self.lora = parse_lora_option(lora, lora_scale)
        self.lora_fuse = lora_fuse
        self.source = Path(model)
        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = dtype or torch.float32
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be torch.float32, torch.float16 or torch.bfloat16")
        self.components: Ideogram4Components | None = None
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults

    def _latent_shape(self, options: dict[str, Any]) -> tuple[int, int, int]:
        channels = 128
        config_path = self.source / "transformer" / "config.json"
        if config_path.is_file():
            channels = int(read_json(config_path).get("in_channels", channels))
        return (1, (options["height"] // 16) * (options["width"] // 16), channels)

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown Ideogram 4 options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        preset_name = o["preset"]
        if not isinstance(preset_name, str) or preset_name not in PRESETS:
            raise ValueError(f"preset must be one of {', '.join(PRESETS)}")
        preset = PRESETS[preset_name]
        o["steps"] = preset.num_steps if o["steps"] is None else o["steps"]
        for name in ("steps", "width", "height"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        for name in ("width", "height"):
            if o[name] % 16:
                raise ValueError(f"{name} must be a multiple of 16")
        guidance = o["guidance"]
        if guidance is not None and not _finite_number(guidance):
            raise ValueError("guidance must be a finite number or None")
        schedule = o["guidance_schedule"]
        if schedule is None:
            if guidance is not None:
                schedule = (float(guidance),) * o["steps"]
            elif o["steps"] == preset.num_steps:
                schedule = preset.guidance_schedule
            else:
                schedule = (7.0,) * o["steps"]
        elif isinstance(schedule, (str, bytes)) or not isinstance(schedule, Sequence):
            raise ValueError("guidance_schedule must be a sequence of finite numbers")
        if len(schedule) != o["steps"]:
            raise ValueError(f"guidance_schedule must have length {o['steps']}")
        if not all(_finite_number(value) for value in schedule):
            raise ValueError("guidance_schedule must be a sequence of finite numbers")
        o["guidance_schedule"] = tuple(float(value) for value in schedule)
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
        self.components = Ideogram4Components.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the Ideogram 4 architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )
        if self.lora:
            # Ideogram4Pipeline.load_lora_weights accepts the conditional transformer only.
            apply_lora(
                {"transformer": self.components.conditional_transformer},
                self.lora,
                fuse=self.lora_fuse,
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
                generator=torch.Generator(device=self.device).manual_seed(seed),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            latents = latents.to(device=self.device, dtype=torch.float32)
        preset = PRESETS[o["preset"]]
        latents = denoise(
            self.components,
            prompt=inputs.prompt,
            steps=o["steps"],
            guidance_schedule=o["guidance_schedule"],
            mu=preset.mu,
            std=preset.std,
            width=o["width"],
            height=o["height"],
            latents=latents,
        )
        pixels = decode_latents(self.components, latents, o["height"], o["width"])
        revision = (self.snapshot or {}).get("revision")
        return ImageResult(
            tuple(Image(item) for item in pixels),
            {
                "model": self.model_id or str(self.source),
                "revision": revision,
                "seed": seed,
                "preset": o["preset"],
                "steps": o["steps"],
                "guidance": o["guidance"],
                "guidance_schedule": o["guidance_schedule"],
                "width": o["width"],
                "height": o["height"],
                "device": str(self.device),
                "dtype": str(self.dtype),
            },
        )
