"""Krea 2 text-to-image backend over a local diffusers-layout snapshot."""

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
from imagehub.checkpoints import load_weights, read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.lora_formats import LoRAOption, apply_lora, parse_lora_option
from imagehub.models.autoencoder_kl_qwen_image import AutoencoderKLQwenImage
from imagehub.models.krea2_transformer import Krea2Transformer2DModel
from imagehub.models.qwen3_vl_text import Qwen3VLTextModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 28,
    "guidance": 4.5,
    "width": 1024,
    "height": 1024,
    "seed": None,
    "latents": None,
    "max_sequence_length": 512,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_PROMPT_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_PROMPT_START = 34
_PROMPT_SUFFIX_TOKENS = 5
_TEXT_PREFIXES = ("language_model.", "model.language_model.")
_VISION_PREFIXES = ("visual.", "model.visual.")
_EXPECTED_UNUSED_TEXT_TENSORS = frozenset({"lm_head.weight"})


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


def pack_latents(latents: Tensor, patch_size: int = 2) -> Tensor:
    batch_size, channels, height, width = latents.shape
    latents = latents.view(
        batch_size,
        channels,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(
        batch_size,
        (height // patch_size) * (width // patch_size),
        channels * patch_size * patch_size,
    )


def unpack_latents(latents: Tensor, height: int, width: int, patch_size: int = 2) -> Tensor:
    batch_size, _, channels = latents.shape
    latents = latents.view(
        batch_size,
        height // patch_size,
        width // patch_size,
        channels // (patch_size * patch_size),
        patch_size,
        patch_size,
    )
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // (patch_size * patch_size), 1, height, width)


def prepare_position_ids(
    text_seq_len: int, grid_height: int, grid_width: int, device: torch.device
) -> Tensor:
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
    return torch.cat([text_ids, image_ids.reshape(grid_height * grid_width, 3)], dim=0)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "krea2-turbo"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


def _load_text_encoder(
    directory: Path, *, device: torch.device, dtype: torch.dtype
) -> Qwen3VLTextModel:
    config = read_json(directory / "config.json")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise CheckpointError(f"Expected text_config in {directory / 'config.json'}")
    state = load_weights(directory)
    unexpected = sorted(
        name
        for name in state
        if not name.startswith(_TEXT_PREFIXES + _VISION_PREFIXES)
        and name not in _EXPECTED_UNUSED_TEXT_TENSORS
    )
    if unexpected:
        raise CheckpointError(
            "Unexpected Qwen3-VL text-encoder tensors: " + ", ".join(unexpected[:20])
        )
    language_state = {}
    for name, value in state.items():
        prefix = next((item for item in _TEXT_PREFIXES if name.startswith(item)), None)
        if prefix is not None:
            language_state[name.removeprefix(prefix)] = value
    with torch.device("meta"):
        model = Qwen3VLTextModel(text_config)
    model.load_checkpoint(language_state)
    model.init_non_persistent_buffers()
    model.to(device=device, dtype=dtype).eval()
    return model


@dataclass
class Krea2Components:
    tokenizer: Qwen2Tokenizer
    text_encoder: Qwen3VLTextModel
    transformer: Krea2Transformer2DModel
    vae: AutoencoderKLQwenImage
    scheduler_config: dict[str, Any]
    text_encoder_select_layers: tuple[int, ...]
    is_distilled: bool
    patch_size: int
    vae_scale_factor: int

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> Krea2Components:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Krea 2 snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            model_index = read_json(root / "model_index.json")
            vae_config = read_json(root / "vae" / "config.json")
            selected = model_index.get(
                "text_encoder_select_layers", (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
            )
            return cls(
                tokenizer=Qwen2Tokenizer(root / "tokenizer"),
                text_encoder=_load_text_encoder(root / "text_encoder", device=device, dtype=dtype),
                transformer=Krea2Transformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKLQwenImage.from_pretrained(
                    root / "vae", device=device, dtype=dtype
                ),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
                text_encoder_select_layers=tuple(int(index) for index in selected),
                is_distilled=bool(model_index.get("is_distilled", False)),
                patch_size=int(model_index.get("patch_size", 2)),
                vae_scale_factor=2 ** len(vae_config.get("temperal_downsample", [False] * 3)),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


@torch.inference_mode()
def encode_prompt(
    components: Krea2Components,
    prompt: str,
    device: torch.device,
    max_sequence_length: int = 512,
) -> tuple[Tensor, Tensor]:
    input_ids, attention_mask = components.tokenizer(
        _PROMPT_PREFIX + prompt,
        max_sequence_length + _PROMPT_START - _PROMPT_SUFFIX_TOKENS,
    )
    suffix_ids = components.tokenizer.tokenize(_PROMPT_SUFFIX)
    suffix = torch.tensor([suffix_ids], dtype=torch.long)
    suffix_mask = torch.ones_like(suffix, dtype=torch.bool)
    input_ids = torch.cat([input_ids, suffix], dim=1).to(device)
    attention_mask = torch.cat([attention_mask, suffix_mask], dim=1).to(device)
    positions = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    position_ids = positions.unsqueeze(0).expand(3, -1, -1)
    hidden_states = components.text_encoder(
        input_ids,
        attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
    )
    selected = torch.stack(
        [hidden_states[index] for index in components.text_encoder_select_layers], dim=2
    )
    return selected[:, _PROMPT_START:], attention_mask[:, _PROMPT_START:]


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
    components: Krea2Components,
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
    dtype = next(components.transformer.parameters()).dtype
    prompt_embeds, prompt_mask = encode_prompt(
        components, prompt, latents.device, max_sequence_length
    )
    prompt_embeds = prompt_embeds.to(dtype)
    use_cfg = guidance > 0
    negative_embeds = negative_mask = None
    if use_cfg:
        negative_embeds, negative_mask = encode_prompt(
            components, negative_prompt or "", latents.device, max_sequence_length
        )
        negative_embeds = negative_embeds.to(dtype)

    grid_height = height // (components.vae_scale_factor * components.patch_size)
    grid_width = width // (components.vae_scale_factor * components.patch_size)
    position_ids = prepare_position_ids(
        prompt_embeds.shape[1], grid_height, grid_width, latents.device
    )
    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    mu = (
        1.15
        if components.is_distilled
        else calculate_shift(
            latents.shape[1],
            int(components.scheduler_config.get("base_image_seq_len", 256)),
            int(components.scheduler_config.get("max_image_seq_len", 6400)),
            float(components.scheduler_config.get("base_shift", 0.5)),
            float(components.scheduler_config.get("max_shift", 1.15)),
        )
    )
    scheduler.set_timesteps(steps, sigmas=_sigmas(steps), mu=mu)
    scheduler.set_begin_index(0)
    for timestep in scheduler.timesteps:
        model_timestep = (
            (timestep / int(scheduler.config["num_train_timesteps"]))
            .expand(latents.shape[0])
            .to(latents.dtype)
        )
        noise_pred = components.transformer(
            latents.to(dtype),
            prompt_embeds,
            model_timestep,
            position_ids,
            prompt_mask,
        )
        if use_cfg:
            assert negative_embeds is not None and negative_mask is not None
            negative_pred = components.transformer(
                latents.to(dtype),
                negative_embeds,
                model_timestep,
                position_ids,
                negative_mask,
            )
            noise_pred = noise_pred + guidance * (noise_pred - negative_pred)
        latents = scheduler.step(noise_pred, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(components: Krea2Components, latents: Tensor, height: int, width: int) -> Tensor:
    latent_height = components.patch_size * (
        height // (components.vae_scale_factor * components.patch_size)
    )
    latent_width = components.patch_size * (
        width // (components.vae_scale_factor * components.patch_size)
    )
    latents = unpack_latents(latents, latent_height, latent_width, patch_size=components.patch_size)
    dtype = components.vae.post_quant_conv.weight.dtype
    channels = int(components.vae.config["z_dim"])
    mean = torch.tensor(components.vae.latents_mean, device=latents.device, dtype=dtype).view(
        1, channels, 1, 1, 1
    )
    # As the reference: divide by the reciprocal of the standard deviation (not multiply by it).
    inverse_std = 1.0 / torch.tensor(
        components.vae.latents_std, device=latents.device, dtype=dtype
    ).view(1, channels, 1, 1, 1)
    image = components.vae.decode(latents.to(dtype) / inverse_std + mean)[:, :, 0]
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class Krea2Backend(TextToImageBackend):
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
        self.components: Krea2Components | None = None
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults

    def _layout(self) -> tuple[int, int, int]:
        in_channels, patch_size, scale = 64, 2, 8
        transformer_config = self.source / "transformer" / "config.json"
        vae_config = self.source / "vae" / "config.json"
        model_index = self.source / "model_index.json"
        if transformer_config.is_file():
            in_channels = int(read_json(transformer_config).get("in_channels", in_channels))
        if model_index.is_file():
            patch_size = int(read_json(model_index).get("patch_size", patch_size))
        if vae_config.is_file():
            temporal = read_json(vae_config).get("temperal_downsample", [False] * 3)
            scale = 2 ** len(temporal)
        return in_channels, patch_size, scale

    def _latent_shape(self, options: dict[str, Any]) -> tuple[int, int, int]:
        in_channels, patch_size, scale = self._layout()
        return (
            1,
            (options["height"] // (scale * patch_size))
            * (options["width"] // (scale * patch_size)),
            in_channels,
        )

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown Krea 2 options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        for name in ("steps", "width", "height", "max_sequence_length"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        _, patch_size, scale = self._layout()
        for name in ("width", "height"):
            if o[name] % (scale * patch_size):
                raise ValueError(f"{name} must be a multiple of {scale * patch_size}")
        guidance = o["guidance"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, int | float)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError("guidance must be a finite number >= 0")
        negative_prompt = o["negative_prompt"]
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise ValueError("negative_prompt must be a string or None")
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
        self.components = Krea2Components.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the Krea 2 architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )
        if self.lora:
            # Krea2Pipeline.load_lora_weights accepts transformer adapters only.
            apply_lora({"transformer": self.components.transformer}, self.lora, fuse=self.lora_fuse)

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        seed, latents = o["seed"], o["latents"]
        if latents is None:
            if seed is None:
                seed = secrets.randbits(63)
            in_channels, patch_size, scale = self._layout()
            raw = torch.randn(
                (
                    1,
                    in_channels // (patch_size * patch_size),
                    o["height"] // scale,
                    o["width"] // scale,
                ),
                generator=torch.Generator("cpu").manual_seed(seed),
                dtype=torch.float32,
            )
            latents = pack_latents(raw, patch_size)
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
        pixels = decode_latents(self.components, final, o["height"], o["width"])
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
