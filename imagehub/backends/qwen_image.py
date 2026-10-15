"""Qwen-Image text-to-image backends over local diffusers-layout snapshots."""

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
from imagehub.checkpoints import load_weights, read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl_qwen_image import AutoencoderKLQwenImage
from imagehub.models.autoencoder_kl_qwen_image21 import AutoencoderKLQwenImage21
from imagehub.models.qwen2_5_vl_text import Qwen25VLTextModel
from imagehub.models.qwen3_vl_text import Qwen3VLTextModel
from imagehub.models.qwen_image21_transformer import (
    QwenImage21KVCache,
    QwenImage21Transformer2DModel,
)
from imagehub.models.qwen_image_transformer import QwenImageTransformer2DModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler, qwen_image_mu
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("processor", "text_encoder", "transformer", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 40,
    "true_cfg": 1.0,
    "width": 1024,
    "height": 1024,
    "num_images": 1,
    "seed": None,
    "latents": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_VAE_SCALE_FACTOR = 16
_SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
_SYSTEM_TEMPLATE = f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
_PROMPT_TEMPLATE = _SYSTEM_TEMPLATE + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
_ORIGINAL_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
_ORIGINAL_DROP_INDEX = 34
_TEXT_PREFIX = "model.language_model."
_VISION_PREFIX = "model.visual."
_EXPECTED_UNUSED_TEXT_TENSORS = frozenset({"lm_head.weight"})


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "qwen-image-2.1"
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
        if not name.startswith((_TEXT_PREFIX, _VISION_PREFIX))
        and name not in _EXPECTED_UNUSED_TEXT_TENSORS
    )
    if unexpected:
        raise CheckpointError(
            "Unexpected Qwen3-VL text-encoder tensors: " + ", ".join(unexpected[:20])
        )
    language_state = {
        name.removeprefix(_TEXT_PREFIX): value
        for name, value in state.items()
        if name.startswith(_TEXT_PREFIX)
    }
    with torch.device("meta"):
        model = Qwen3VLTextModel(text_config)
    model.load_checkpoint(language_state)
    model.init_non_persistent_buffers()
    model.to(device=device, dtype=dtype).eval()
    return model


@dataclass
class QwenImageComponents:
    tokenizer: Qwen2Tokenizer
    text_encoder: Qwen3VLTextModel
    transformer: QwenImage21Transformer2DModel
    vae: AutoencoderKLQwenImage21
    scheduler_config: dict[str, Any]

    @staticmethod
    def load_text_encoder(
        directory: str | os.PathLike[str], *, device: torch.device, dtype: torch.dtype
    ) -> Qwen3VLTextModel:
        return _load_text_encoder(Path(directory), device=device, dtype=dtype)

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> QwenImageComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Qwen-Image snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            return cls(
                tokenizer=Qwen2Tokenizer(root / "processor"),
                text_encoder=_load_text_encoder(root / "text_encoder", device=device, dtype=dtype),
                transformer=QwenImage21Transformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKLQwenImage21.from_pretrained(
                    root / "vae", device=device, dtype=dtype
                ),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


@dataclass
class OriginalQwenImageComponents:
    tokenizer: Qwen2Tokenizer
    text_encoder: Qwen25VLTextModel
    transformer: QwenImageTransformer2DModel
    vae: AutoencoderKLQwenImage
    scheduler_config: dict[str, Any]

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> OriginalQwenImageComponents:
        root = Path(directory)
        names = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
        missing = [name for name in names if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not an original Qwen-Image snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            return cls(
                tokenizer=Qwen2Tokenizer(root / "tokenizer"),
                text_encoder=Qwen25VLTextModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                transformer=QwenImageTransformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKLQwenImage.from_pretrained(
                    root / "vae", device=device, dtype=dtype
                ),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


@torch.inference_mode()
def encode_prompt(
    components: QwenImageComponents,
    prompt: str,
    device: torch.device,
    num_images: int = 1,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Encode the raw Qwen template and drop its leading system-role tokens."""
    prompt = prompt or " "
    prompt_ids = components.tokenizer.tokenize(_PROMPT_TEMPLATE.format(prompt))
    drop_idx = len(components.tokenizer.tokenize(_SYSTEM_TEMPLATE))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    hidden_states = components.text_encoder(input_ids, attention_mask)[-1][:, drop_idx:]
    sequence_length = hidden_states.shape[1]
    hidden_states = hidden_states.repeat(1, num_images, 1).view(num_images, sequence_length, -1)
    image_pad_mask = torch.zeros(1, sequence_length, dtype=torch.bool, device=device)
    return hidden_states, None, image_pad_mask


@torch.inference_mode()
def denoise(
    components: QwenImageComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    true_cfg: float,
    width: int,
    height: int,
    latents: Tensor,
) -> Tensor:
    """Run the Qwen-Image 2.1 flow-match loop with its text KV cache."""
    dtype = cast(torch.dtype, components.transformer.img_in.weight.dtype)
    batch_size = latents.shape[0]
    prompt_embeds, prompt_mask, image_pad_mask = encode_prompt(
        components, prompt, latents.device, batch_size
    )
    prompt_embeds = prompt_embeds.to(dtype)
    use_cfg = true_cfg > 1 and negative_prompt is not None
    negative_embeds = negative_mask = negative_image_mask = None
    if use_cfg:
        assert negative_prompt is not None
        negative_embeds, negative_mask, negative_image_mask = encode_prompt(
            components, negative_prompt, latents.device, batch_size
        )
        negative_embeds = negative_embeds.to(dtype)

    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    sigmas = torch.linspace(1.0, 1 / steps, steps, dtype=torch.float64).tolist()
    scheduler.set_timesteps(
        steps,
        sigmas=sigmas,
        mu=qwen_image_mu(height, width, components.scheduler_config),
    )
    scheduler.set_begin_index(0)
    target_slots = latents.shape[1] // 4
    image_pad_mask = torch.cat([image_pad_mask, image_pad_mask.new_ones(1, target_slots)], dim=1)
    if negative_image_mask is not None:
        negative_image_mask = torch.cat(
            [negative_image_mask, negative_image_mask.new_ones(1, target_slots)], dim=1
        )
    img_shapes = [[(1, height // _VAE_SCALE_FACTOR, width // _VAE_SCALE_FACTOR)]] * batch_size
    num_layers = len(components.transformer.transformer_blocks)
    cache_enabled = components.transformer.causal_condition
    condition_cache = QwenImage21KVCache(num_layers) if cache_enabled else None
    negative_cache = QwenImage21KVCache(num_layers) if cache_enabled and use_cfg else None

    for index, timestep in enumerate(scheduler.timesteps):
        cache_mode = (
            "extract" if cache_enabled and index == 0 else "cached" if cache_enabled else None
        )
        model_timestep = timestep.expand(batch_size).to(latents.dtype) / 1000
        noise_pred = components.transformer(
            hidden_states=latents,
            timestep=model_timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes,
            img_mask=image_pad_mask,
            kv_cache=condition_cache,
            kv_cache_mode=cache_mode,
        )[:, -latents.shape[1] :]
        if use_cfg:
            assert negative_embeds is not None and negative_image_mask is not None
            negative_pred = components.transformer(
                hidden_states=latents,
                timestep=model_timestep,
                encoder_hidden_states=negative_embeds,
                encoder_hidden_states_mask=negative_mask,
                img_shapes=img_shapes,
                img_mask=negative_image_mask,
                kv_cache=negative_cache,
                kv_cache_mode=cache_mode,
            )[:, -latents.shape[1] :]
            noise_pred = negative_pred + true_cfg * (noise_pred - negative_pred)
        latents = scheduler.step(noise_pred, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(
    vae: AutoencoderKLQwenImage21, latents: Tensor, height: int, width: int
) -> Tensor:
    """Unpack and decode final latents to uint8 RGBA images on CPU."""
    batch_size, _, channels = latents.shape
    latent_height = 2 * (height // (_VAE_SCALE_FACTOR * 2))
    latent_width = 2 * (width // (_VAE_SCALE_FACTOR * 2))
    latents = latents.transpose(1, 2).reshape(batch_size, channels, latent_height, latent_width)
    dtype = vae.post_quant_conv.weight.dtype
    latents = latents.to(dtype)
    if vae.latents_mean is not None and vae.latents_std is not None:
        mean = torch.tensor(vae.latents_mean, device=latents.device, dtype=dtype).view(
            1, channels, 1, 1
        )
        std = torch.tensor(vae.latents_std, device=latents.device, dtype=dtype).view(
            1, channels, 1, 1
        )
        latents = latents * std + mean
    image = vae.decode(latents)
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


@torch.inference_mode()
def encode_original_prompt(
    components: OriginalQwenImageComponents,
    prompt: str,
    device: torch.device,
    num_images: int = 1,
) -> tuple[Tensor, Tensor | None]:
    """Encode the original release's prompt template and drop its 34-token prefix."""
    prompt_ids = components.tokenizer.tokenize(_ORIGINAL_PROMPT_TEMPLATE.format(prompt))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    hidden_states = components.text_encoder(input_ids, attention_mask)[-1]
    hidden_states = hidden_states[:, _ORIGINAL_DROP_INDEX : _ORIGINAL_DROP_INDEX + 512]
    sequence_length = hidden_states.shape[1]
    hidden_states = hidden_states.repeat(1, num_images, 1).view(num_images, sequence_length, -1)
    return hidden_states, None


@torch.inference_mode()
def denoise_original(
    components: OriginalQwenImageComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    true_cfg: float,
    width: int,
    height: int,
    latents: Tensor,
) -> Tensor:
    """Run the original Qwen-Image flow-match loop."""
    dtype = cast(torch.dtype, components.transformer.img_in.weight.dtype)
    batch_size = latents.shape[0]
    prompt_embeds, prompt_mask = encode_original_prompt(
        components, prompt, latents.device, batch_size
    )
    prompt_embeds = prompt_embeds.to(dtype)
    use_cfg = true_cfg > 1 and negative_prompt is not None
    negative_embeds = negative_mask = None
    if use_cfg:
        assert negative_prompt is not None
        negative_embeds, negative_mask = encode_original_prompt(
            components, negative_prompt, latents.device, batch_size
        )
        negative_embeds = negative_embeds.to(dtype)

    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    sigmas = torch.linspace(1.0, 1 / steps, steps, dtype=torch.float64).tolist()
    scheduler.set_timesteps(
        steps,
        sigmas=sigmas,
        mu=qwen_image_mu(height, width, components.scheduler_config),
    )
    scheduler.set_begin_index(0)
    img_shapes = [[(1, height // 16, width // 16)]] * batch_size
    for timestep in scheduler.timesteps:
        model_timestep = timestep.expand(batch_size).to(latents.dtype) / 1000
        noise_pred = components.transformer(
            hidden_states=latents,
            timestep=model_timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes,
        )
        if use_cfg:
            assert negative_embeds is not None
            negative_pred = components.transformer(
                hidden_states=latents,
                timestep=model_timestep,
                encoder_hidden_states=negative_embeds,
                encoder_hidden_states_mask=negative_mask,
                img_shapes=img_shapes,
            )
            combined = negative_pred + true_cfg * (noise_pred - negative_pred)
            condition_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            combined_norm = torch.norm(combined, dim=-1, keepdim=True)
            noise_pred = combined * (condition_norm / combined_norm)
        latents = scheduler.step(noise_pred, timestep, latents)
    return latents


@torch.inference_mode()
def decode_original_latents(
    vae: AutoencoderKLQwenImage, latents: Tensor, height: int, width: int
) -> Tensor:
    """Unpack and decode original Qwen-Image latents to uint8 RGBA images."""
    batch_size, _, channels = latents.shape
    latent_height = 2 * (height // 16)
    latent_width = 2 * (width // 16)
    latents = latents.view(batch_size, latent_height // 2, latent_width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5).reshape(
        batch_size, channels // 4, 1, latent_height, latent_width
    )
    dtype = vae.post_quant_conv.weight.dtype
    latents = latents.to(dtype)
    mean = torch.tensor(vae.latents_mean, device=latents.device, dtype=dtype).view(
        1, channels // 4, 1, 1, 1
    )
    std = torch.tensor(vae.latents_std, device=latents.device, dtype=dtype).view(
        1, channels // 4, 1, 1, 1
    )
    image = vae.decode(latents * std + mean)[:, :, 0]
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class QwenImageBackend(TextToImageBackend):
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
        self.components: QwenImageComponents | OriginalQwenImageComponents | None = None
        self.original_release = model_id == "qwen-image"
        model_index_path = self.source / "model_index.json"
        if model_index_path.is_file():
            self.original_release = (
                read_json(model_index_path).get("_class_name") == "QwenImagePipeline"
            )
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        release_defaults = {"steps": 50, "true_cfg": 4.0} if self.original_release else {}
        self.option_defaults = OPTION_DEFAULTS | release_defaults | defaults

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown Qwen-Image options: {', '.join(unknown)}")
        selected = self.option_defaults | merged
        if selected["negative_prompt"] is not None and not isinstance(
            selected["negative_prompt"], str
        ):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "num_images", "width", "height"):
            _positive_int(selected, name)
        if selected["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        size_multiple = 16 if self.original_release else 32
        for name in ("width", "height"):
            if selected[name] % size_multiple:
                raise ValueError(f"{name} must be a multiple of {size_multiple}")
        true_cfg = selected["true_cfg"]
        if (
            isinstance(true_cfg, bool)
            or not isinstance(true_cfg, int | float)
            or not math.isfinite(true_cfg)
            or true_cfg < 0
        ):
            raise ValueError("true_cfg must be a finite number >= 0")
        seed, latents = selected["seed"], selected["latents"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        shape = (
            selected["num_images"],
            (selected["height"] // _VAE_SCALE_FACTOR) * (selected["width"] // _VAE_SCALE_FACTOR),
            64,
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
        return selected

    def _load(self) -> None:
        model_index_path = self.source / "model_index.json"
        if model_index_path.is_file():
            self.original_release = (
                read_json(model_index_path).get("_class_name") == "QwenImagePipeline"
            )
        component_type = (
            OriginalQwenImageComponents if self.original_release else QwenImageComponents
        )
        self.components = component_type.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the Qwen-Image architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed, latents = options["seed"], options["latents"]
        latent_channels = int(self.components.transformer.config.get("in_channels", 64))
        latent_height = options["height"] // _VAE_SCALE_FACTOR
        latent_width = options["width"] // _VAE_SCALE_FACTOR
        if latents is None:
            if seed is None:
                seed = secrets.randbits(63)
            generator = torch.Generator("cpu").manual_seed(seed)
            if self.original_release:
                latent_channels //= 4
                raw_height = latent_height * 2
                raw_width = latent_width * 2
                raw = torch.randn(
                    (
                        options["num_images"],
                        1,
                        latent_channels,
                        raw_height,
                        raw_width,
                    ),
                    generator=generator,
                    dtype=self.dtype,
                )
                latents = raw.view(
                    options["num_images"],
                    latent_channels,
                    raw_height // 2,
                    2,
                    raw_width // 2,
                    2,
                )
                latents = latents.permute(0, 2, 4, 1, 3, 5).reshape(
                    options["num_images"],
                    (raw_height // 2) * (raw_width // 2),
                    latent_channels * 4,
                )
            else:
                latents = torch.randn(
                    (
                        options["num_images"],
                        1,
                        latent_channels,
                        latent_height,
                        latent_width,
                    ),
                    generator=generator,
                    dtype=self.dtype,
                )
                latents = latents.view(
                    options["num_images"], latent_channels, latent_height * latent_width
                ).transpose(1, 2)
        common = {
            "prompt": inputs.prompt,
            "negative_prompt": options["negative_prompt"],
            "steps": options["steps"],
            "true_cfg": float(options["true_cfg"]),
            "width": options["width"],
            "height": options["height"],
            "latents": latents.to(self.dtype).to(self.device),
        }
        if self.original_release:
            original = cast(OriginalQwenImageComponents, self.components)
            final = denoise_original(original, **common)
            pixels = decode_original_latents(
                original.vae, final, options["height"], options["width"]
            )
        else:
            release_21 = cast(QwenImageComponents, self.components)
            final = denoise(release_21, **common)
            pixels = decode_latents(release_21.vae, final, options["height"], options["width"])
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": options["steps"],
            "true_cfg": float(options["true_cfg"]),
            "width": options["width"],
            "height": options["height"],
            "negative_prompt": options["negative_prompt"],
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
