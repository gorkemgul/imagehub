"""Kandinsky 5 text-to-image backend over a local diffusers-layout snapshot."""

from __future__ import annotations

import math
import os
import re
import secrets
import warnings
from dataclasses import dataclass
from html import unescape
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
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.kandinsky5_transformer import Kandinsky5Transformer3DModel
from imagehub.models.qwen2_5_vl_text import Qwen25VLTextModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization import CLIPTokenizer
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = (
    "tokenizer",
    "tokenizer_2",
    "text_encoder",
    "text_encoder_2",
    "transformer",
    "vae",
    "scheduler",
)
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": "",
    "steps": 50,
    "guidance": 3.5,
    "width": 1024,
    "height": 1024,
    "num_images": 1,
    "seed": None,
    "latents": None,
    "max_sequence_length": 512,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_VAE_SCALE_FACTOR = 8
_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a promt engineer. Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>"
)
_PROMPT_TEMPLATE_ENCODE_START_IDX = 41
_RESOLUTIONS = [
    (1024, 1024),
    (640, 1408),
    (1408, 640),
    (768, 1280),
    (1280, 768),
    (896, 1152),
    (1152, 896),
]


def _clean_prompt(prompt: str) -> str:
    prompt = unescape(unescape(prompt)).strip()
    return re.sub(r"\s+", " ", prompt).strip()


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "kandinsky-5-t2i-lite"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


def _clip_text_encoder(
    directory: Path, *, device: torch.device, dtype: torch.dtype
) -> CLIPTextModel:
    """Load the text-only part of a full CLIPModel checkpoint for the secondary encoder."""
    config = read_json(directory / "config.json")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise CheckpointError(f"Expected text_config in {directory / 'config.json'}")
    state = load_weights(
        directory,
        skip_prefixes=("vision_model.", "visual_projection.", "text_projection.", "logit_scale"),
    )
    state.pop("text_model.embeddings.position_ids", None)
    with torch.device("meta"):
        model = CLIPTextModel(text_config)
    model.load_checkpoint(state)
    model.init_non_persistent_buffers()
    model.to(device=device, dtype=dtype).eval()
    return model


@dataclass
class Kandinsky5Components:
    tokenizer: Qwen2Tokenizer
    tokenizer_2: CLIPTokenizer
    text_encoder: Qwen25VLTextModel
    text_encoder_2: CLIPTextModel
    transformer: Kandinsky5Transformer3DModel
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
    ) -> Kandinsky5Components:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Kandinsky 5 snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            return cls(
                tokenizer=Qwen2Tokenizer(root / "tokenizer"),
                tokenizer_2=CLIPTokenizer(root / "tokenizer_2"),
                text_encoder=Qwen25VLTextModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                text_encoder_2=_clip_text_encoder(
                    root / "text_encoder_2", device=device, dtype=dtype
                ),
                transformer=Kandinsky5Transformer3DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=_flow_match_scheduler_config(
                    read_json(root / "scheduler" / "scheduler_config.json")
                ),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _flow_match_scheduler_config(config: dict[str, Any]) -> dict[str, Any]:
    class_name = config.get("_class_name", "FlowMatchEulerDiscreteScheduler")
    if class_name != "FlowMatchEulerDiscreteScheduler":
        raise ValueError(f"Unsupported Kandinsky 5 scheduler {class_name}")
    return config


def _snap_resolution(width: int, height: int) -> tuple[int, int]:
    if (width, height) in _RESOLUTIONS:
        return width, height
    ratios = [abs((w / h) - (width / height)) for w, h in _RESOLUTIONS]
    return _RESOLUTIONS[ratios.index(min(ratios))]


@torch.inference_mode()
def _encode_prompt_qwen(
    components: Kandinsky5Components,
    prompt: str,
    device: torch.device,
    max_sequence_length: int,
) -> tuple[Tensor, Tensor]:
    full_text = _PROMPT_TEMPLATE.format(prompt)
    max_allowed_len = _PROMPT_TEMPLATE_ENCODE_START_IDX + max_sequence_length

    ids = components.tokenizer.tokenize(full_text)
    if len(ids) > max_allowed_len:
        ids = ids[:max_allowed_len]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([[True] * len(ids)], dtype=torch.bool, device=device)
    hidden_states = components.text_encoder(input_ids, attention_mask)[-1]
    embeds = hidden_states[:, _PROMPT_TEMPLATE_ENCODE_START_IDX:]
    mask = attention_mask[:, _PROMPT_TEMPLATE_ENCODE_START_IDX:]
    cu_seqlens = torch.cumsum(mask.sum(1), dim=0)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0).to(dtype=torch.int32)
    return embeds, cu_seqlens


@torch.inference_mode()
def _encode_prompt_clip(
    components: Kandinsky5Components,
    prompt: str,
    device: torch.device,
) -> Tensor:
    ids = torch.tensor([components.tokenizer_2(prompt)], dtype=torch.long, device=device)
    last_hidden_state = cast(Tensor, components.text_encoder_2(ids))
    eos_token_id = int(components.text_encoder_2.config["eos_token_id"])
    if eos_token_id == 2:
        positions = ids.to(dtype=torch.int).argmax(dim=-1)
    else:
        positions = (ids.to(dtype=torch.int) == eos_token_id).int().argmax(dim=-1)
    return last_hidden_state[torch.arange(last_hidden_state.shape[0], device=device), positions]


@torch.inference_mode()
def encode_prompt(
    components: Kandinsky5Components,
    prompt: str,
    device: torch.device,
    num_images: int,
    max_sequence_length: int,
) -> tuple[Tensor, Tensor, Tensor]:
    prompt = _clean_prompt(prompt)
    qwen_embeds, cu_seqlens = _encode_prompt_qwen(components, prompt, device, max_sequence_length)
    clip_embeds = _encode_prompt_clip(components, prompt, device)

    sequence_length = qwen_embeds.shape[1]
    qwen_embeds = qwen_embeds.repeat(1, num_images, 1).view(num_images, sequence_length, -1)
    clip_embeds = clip_embeds.repeat(num_images, 1)

    original_lengths = cu_seqlens.diff()
    repeated_lengths = original_lengths.repeat_interleave(num_images)
    repeated_cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), repeated_lengths.cumsum(0)])
    return qwen_embeds, clip_embeds, repeated_cu_seqlens


@torch.inference_mode()
def denoise(
    components: Kandinsky5Components,
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
    dtype = cast(torch.dtype, components.transformer.time_embeddings.out_layer.weight.dtype)
    batch_size = latents.shape[0]
    qwen_embeds, clip_embeds, cu_seqlens = encode_prompt(
        components, prompt, latents.device, batch_size, max_sequence_length
    )
    qwen_embeds = qwen_embeds.to(dtype)
    clip_embeds = clip_embeds.to(dtype)

    do_cfg = guidance > 1.0
    negative_qwen = negative_clip = negative_cu = None
    if do_cfg:
        negative_qwen, negative_clip, negative_cu = encode_prompt(
            components, negative_prompt or "", latents.device, batch_size, max_sequence_length
        )
        negative_qwen = negative_qwen.to(dtype)
        negative_clip = negative_clip.to(dtype)

    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    scheduler.set_timesteps(steps)
    scheduler.set_begin_index(0)

    latent_height = height // _VAE_SCALE_FACTOR
    latent_width = width // _VAE_SCALE_FACTOR
    visual_rope_pos = [
        torch.arange(1, device=latents.device),
        torch.arange(latent_height // 2, device=latents.device),
        torch.arange(latent_width // 2, device=latents.device),
    ]
    text_rope_pos = torch.arange(qwen_embeds.shape[1], device=latents.device)
    negative_text_rope_pos = (
        torch.arange(negative_qwen.shape[1], device=latents.device)
        if negative_qwen is not None
        else None
    )

    for timestep in scheduler.timesteps:
        model_timestep = timestep.expand(batch_size)
        pred_velocity = components.transformer(
            hidden_states=latents.to(dtype),
            encoder_hidden_states=qwen_embeds,
            timestep=model_timestep.to(dtype),
            pooled_projections=clip_embeds,
            visual_rope_pos=visual_rope_pos,
            text_rope_pos=text_rope_pos,
        )

        if (
            do_cfg
            and negative_qwen is not None
            and negative_clip is not None
            and negative_cu is not None
        ):
            uncond_pred = components.transformer(
                hidden_states=latents.to(dtype),
                encoder_hidden_states=negative_qwen,
                timestep=model_timestep.to(dtype),
                pooled_projections=negative_clip,
                visual_rope_pos=visual_rope_pos,
                text_rope_pos=cast(Tensor, negative_text_rope_pos),
            )
            pred_velocity = uncond_pred + guidance * (pred_velocity - uncond_pred)

        latents = scheduler.step(pred_velocity, timestep, latents)

    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: Tensor, height: int, width: int) -> Tensor:
    latent_height = height // _VAE_SCALE_FACTOR
    latent_width = width // _VAE_SCALE_FACTOR
    latents = latents[:, :, :, :, : latents.shape[-1]]
    latents = latents.reshape(latents.shape[0], 1, latent_height, latent_width, -1)
    latents = latents.permute(0, 4, 1, 2, 3)
    latents = latents.reshape(latents.shape[0], -1, latent_height, latent_width)
    latents = latents / vae.scaling_factor
    image = vae.decode(latents.to(vae.decoder.conv_in.weight.dtype))
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class Kandinsky5Backend(TextToImageBackend):
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
        self.components: Kandinsky5Components | None = None
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown Kandinsky 5 options: {', '.join(unknown)}")
        selected = self.option_defaults | merged
        if selected["negative_prompt"] is not None and not isinstance(
            selected["negative_prompt"], str
        ):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "num_images", "width", "height", "max_sequence_length"):
            _positive_int(selected, name)
        if selected["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        if selected["max_sequence_length"] > 1024:
            raise ValueError("max_sequence_length must be at most 1024")
        width, height = _snap_resolution(selected["width"], selected["height"])
        for name, value in (("width", width), ("height", height)):
            if value % _VAE_SCALE_FACTOR:
                raise ValueError(f"{name} must be a multiple of {_VAE_SCALE_FACTOR}")
        selected["width"] = width
        selected["height"] = height
        guidance = selected["guidance"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, int | float)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError("guidance must be a finite number >= 0")
        seed, latents = selected["seed"], selected["latents"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        latent_height = height // _VAE_SCALE_FACTOR
        latent_width = width // _VAE_SCALE_FACTOR
        in_visual_dim = self._in_visual_dim()
        shape = (selected["num_images"], 1, latent_height, latent_width, in_visual_dim)
        if latents is not None:
            if (
                not isinstance(latents, Tensor)
                or not latents.is_floating_point()
                or tuple(latents.shape) != shape
            ):
                raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        selected["latent_shape"] = shape
        return selected

    def _in_visual_dim(self) -> int:
        try:
            config = read_json(self.source / "transformer" / "config.json")
        except FileNotFoundError:
            return 16
        return int(config.get("in_visual_dim", 16))

    def _load(self) -> None:
        self.components = Kandinsky5Components.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the Kandinsky 5 architecture, so it was loaded",
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
            if seed is None:
                seed = secrets.randbits(63)
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
            width=options["width"],
            height=options["height"],
            latents=latents.to(self.dtype).to(self.device),
            max_sequence_length=options["max_sequence_length"],
        )
        pixels = decode_latents(self.components.vae, final, options["height"], options["width"])
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": options["steps"],
            "guidance": float(options["guidance"]),
            "negative_prompt": options["negative_prompt"],
            "width": options["width"],
            "height": options["height"],
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
