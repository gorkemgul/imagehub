"""Stable Diffusion 3 text-to-image backend over a local diffusers-layout snapshot."""

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
from torch.nn import functional as F

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.lora_formats import LoRAOption, apply_lora, parse_lora_option
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.sd3_transformer import SD3Transformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization import CLIPTokenizer
from imagehub.tokenization_t5 import T5Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = (
    "tokenizer",
    "tokenizer_2",
    "tokenizer_3",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "transformer",
    "vae",
    "scheduler",
)
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 28,
    "guidance": 3.5,
    "width": None,
    "height": None,
    "num_images": 1,
    "seed": None,
    "latents": None,
    "max_sequence_length": 256,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "sd3.5-medium"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class StableDiffusion3Components:
    tokenizer: CLIPTokenizer
    tokenizer_2: CLIPTokenizer
    tokenizer_3: T5Tokenizer
    text_encoder: CLIPTextModelWithProjection
    text_encoder_2: CLIPTextModelWithProjection
    text_encoder_3: T5EncoderModel
    transformer: SD3Transformer2DModel
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
    ) -> StableDiffusion3Components:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Stable Diffusion 3 snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            return cls(
                tokenizer=CLIPTokenizer(root / "tokenizer"),
                tokenizer_2=CLIPTokenizer(root / "tokenizer_2"),
                tokenizer_3=T5Tokenizer(root / "tokenizer_3"),
                text_encoder=CLIPTextModelWithProjection.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                text_encoder_2=CLIPTextModelWithProjection.from_pretrained(
                    root / "text_encoder_2", device=device, dtype=dtype
                ),
                text_encoder_3=T5EncoderModel.from_pretrained(
                    root / "text_encoder_3", device=device, dtype=dtype
                ),
                transformer=SD3Transformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _clip_prompt_embeds(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModelWithProjection,
    prompt: str,
    device: torch.device,
    num_images: int,
) -> tuple[Tensor, Tensor]:
    input_ids = torch.tensor([tokenizer(prompt)], device=device)
    pooled, _, hidden_states = text_encoder(input_ids, output_hidden_states=True)
    prompt_embeds = hidden_states[-2].to(text_encoder.text_projection.weight.dtype)
    _, sequence_length, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images, 1)
    prompt_embeds = prompt_embeds.view(num_images, sequence_length, -1)
    pooled = pooled.repeat(1, num_images).view(num_images, -1)
    return prompt_embeds, pooled


def _t5_prompt_embeds(
    components: StableDiffusion3Components,
    prompt: str,
    device: torch.device,
    num_images: int,
    max_sequence_length: int,
) -> Tensor:
    input_ids, _ = components.tokenizer_3([prompt], max_sequence_length)
    prompt_embeds = components.text_encoder_3(input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(components.text_encoder_3.shared.weight.dtype)
    _, sequence_length, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images, 1)
    return prompt_embeds.view(num_images, sequence_length, -1)


@torch.inference_mode()
def encode_prompt(
    components: StableDiffusion3Components,
    prompt: str,
    negative_prompt: str | None,
    device: torch.device,
    num_images: int,
    max_sequence_length: int,
    do_classifier_free_guidance: bool,
) -> tuple[Tensor, Tensor | None, Tensor, Tensor | None]:
    prompt_1, pooled_1 = _clip_prompt_embeds(
        components.tokenizer, components.text_encoder, prompt, device, num_images
    )
    prompt_2, pooled_2 = _clip_prompt_embeds(
        components.tokenizer_2, components.text_encoder_2, prompt, device, num_images
    )
    clip_prompt = torch.cat([prompt_1, prompt_2], dim=-1)
    t5_prompt = _t5_prompt_embeds(components, prompt, device, num_images, max_sequence_length)
    clip_prompt = F.pad(clip_prompt, (0, t5_prompt.shape[-1] - clip_prompt.shape[-1]))
    prompt_embeds = torch.cat([clip_prompt, t5_prompt], dim=-2)
    pooled_prompt = torch.cat([pooled_1, pooled_2], dim=-1)

    if not do_classifier_free_guidance:
        return prompt_embeds, None, pooled_prompt, None

    negative = negative_prompt or ""
    negative_1, negative_pooled_1 = _clip_prompt_embeds(
        components.tokenizer, components.text_encoder, negative, device, num_images
    )
    negative_2, negative_pooled_2 = _clip_prompt_embeds(
        components.tokenizer_2, components.text_encoder_2, negative, device, num_images
    )
    negative_clip = torch.cat([negative_1, negative_2], dim=-1)
    negative_t5 = _t5_prompt_embeds(components, negative, device, num_images, max_sequence_length)
    negative_clip = F.pad(negative_clip, (0, negative_t5.shape[-1] - negative_clip.shape[-1]))
    negative_embeds = torch.cat([negative_clip, negative_t5], dim=-2)
    negative_pooled = torch.cat([negative_pooled_1, negative_pooled_2], dim=-1)
    return prompt_embeds, negative_embeds, pooled_prompt, negative_pooled


@torch.inference_mode()
def denoise(
    components: StableDiffusion3Components,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    latents: Tensor,
    max_sequence_length: int,
) -> Tensor:
    use_cfg = guidance > 1.0
    prompt_embeds, negative_embeds, pooled, negative_pooled = encode_prompt(
        components,
        prompt,
        negative_prompt,
        latents.device,
        latents.shape[0],
        max_sequence_length,
        use_cfg,
    )
    if use_cfg:
        assert negative_embeds is not None and negative_pooled is not None
        prompt_embeds = torch.cat([negative_embeds, prompt_embeds])
        pooled = torch.cat([negative_pooled, pooled])

    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    scheduler_kwargs: dict[str, float] = {}
    if scheduler.config["use_dynamic_shifting"]:
        height, width = latents.shape[-2:]
        patch_size = int(components.transformer.config["patch_size"])
        image_sequence_length = (height // patch_size) * (width // patch_size)
        base_length = float(scheduler.config["base_image_seq_len"])
        max_length = float(scheduler.config["max_image_seq_len"])
        base_shift = float(scheduler.config["base_shift"])
        max_shift = float(scheduler.config["max_shift"])
        slope = (max_shift - base_shift) / (max_length - base_length)
        scheduler_kwargs["mu"] = image_sequence_length * slope + base_shift - slope * base_length
    scheduler.set_timesteps(steps, mu=scheduler_kwargs.get("mu"))
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents] * 2) if use_cfg else latents
        model_timestep = timestep.expand(model_input.shape[0])
        noise = components.transformer(
            model_input,
            prompt_embeds,
            pooled,
            model_timestep,
        )
        if use_cfg:
            noise_unconditional, noise_text = noise.chunk(2)
            noise = noise_unconditional + guidance * (noise_text - noise_unconditional)
        latents = scheduler.step(noise, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: Tensor) -> Tensor:
    """Decode final latents to uint8 images of shape (batch, height, width, 3) on CPU."""
    latents = latents / vae.scaling_factor + (vae.shift_factor or 0.0)
    image = vae.decode(latents.to(vae.decoder.conv_in.weight.dtype))
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class StableDiffusion3Backend(TextToImageBackend):
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
        self.components: StableDiffusion3Components | None = None
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
            raise TypeError(f"Unknown Stable Diffusion 3 options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        try:
            transformer_config = read_json(self.source / "transformer" / "config.json")
        except FileNotFoundError:
            transformer_config = {"sample_size": 128, "in_channels": 16}
        try:
            vae_config = read_json(self.source / "vae" / "config.json")
        except FileNotFoundError:
            vae_config = {"block_out_channels": [1, 1, 1, 1]}
        vae_scale = 2 ** (len(vae_config.get("block_out_channels", [1, 1, 1, 1])) - 1)
        if o["width"] is None:
            o["width"] = int(transformer_config.get("sample_size", 128)) * vae_scale
        if o["height"] is None:
            o["height"] = int(transformer_config.get("sample_size", 128)) * vae_scale
        if o["negative_prompt"] is not None and not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
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
        shape = (
            o["num_images"],
            int(transformer_config.get("in_channels", 16)),
            o["height"] // vae_scale,
            o["width"] // vae_scale,
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
        o["latent_shape"] = shape
        return o

    def _load(self) -> None:
        self.components = StableDiffusion3Components.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the Stable Diffusion 3 architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )
        if self.lora:
            c = self.components
            apply_lora(
                {
                    "transformer": c.transformer,
                    "text_encoder": c.text_encoder,
                    "text_encoder_2": c.text_encoder_2,
                },
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
            "negative_prompt": o["negative_prompt"],
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
