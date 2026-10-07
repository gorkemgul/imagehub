"""HunyuanDiT text-to-image backend over a local diffusers-layout snapshot."""

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
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.bert import BertModel
from imagehub.models.hunyuan_dit_transformer import (
    HunyuanDiT2DModel,
    get_2d_rotary_pos_embed,
    get_resize_crop_region_for_grid,
)
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.ddpm import DDPMScheduler
from imagehub.tokenization_bert import BertTokenizer
from imagehub.tokenization_t5 import T5Tokenizer
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
    "steps": 25,
    "guidance": 5.0,
    "width": 1024,
    "height": 1024,
    "seed": None,
    "negative_prompt": None,
    "guidance_rescale": 0.0,
    "use_resolution_binning": True,
    "latents": None,
}
SUPPORTED_SHAPE = (
    (1024, 1024),
    (1280, 1280),
    (1024, 768),
    (1152, 864),
    (1280, 960),
    (768, 1024),
    (864, 1152),
    (960, 1280),
    (1280, 768),
    (768, 1280),
)
STANDARD_RATIO = (1.0, 4.0 / 3.0, 3.0 / 4.0, 16.0 / 9.0, 9.0 / 16.0)
STANDARD_SHAPE = (
    ((1024, 1024), (1280, 1280)),
    ((1024, 768), (1152, 864), (1280, 960)),
    ((768, 1024), (864, 1152), (960, 1280)),
    ((1280, 768),),
    ((768, 1280),),
)
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def map_to_standard_shapes(target_width: int, target_height: int) -> tuple[int, int]:
    target_ratio = target_width / target_height
    ratio_index = min(
        range(len(STANDARD_RATIO)), key=lambda index: abs(STANDARD_RATIO[index] - target_ratio)
    )
    target_area = target_width * target_height
    return min(
        STANDARD_SHAPE[ratio_index],
        key=lambda shape: abs(shape[0] * shape[1] - target_area),
    )


def get_add_time_ids(
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    batch_size: int = 1,
) -> Tensor:
    values = [1024, 1024, height, width, 0, 0]
    return torch.tensor([values], dtype=dtype, device=device).repeat(batch_size, 1)


def rescale_noise_cfg(
    noise_cfg: Tensor, noise_pred_text: Tensor, guidance_rescale: float = 0.0
) -> Tensor:
    dimensions = list(range(1, noise_pred_text.ndim))
    std_text = noise_pred_text.std(dim=dimensions, keepdim=True)
    std_cfg = noise_cfg.std(dim=dimensions, keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "hunyuan-dit-v1.1-distilled"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class HunyuanDiTComponents:
    tokenizer: BertTokenizer
    tokenizer_2: T5Tokenizer
    text_encoder: BertModel
    text_encoder_2: T5EncoderModel
    transformer: HunyuanDiT2DModel
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
    ) -> HunyuanDiTComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a HunyuanDiT snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            return cls(
                tokenizer=BertTokenizer(root / "tokenizer"),
                tokenizer_2=T5Tokenizer(root / "tokenizer_2"),
                text_encoder=BertModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                text_encoder_2=T5EncoderModel.from_pretrained(
                    root / "text_encoder_2", device=device, dtype=dtype
                ),
                transformer=HunyuanDiT2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _prompt_embeds(
    tokenizer: BertTokenizer | T5Tokenizer,
    text_encoder: BertModel | T5EncoderModel,
    prompt: str,
    max_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    input_ids, attention_mask = tokenizer([prompt], max_length)
    encoded = text_encoder(input_ids.to(device), attention_mask.to(device))
    embeds = encoded[0] if isinstance(encoded, tuple) else encoded
    return embeds.to(dtype=dtype), attention_mask.to(device)


@torch.inference_mode()
def encode_prompt(
    components: HunyuanDiTComponents,
    prompt: str,
    negative_prompt: str | None,
    device: torch.device,
    do_classifier_free_guidance: bool,
) -> tuple[
    Tensor, Tensor | None, Tensor, Tensor | None, Tensor, Tensor | None, Tensor, Tensor | None
]:
    dtype = components.transformer.pos_embed.proj.weight.dtype
    prompt_embeds, prompt_mask = _prompt_embeds(
        components.tokenizer, components.text_encoder, prompt, 77, device, dtype
    )
    prompt_embeds_2, prompt_mask_2 = _prompt_embeds(
        components.tokenizer_2, components.text_encoder_2, prompt, 256, device, dtype
    )
    if not do_classifier_free_guidance:
        return (
            prompt_embeds,
            None,
            prompt_mask,
            None,
            prompt_embeds_2,
            None,
            prompt_mask_2,
            None,
        )
    negative = negative_prompt or ""
    negative_embeds, negative_mask = _prompt_embeds(
        components.tokenizer, components.text_encoder, negative, 77, device, dtype
    )
    negative_embeds_2, negative_mask_2 = _prompt_embeds(
        components.tokenizer_2, components.text_encoder_2, negative, 256, device, dtype
    )
    return (
        prompt_embeds,
        negative_embeds,
        prompt_mask,
        negative_mask,
        prompt_embeds_2,
        negative_embeds_2,
        prompt_mask_2,
        negative_mask_2,
    )


@torch.inference_mode()
def denoise(
    components: HunyuanDiTComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    guidance_rescale: float,
    width: int,
    height: int,
    latents: Tensor,
    generator: torch.Generator,
) -> Tensor:
    use_cfg = guidance > 1.0
    (
        prompt_embeds,
        negative_embeds,
        prompt_mask,
        negative_mask,
        prompt_embeds_2,
        negative_embeds_2,
        prompt_mask_2,
        negative_mask_2,
    ) = encode_prompt(components, prompt, negative_prompt, latents.device, use_cfg)
    if use_cfg:
        assert negative_embeds is not None and negative_mask is not None
        assert negative_embeds_2 is not None and negative_mask_2 is not None
        prompt_embeds = torch.cat([negative_embeds, prompt_embeds])
        prompt_mask = torch.cat([negative_mask, prompt_mask])
        prompt_embeds_2 = torch.cat([negative_embeds_2, prompt_embeds_2])
        prompt_mask_2 = torch.cat([negative_mask_2, prompt_mask_2])

    scheduler = DDPMScheduler(components.scheduler_config)
    scheduler.set_timesteps(steps)
    patch_size = int(components.transformer.config["patch_size"])
    grid_height = height // 8 // patch_size
    grid_width = width // 8 // patch_size
    base_size = 512 // 8 // patch_size
    crops = get_resize_crop_region_for_grid((grid_height, grid_width), base_size)
    image_rotary_emb = get_2d_rotary_pos_embed(
        components.transformer.inner_dim // components.transformer.num_heads,
        crops,
        (grid_height, grid_width),
        device=latents.device,
    )
    batch_size = latents.shape[0]
    add_time_ids = get_add_time_ids(
        height, width, prompt_embeds.dtype, latents.device, batch_size=batch_size
    )
    style = torch.zeros(batch_size, device=latents.device, dtype=torch.long)
    if use_cfg:
        add_time_ids = torch.cat([add_time_ids] * 2)
        style = torch.cat([style] * 2)

    for timestep in scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2) if use_cfg else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
        model_timestep = timestep.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
        noise_pred = components.transformer(
            latent_model_input,
            model_timestep,
            prompt_embeds,
            prompt_mask,
            prompt_embeds_2,
            prompt_mask_2,
            add_time_ids,
            style,
            image_rotary_emb,
        )
        noise_pred, _ = noise_pred.chunk(2, dim=1)
        if use_cfg:
            noise_unconditional, noise_text = noise_pred.chunk(2)
            noise_pred = noise_unconditional + guidance * (noise_text - noise_unconditional)
            if guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_text, guidance_rescale)
        latents = scheduler.step(noise_pred, timestep, latents, generator=generator)
    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: Tensor) -> Tensor:
    latents = latents / vae.scaling_factor
    image = vae.decode(latents.to(vae.decoder.conv_in.weight.dtype))
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class HunyuanDiTBackend(TextToImageBackend):
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
        self.components: HunyuanDiTComponents | None = None
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
            raise TypeError(f"Unknown HunyuanDiT options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if o["negative_prompt"] is not None and not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        if type(o["use_resolution_binning"]) is not bool:
            raise TypeError("use_resolution_binning must be a bool")
        for name in ("steps", "width", "height"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        if o["use_resolution_binning"] and (o["height"], o["width"]) not in SUPPORTED_SHAPE:
            o["width"], o["height"] = map_to_standard_shapes(o["width"], o["height"])
        for name in ("width", "height"):
            if o[name] % 16:
                raise ValueError(f"{name} must be a multiple of 16")
        for name in ("guidance", "guidance_rescale"):
            value = o[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite number >= 0")
        seed = o["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        try:
            transformer_config = read_json(self.source / "transformer" / "config.json")
        except FileNotFoundError:
            transformer_config = {"in_channels": 4}
        try:
            vae_config = read_json(self.source / "vae" / "config.json")
        except FileNotFoundError:
            vae_config = {"block_out_channels": [1, 1, 1, 1]}
        vae_scale = 2 ** (len(vae_config.get("block_out_channels", [1, 1, 1, 1])) - 1)
        shape = (
            1,
            int(transformer_config.get("in_channels", 4)),
            o["height"] // vae_scale,
            o["width"] // vae_scale,
        )
        latents = o["latents"]
        if latents is not None and (
            not isinstance(latents, Tensor)
            or not latents.is_floating_point()
            or tuple(latents.shape) != shape
        ):
            raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        o["latent_shape"] = shape
        return o

    def _load(self) -> None:
        self.components = HunyuanDiTComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the HunyuanDiT architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        seed = o["seed"]
        if seed is None:
            seed = secrets.randbits(63)
        generator = torch.Generator("cpu").manual_seed(seed)
        latents = o["latents"]
        if latents is None:
            latents = torch.randn(o["latent_shape"], generator=generator, dtype=torch.float32)
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
            steps=o["steps"],
            guidance=float(o["guidance"]),
            guidance_rescale=float(o["guidance_rescale"]),
            width=o["width"],
            height=o["height"],
            latents=latents.to(torch.float32).to(self.device),
            generator=generator,
        )
        pixels = decode_latents(self.components.vae, final)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": o["steps"],
            "guidance": float(o["guidance"]),
            "guidance_rescale": float(o["guidance_rescale"]),
            "width": o["width"],
            "height": o["height"],
            "negative_prompt": o["negative_prompt"],
            "scheduler": "ddpm",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
