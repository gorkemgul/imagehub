"""Kandinsky 2.2 text-to-image backend over a local two-repository snapshot."""

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
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.clip_vision import CLIPVisionModelWithProjection
from imagehub.models.prior_transformer import PriorTransformer
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.models.vq_model import VQModel
from imagehub.schedulers.ddpm import DDPMScheduler
from imagehub.schedulers.unclip import UnCLIPScheduler
from imagehub.tokenization import CLIPTokenizer
from imagehub.types import ImageResult, T2IRequest

OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 100,
    "guidance": 4.0,
    "prior_steps": 25,
    "prior_guidance": 4.0,
    "width": 512,
    "height": 512,
    "seed": None,
    "negative_prompt": None,
    "latents": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "kandinsky-2.2"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class Kandinsky22Components:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModelWithProjection
    image_encoder: CLIPVisionModelWithProjection
    prior: PriorTransformer
    prior_scheduler_config: dict[str, Any]
    unet: UNet2DConditionModel
    movq: VQModel
    scheduler_config: dict[str, Any]

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> Kandinsky22Components:
        root = Path(directory)
        required = ("movq", "scheduler", "unet", "prior")
        missing = [name for name in required if not (root / name).is_dir()]
        prior_root = root / "prior"
        prior_required = ("image_encoder", "prior", "scheduler", "text_encoder", "tokenizer")
        missing += [f"prior/{name}" for name in prior_required if not (prior_root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Kandinsky 2.2 snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            text_config = read_json(prior_root / "text_encoder" / "config.json")
            return cls(
                tokenizer=CLIPTokenizer(
                    prior_root / "tokenizer",
                    max_length=int(text_config.get("max_position_embeddings", 77)),
                ),
                text_encoder=CLIPTextModelWithProjection.from_pretrained(
                    prior_root / "text_encoder", device=device, dtype=dtype
                ),
                image_encoder=CLIPVisionModelWithProjection.from_pretrained(
                    prior_root / "image_encoder", device=device, dtype=dtype
                ),
                prior=PriorTransformer.from_pretrained(
                    prior_root / "prior", device=device, dtype=dtype
                ),
                prior_scheduler_config=read_json(
                    prior_root / "scheduler" / "scheduler_config.json"
                ),
                unet=UNet2DConditionModel.from_pretrained(
                    root / "unet", device=device, dtype=dtype
                ),
                movq=VQModel.from_pretrained(root / "movq", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _tokenize(tokenizer: CLIPTokenizer, text: str, device: torch.device) -> tuple[Tensor, Tensor]:
    content_length = min(len(tokenizer.tokenize(text)), tokenizer.max_length - 2)
    input_ids = torch.tensor([tokenizer(text)], dtype=torch.long, device=device)
    mask = torch.zeros((1, tokenizer.max_length), dtype=torch.bool, device=device)
    mask[:, : content_length + 2] = True
    return input_ids, mask


@torch.inference_mode()
def encode_prompt(
    components: Kandinsky22Components,
    prompts: list[str],
    negative_prompts: list[str] | None,
    device: torch.device,
    do_classifier_free_guidance: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    ids_and_masks = [_tokenize(components.tokenizer, prompt, device) for prompt in prompts]
    input_ids = torch.cat([item[0] for item in ids_and_masks])
    text_mask = torch.cat([item[1] for item in ids_and_masks])
    prompt_embeds, hidden_states = components.text_encoder(input_ids)
    if do_classifier_free_guidance:
        uncond = negative_prompts or [""] * len(prompts)
        ids_and_masks = [_tokenize(components.tokenizer, prompt, device) for prompt in uncond]
        uncond_ids = torch.cat([item[0] for item in ids_and_masks])
        uncond_mask = torch.cat([item[1] for item in ids_and_masks])
        negative_embeds, negative_hidden = components.text_encoder(uncond_ids)
        prompt_embeds = torch.cat([negative_embeds, prompt_embeds])
        hidden_states = torch.cat([negative_hidden, hidden_states])
        text_mask = torch.cat([uncond_mask, text_mask])
    return prompt_embeds, hidden_states, text_mask


@torch.inference_mode()
def get_zero_embed(
    components: Kandinsky22Components, batch_size: int, device: torch.device
) -> Tensor:
    image_size = int(components.image_encoder.config["image_size"])
    dtype = components.image_encoder.visual_projection.weight.dtype
    pixels = torch.zeros(1, 3, image_size, image_size, dtype=dtype, device=device)
    image_embeds, _ = components.image_encoder(pixels)
    return image_embeds.repeat(batch_size, 1)


@torch.inference_mode()
def run_prior(
    components: Kandinsky22Components,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    latents: Tensor | None,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    prompts = [prompt] if negative_prompt is None else [prompt, negative_prompt]
    negative_prompts = None if negative_prompt is None else [negative_prompt] * 2
    use_cfg = guidance > 1.0
    prompt_embeds, hidden_states, text_mask = encode_prompt(
        components, prompts, negative_prompts, device, use_cfg
    )
    batch_size = len(prompts)
    embedding_dim = int(components.prior.config["embedding_dim"])
    if latents is None:
        latents = torch.randn(
            (batch_size, embedding_dim),
            generator=generator,
            dtype=prompt_embeds.dtype,
            device="cpu",
        ).to(device)
    else:
        latents = latents.to(device=device, dtype=prompt_embeds.dtype)
    scheduler = UnCLIPScheduler(components.prior_scheduler_config)
    scheduler.set_timesteps(steps)
    for index, timestep in enumerate(scheduler.timesteps):
        model_input = torch.cat([latents] * 2) if use_cfg else latents
        predicted = components.prior(
            model_input,
            timestep,
            prompt_embeds,
            encoder_hidden_states=hidden_states,
            attention_mask=text_mask,
        )
        if use_cfg:
            unconditional, conditional = predicted.chunk(2)
            predicted = unconditional + guidance * (conditional - unconditional)
        previous = scheduler.timesteps[index + 1] if index + 1 < len(scheduler.timesteps) else None
        latents = scheduler.step(
            predicted,
            int(timestep),
            latents,
            prev_timestep=None if previous is None else int(previous),
            generator=generator,
        )
    image_embeddings = components.prior.post_process_latents(latents)
    if negative_prompt is None:
        return image_embeddings, get_zero_embed(components, image_embeddings.shape[0], device)
    positive, negative = image_embeddings.chunk(2)
    return positive, negative


def _latent_size(size: int, scale_factor: int) -> int:
    return math.ceil(size / scale_factor**2) * scale_factor


@torch.inference_mode()
def denoise_decoder(
    components: Kandinsky22Components,
    image_embeds: Tensor,
    negative_image_embeds: Tensor,
    *,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    generator: torch.Generator,
) -> Tensor:
    use_cfg = guidance > 1.0
    if use_cfg:
        image_embeds = torch.cat([negative_image_embeds, image_embeds]).to(
            dtype=components.unet.conv_in.weight.dtype
        )
    scale_factor = 2 ** (len(components.movq.config["block_out_channels"]) - 1)
    latent_height = _latent_size(height, scale_factor)
    latent_width = _latent_size(width, scale_factor)
    latents = torch.randn(
        (1, int(components.unet.config["in_channels"]), latent_height, latent_width),
        generator=generator,
        dtype=image_embeds.dtype,
        device="cpu",
    ).to(image_embeds.device)
    scheduler = DDPMScheduler(components.scheduler_config)
    scheduler.set_timesteps(steps)
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents] * 2) if use_cfg else latents
        noise_pred = components.unet(model_input, timestep, None, image_embeds=image_embeds)
        if use_cfg:
            noise, variance = noise_pred.split(latents.shape[1], dim=1)
            noise_unconditional, noise_text = noise.chunk(2)
            _, variance_text = variance.chunk(2)
            noise_pred = noise_unconditional + guidance * (noise_text - noise_unconditional)
            noise_pred = torch.cat([noise_pred, variance_text], dim=1)
        if components.scheduler_config.get("variance_type") not in ("learned", "learned_range"):
            noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)
        latents = scheduler.step(noise_pred, timestep, latents, generator=generator)
    return latents


@torch.inference_mode()
def decode_latents(movq: VQModel, latents: Tensor) -> Tensor:
    image = movq.decode(latents, force_not_quantize=True)
    image = (image * 0.5 + 0.5).clamp(0, 1)
    return (image.permute(0, 2, 3, 1).float().cpu() * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class Kandinsky22Backend(TextToImageBackend):
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
        self.components: Kandinsky22Components | None = None
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
            raise TypeError(f"Unknown Kandinsky 2.2 options: {', '.join(unknown)}")
        result = self.option_defaults | merged
        if result["negative_prompt"] is not None and not isinstance(result["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "prior_steps", "width", "height"):
            _positive_int(result, name)
        for name in ("steps", "prior_steps"):
            if result[name] > 1000:
                raise ValueError(f"{name} must be at most 1000")
        for name in ("width", "height"):
            if result[name] % 8:
                raise ValueError(f"{name} must be a multiple of 8")
        for name in ("guidance", "prior_guidance"):
            value = result[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite number >= 0")
        seed = result["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        try:
            prior_config = read_json(self.source / "prior" / "prior" / "config.json")
        except FileNotFoundError:
            prior_config = {"embedding_dim": 1280}
        shape = (
            2 if result["negative_prompt"] is not None else 1,
            int(prior_config.get("embedding_dim", 1280)),
        )
        latents = result["latents"]
        if latents is not None and (
            not isinstance(latents, Tensor)
            or not latents.is_floating_point()
            or tuple(latents.shape) != shape
        ):
            raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        return result

    def _load(self) -> None:
        self.components = Kandinsky22Components.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the Kandinsky 2.2 architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed = options["seed"]
        if seed is None:
            seed = secrets.randbits(63)
        generator = torch.Generator("cpu").manual_seed(seed)
        image_embeds, negative_image_embeds = run_prior(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=options["negative_prompt"],
            steps=options["prior_steps"],
            guidance=float(options["prior_guidance"]),
            latents=options["latents"],
            generator=generator,
            device=self.device,
        )
        latents = denoise_decoder(
            self.components,
            image_embeds,
            negative_image_embeds,
            steps=options["steps"],
            guidance=float(options["guidance"]),
            width=options["width"],
            height=options["height"],
            generator=generator,
        )
        pixels = decode_latents(self.components.movq, latents)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": options["steps"],
            "guidance": float(options["guidance"]),
            "prior_steps": options["prior_steps"],
            "prior_guidance": float(options["prior_guidance"]),
            "width": options["width"],
            "height": options["height"],
            "negative_prompt": options["negative_prompt"],
            "scheduler": "ddpm",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
