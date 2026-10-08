"""Stable Cascade text-to-image backend over a local two-repository snapshot."""

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
from imagehub.models.paella_vq import PaellaVQModel
from imagehub.models.stable_cascade_unet import StableCascadeUNet
from imagehub.schedulers.ddpm_wuerstchen import DDPMWuerstchenScheduler
from imagehub.tokenization import CLIPTokenizer
from imagehub.types import ImageResult, T2IRequest

OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 12,
    "guidance": 0.0,
    "prior_steps": 60,
    "prior_guidance": 4.0,
    "width": 512,
    "height": 512,
    "seed": None,
    "negative_prompt": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "stable-cascade"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


def _component_names(root: Path, model_id: str | None) -> tuple[str, str]:
    """Names of the prior and decoder directories of the weight set stored in ``root``."""
    spec = MODEL_CATALOG.get(model_id) if model_id is not None else None
    if spec is not None and "prior_component" in spec.config:
        return str(spec.config["prior_component"]), str(spec.config["decoder_component"])
    if not (root / "decoder").is_dir() and (root / "decoder_lite").is_dir():
        return "prior_lite", "decoder_lite"
    return "prior", "decoder"


@dataclass
class StableCascadeComponents:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModelWithProjection
    prior: StableCascadeUNet
    prior_scheduler_config: dict[str, Any]
    resolution_multiple: float
    decoder: StableCascadeUNet
    decoder_scheduler_config: dict[str, Any]
    latent_dim_scale: float
    vqgan: PaellaVQModel

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> StableCascadeComponents:
        root = Path(directory)
        prior_root = root / "prior"
        prior_name, decoder_name = _component_names(root, model_id)
        required = (decoder_name, "scheduler", "vqgan", "prior")
        missing = [name for name in required if not (root / name).is_dir()]
        prior_required = (prior_name, "scheduler", "text_encoder", "tokenizer")
        missing += [f"prior/{name}" for name in prior_required if not (prior_root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Stable Cascade snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            text_config = read_json(prior_root / "text_encoder" / "config.json")
            prior_index = read_json(prior_root / "model_index.json")
            decoder_index = read_json(root / "model_index.json")
            return cls(
                tokenizer=CLIPTokenizer(
                    prior_root / "tokenizer",
                    max_length=int(text_config.get("max_position_embeddings", 77)),
                ),
                text_encoder=CLIPTextModelWithProjection.from_pretrained(
                    prior_root / "text_encoder", device=device, dtype=dtype
                ),
                prior=StableCascadeUNet.from_pretrained(
                    prior_root / prior_name, device=device, dtype=dtype
                ),
                prior_scheduler_config=read_json(
                    prior_root / "scheduler" / "scheduler_config.json"
                ),
                resolution_multiple=float(prior_index.get("resolution_multiple", 42.67)),
                decoder=StableCascadeUNet.from_pretrained(
                    root / decoder_name, device=device, dtype=dtype
                ),
                decoder_scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
                latent_dim_scale=float(decoder_index.get("latent_dim_scale", 10.67)),
                vqgan=PaellaVQModel.from_pretrained(root / "vqgan", device=device, dtype=dtype),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _tokenize(tokenizer: CLIPTokenizer, text: str, device: torch.device) -> tuple[Tensor, Tensor]:
    content_length = min(len(tokenizer.tokenize(text)), tokenizer.max_length - 2)
    input_ids = torch.tensor([tokenizer(text)], dtype=torch.long, device=device)
    mask = torch.zeros((1, tokenizer.max_length), dtype=torch.long, device=device)
    mask[:, : content_length + 2] = 1
    return input_ids, mask


@torch.inference_mode()
def encode_text(
    components: StableCascadeComponents, text: str, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Hidden states before the final layer norm and the pooled projection, as the prior uses."""
    input_ids, mask = _tokenize(components.tokenizer, text, device)
    text_embeds, _, hidden_states = components.text_encoder(
        input_ids, attention_mask=mask, output_hidden_states=True
    )
    return hidden_states[-1], text_embeds.unsqueeze(1)


@torch.inference_mode()
def run_prior(
    components: StableCascadeComponents,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    height: int,
    width: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Return the image embeddings and the pooled text embeddings of both branches."""
    use_cfg = guidance > 1.0
    prompt_embeds, pooled = encode_text(components, prompt, device)
    negative_pooled: Tensor | None = None
    if use_cfg:
        negative_embeds, negative_pooled = encode_text(components, negative_prompt, device)
        text_states = torch.cat([prompt_embeds, negative_embeds])
        text_pooled = torch.cat([pooled, negative_pooled])
    else:
        text_states, text_pooled = prompt_embeds, pooled
    dtype = next(components.prior.parameters()).dtype
    image_channels = int(components.prior.config["clip_image_in_channels"])
    zeros = torch.zeros(1, 1, image_channels, device=device, dtype=dtype)
    image_embeds = torch.cat([zeros, zeros]) if use_cfg else zeros

    shape = (
        1,
        int(components.prior.config["in_channels"]),
        math.ceil(height / components.resolution_multiple),
        math.ceil(width / components.resolution_multiple),
    )
    latents = torch.randn(shape, generator=generator, dtype=dtype, device="cpu").to(device)
    scheduler = DDPMWuerstchenScheduler(components.prior_scheduler_config)
    scheduler.set_timesteps(steps)
    scheduler.timesteps = scheduler.timesteps.to(device)
    latents = latents * scheduler.init_noise_sigma
    for timestep in scheduler.timesteps[:-1]:
        ratio = timestep.expand(latents.size(0)).to(dtype)
        predicted = components.prior(
            torch.cat([latents] * 2) if use_cfg else latents,
            torch.cat([ratio] * 2) if use_cfg else ratio,
            text_pooled,
            clip_text=text_states,
            clip_img=image_embeds,
        )
        if use_cfg:
            text_pred, uncond_pred = predicted.chunk(2)
            predicted = torch.lerp(uncond_pred, text_pred, guidance)
        latents = scheduler.step(predicted, ratio, latents, generator=generator)
    return latents, pooled, negative_pooled


@torch.inference_mode()
def denoise_decoder(
    components: StableCascadeComponents,
    image_embeddings: Tensor,
    pooled: Tensor,
    negative_pooled: Tensor | None,
    *,
    steps: int,
    guidance: float,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    use_cfg = guidance > 1.0
    dtype = next(components.decoder.parameters()).dtype
    if use_cfg:
        assert negative_pooled is not None
        pooled = torch.cat([pooled, negative_pooled])
        effnet = torch.cat([image_embeddings, torch.zeros_like(image_embeddings)])
    else:
        effnet = image_embeddings
    shape = (
        image_embeddings.shape[0],
        4,
        int(image_embeddings.shape[2] * components.latent_dim_scale),
        int(image_embeddings.shape[3] * components.latent_dim_scale),
    )
    latents = torch.randn(shape, generator=generator, dtype=dtype, device="cpu").to(device)
    scheduler = DDPMWuerstchenScheduler(components.decoder_scheduler_config)
    scheduler.set_timesteps(steps)
    scheduler.timesteps = scheduler.timesteps.to(device)
    latents = latents * scheduler.init_noise_sigma
    for timestep in scheduler.timesteps[:-1]:
        ratio = timestep.expand(latents.size(0)).to(dtype)
        predicted = components.decoder(
            torch.cat([latents] * 2) if use_cfg else latents,
            torch.cat([ratio] * 2) if use_cfg else ratio,
            pooled,
            effnet=effnet,
        )
        if use_cfg:
            text_pred, uncond_pred = predicted.chunk(2)
            predicted = torch.lerp(uncond_pred, text_pred, guidance)
        latents = scheduler.step(predicted, ratio, latents, generator=generator)
    return latents


@torch.inference_mode()
def decode_latents(vqgan: PaellaVQModel, latents: Tensor) -> Tensor:
    image = vqgan.decode(latents * vqgan.scale_factor).clamp(0, 1)
    return (image.permute(0, 2, 3, 1).float().cpu() * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class StableCascadeBackend(TextToImageBackend):
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
        self.components: StableCascadeComponents | None = None
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
            raise TypeError(f"Unknown Stable Cascade options: {', '.join(unknown)}")
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
        if result["guidance"] > 1 and not result["prior_guidance"] > 1:
            raise ValueError(
                "guidance above 1 needs the negative text embeddings, so prior_guidance must be "
                "above 1 as well"
            )
        seed = result["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        return result

    def _load(self) -> None:
        self.components = StableCascadeComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the Stable Cascade architecture, so it was loaded",
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
        negative_prompt = options["negative_prompt"]
        embeddings, pooled, negative_pooled = run_prior(
            self.components,
            prompt=inputs.prompt,
            negative_prompt="" if negative_prompt is None else negative_prompt,
            steps=options["prior_steps"],
            guidance=float(options["prior_guidance"]),
            height=options["height"],
            width=options["width"],
            generator=generator,
            device=self.device,
        )
        latents = denoise_decoder(
            self.components,
            embeddings,
            pooled,
            negative_pooled,
            steps=options["steps"],
            guidance=float(options["guidance"]),
            generator=generator,
            device=self.device,
        )
        pixels = decode_latents(self.components.vqgan, latents)
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
            "negative_prompt": negative_prompt,
            "scheduler": "ddpm_wuerstchen",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
