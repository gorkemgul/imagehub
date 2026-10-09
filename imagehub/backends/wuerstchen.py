"""Würstchen v2 text-to-image backend over a local two-repository snapshot."""

from __future__ import annotations

import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_cascade import decode_latents
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.paella_vq import PaellaVQModel
from imagehub.models.wuerstchen import WuerstchenDiffNeXt, WuerstchenPrior
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
# Published values of the pipeline configuration, used to check sizes before loading.
RESOLUTION_MULTIPLE = 42.67
LATENT_DIM_SCALE = 10.67
# Patch size 2 and three stride-2 downsampling stages of the decoder.
_LATENT_MULTIPLE = 16
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "wuerstchen-v2"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class WuerstchenComponents:
    prior_tokenizer: CLIPTokenizer
    prior_text_encoder: CLIPTextModel
    prior: WuerstchenPrior
    prior_scheduler_config: dict[str, Any]
    resolution_multiple: float
    latent_mean: float
    latent_std: float
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    decoder: WuerstchenDiffNeXt
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
    ) -> WuerstchenComponents:
        root = Path(directory)
        prior_root = root / "prior"
        required = ("decoder", "scheduler", "text_encoder", "tokenizer", "vqgan", "prior")
        missing = [name for name in required if not (root / name).is_dir()]
        prior_required = ("prior", "scheduler", "text_encoder", "tokenizer")
        missing += [f"prior/{name}" for name in prior_required if not (prior_root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Würstchen snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            prior_index = read_json(prior_root / "model_index.json")
            decoder_index = read_json(root / "model_index.json")
            return cls(
                prior_tokenizer=CLIPTokenizer(
                    prior_root / "tokenizer",
                    max_length=int(
                        read_json(prior_root / "text_encoder" / "config.json").get(
                            "max_position_embeddings", 77
                        )
                    ),
                ),
                prior_text_encoder=CLIPTextModel.from_pretrained(
                    prior_root / "text_encoder", device=device, dtype=dtype
                ),
                prior=WuerstchenPrior.from_pretrained(
                    prior_root / "prior", device=device, dtype=dtype
                ),
                prior_scheduler_config=read_json(
                    prior_root / "scheduler" / "scheduler_config.json"
                ),
                resolution_multiple=float(
                    prior_index.get("resolution_multiple", RESOLUTION_MULTIPLE)
                ),
                latent_mean=float(prior_index.get("latent_mean", 42.0)),
                latent_std=float(prior_index.get("latent_std", 1.0)),
                tokenizer=CLIPTokenizer(
                    root / "tokenizer",
                    max_length=int(
                        read_json(root / "text_encoder" / "config.json").get(
                            "max_position_embeddings", 77
                        )
                    ),
                ),
                text_encoder=CLIPTextModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                decoder=WuerstchenDiffNeXt.from_pretrained(
                    root / "decoder", device=device, dtype=dtype
                ),
                decoder_scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
                latent_dim_scale=float(decoder_index.get("latent_dim_scale", LATENT_DIM_SCALE)),
                vqgan=PaellaVQModel.from_pretrained(root / "vqgan", device=device, dtype=dtype),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def prior_latent_size(extent: int, resolution_multiple: float) -> int:
    return math.ceil(extent / resolution_multiple)


def decoder_latent_size(prior_size: int, latent_dim_scale: float) -> int:
    return int(prior_size * latent_dim_scale)


def _check_sizes(
    height: int, width: int, resolution_multiple: float, latent_dim_scale: float
) -> None:
    for name, extent in (("height", height), ("width", width)):
        size = decoder_latent_size(prior_latent_size(extent, resolution_multiple), latent_dim_scale)
        if size % _LATENT_MULTIPLE:
            raise ValueError(
                f"{name}={extent} gives a decoder latent size of {size}, which must be a "
                f"multiple of {_LATENT_MULTIPLE}; multiples of 128 always work"
            )


def _tokenize(tokenizer: CLIPTokenizer, text: str, device: torch.device) -> tuple[Tensor, Tensor]:
    content_length = min(len(tokenizer.tokenize(text)), tokenizer.max_length - 2)
    input_ids = torch.tensor([tokenizer(text)], dtype=torch.long, device=device)
    mask = torch.zeros((1, tokenizer.max_length), dtype=torch.long, device=device)
    mask[:, : content_length + 2] = 1
    return input_ids, mask


@torch.inference_mode()
def encode_text(
    tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, text: str, device: torch.device
) -> Tensor:
    """Final-layer-norm hidden states of the padded prompt, masked as the reference does."""
    input_ids, mask = _tokenize(tokenizer, text, device)
    return cast(Tensor, text_encoder(input_ids, attention_mask=mask))


def _scheduler(config: dict[str, Any], steps: int, device: torch.device) -> DDPMWuerstchenScheduler:
    scheduler = DDPMWuerstchenScheduler(config)
    scheduler.set_timesteps(steps)
    scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler


@torch.inference_mode()
def run_prior(
    components: WuerstchenComponents,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    height: int,
    width: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    """Return the image embeddings for the prompt."""
    use_cfg = guidance > 1.0
    states = encode_text(components.prior_tokenizer, components.prior_text_encoder, prompt, device)
    if use_cfg:
        negative = encode_text(
            components.prior_tokenizer, components.prior_text_encoder, negative_prompt, device
        )
        states = torch.cat([states, negative])
    dtype = states.dtype
    shape = (
        1,
        int(components.prior.config["c_in"]),
        prior_latent_size(height, components.resolution_multiple),
        prior_latent_size(width, components.resolution_multiple),
    )
    latents = torch.randn(shape, generator=generator, dtype=dtype, device="cpu").to(device)
    scheduler = _scheduler(components.prior_scheduler_config, steps, device)
    latents = latents * scheduler.init_noise_sigma
    for timestep in scheduler.timesteps[:-1]:
        ratio = timestep.expand(latents.size(0)).to(dtype)
        predicted = components.prior(
            torch.cat([latents] * 2) if use_cfg else latents,
            torch.cat([ratio] * 2) if use_cfg else ratio,
            states,
        )
        if use_cfg:
            text_pred, uncond_pred = predicted.chunk(2)
            predicted = torch.lerp(uncond_pred, text_pred, guidance)
        latents = scheduler.step(predicted, ratio, latents, generator=generator)
    return latents * components.latent_mean - components.latent_std


@torch.inference_mode()
def denoise_decoder(
    components: WuerstchenComponents,
    image_embeddings: Tensor,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    use_cfg = guidance > 1.0
    dtype = next(components.decoder.parameters()).dtype
    states = encode_text(components.tokenizer, components.text_encoder, prompt, device)
    if use_cfg:
        negative = encode_text(
            components.tokenizer, components.text_encoder, negative_prompt, device
        )
        states = torch.cat([states, negative])
        effnet = torch.cat([image_embeddings, torch.zeros_like(image_embeddings)])
    else:
        effnet = image_embeddings
    shape = (
        image_embeddings.shape[0],
        4,
        decoder_latent_size(image_embeddings.shape[2], components.latent_dim_scale),
        decoder_latent_size(image_embeddings.shape[3], components.latent_dim_scale),
    )
    latents = torch.randn(shape, generator=generator, dtype=dtype, device="cpu").to(device)
    scheduler = _scheduler(components.decoder_scheduler_config, steps, device)
    latents = latents * scheduler.init_noise_sigma
    for timestep in scheduler.timesteps[:-1]:
        ratio = timestep.expand(latents.size(0)).to(dtype)
        predicted = components.decoder(
            torch.cat([latents] * 2) if use_cfg else latents,
            torch.cat([ratio] * 2) if use_cfg else ratio,
            effnet,
            states,
        )
        assert isinstance(predicted, Tensor)
        if use_cfg:
            text_pred, uncond_pred = predicted.chunk(2)
            predicted = torch.lerp(uncond_pred, text_pred, guidance)
        latents = scheduler.step(predicted, ratio, latents, generator=generator)
    return latents


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class WuerstchenBackend(TextToImageBackend):
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
        self.components: WuerstchenComponents | None = None
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
            raise TypeError(f"Unknown Würstchen options: {', '.join(unknown)}")
        result = self.option_defaults | merged
        if result["negative_prompt"] is not None and not isinstance(result["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "prior_steps", "width", "height"):
            _positive_int(result, name)
        for name in ("steps", "prior_steps"):
            if result[name] > 1000:
                raise ValueError(f"{name} must be at most 1000")
        for name in ("guidance", "prior_guidance"):
            value = result[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite number >= 0")
        _check_sizes(result["height"], result["width"], RESOLUTION_MULTIPLE, LATENT_DIM_SCALE)
        seed = result["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        return result

    def _load(self) -> None:
        self.components = WuerstchenComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        _check_sizes(
            options["height"],
            options["width"],
            self.components.resolution_multiple,
            self.components.latent_dim_scale,
        )
        seed = options["seed"]
        if seed is None:
            seed = secrets.randbits(63)
        generator = torch.Generator("cpu").manual_seed(seed)
        negative_prompt = options["negative_prompt"]
        negative = "" if negative_prompt is None else negative_prompt
        embeddings = run_prior(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=negative,
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
            prompt=inputs.prompt,
            negative_prompt=negative,
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
            "width": int(pixels.shape[2]),
            "height": int(pixels.shape[1]),
            "negative_prompt": negative_prompt,
            "scheduler": "ddpm_wuerstchen",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
