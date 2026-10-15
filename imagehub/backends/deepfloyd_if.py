"""DeepFloyd IF stage-I plus stage-II pixel-space generation backend."""

from __future__ import annotations

import math
import os
import secrets
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
from imagehub.models.t5 import T5EncoderModel
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.ddpm import DDPMScheduler
from imagehub.tokenization_t5 import T5Tokenizer
from imagehub.types import ImageResult, T2IRequest

OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 50,
    "guidance": 7.0,
    "superresolution_steps": 50,
    "superresolution_guidance": 4.0,
    "noise_level": 250,
    "width": 256,
    "height": 256,
    "seed": None,
    "latents": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_STAGE_I_SIZE = 64
_MAX_LENGTH = 77


@dataclass
class IFComponents:
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel
    stage_i: UNet2DConditionModel
    stage_i_scheduler: dict[str, Any]
    stage_ii: UNet2DConditionModel
    stage_ii_scheduler: dict[str, Any]
    image_noising_scheduler: dict[str, Any]

    @classmethod
    def load(cls, root: Path, *, device: torch.device, dtype: torch.dtype) -> IFComponents:
        stage_ii_root = root / "stage_ii"
        required = ("tokenizer", "text_encoder", "unet", "scheduler")
        missing = [name for name in required if not (root / name).is_dir()]
        missing += [
            f"stage_ii/{name}"
            for name in ("unet", "scheduler", "image_noising_scheduler")
            if not (stage_ii_root / name).is_dir()
        ]
        if missing:
            raise CheckpointError(
                f"{root} is not a DeepFloyd IF snapshot (missing {', '.join(missing)}). "
                "Download a snapshot with: python -m imagehub.pretrained deepfloyd-if-i-xl "
                f"{root}"
            )
        return cls(
            tokenizer=T5Tokenizer(root / "tokenizer"),
            text_encoder=T5EncoderModel.from_pretrained(
                root / "text_encoder", device=device, dtype=dtype
            ),
            stage_i=UNet2DConditionModel.from_pretrained(root / "unet", device=device, dtype=dtype),
            stage_i_scheduler=read_json(root / "scheduler" / "scheduler_config.json"),
            stage_ii=UNet2DConditionModel.from_pretrained(
                stage_ii_root / "unet", device=device, dtype=dtype
            ),
            stage_ii_scheduler=read_json(stage_ii_root / "scheduler" / "scheduler_config.json"),
            image_noising_scheduler=read_json(
                stage_ii_root / "image_noising_scheduler" / "scheduler_config.json"
            ),
        )


@torch.inference_mode()
def encode_prompt(
    components: IFComponents,
    prompt: str,
    negative_prompt: str | None,
    *,
    guidance: float,
    device: torch.device,
) -> Tensor:
    texts = [prompt] + ([negative_prompt or ""] if guidance > 1 else [])
    ids, mask = components.tokenizer(texts, max_length=_MAX_LENGTH)
    embeds = components.text_encoder(ids.to(device), mask.to(device))[0]
    if guidance > 1:
        return torch.cat([embeds[1:], embeds[:1]])
    return embeds


def _noise(
    shape: tuple[int, ...], generator: torch.Generator, dtype: torch.dtype, device: torch.device
) -> Tensor:
    return torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device)


@torch.inference_mode()
def _denoise(
    unet: UNet2DConditionModel,
    scheduler_config: dict[str, Any],
    context: Tensor,
    latents: Tensor,
    *,
    steps: int,
    guidance: float,
    generator: torch.Generator,
    upscaled: Tensor | None = None,
    noise_level: int | None = None,
) -> Tensor:
    scheduler = DDPMScheduler(scheduler_config)
    scheduler.set_timesteps(steps)
    use_cfg = guidance > 1
    labels = None
    if noise_level is not None:
        labels = torch.full(
            (latents.shape[0],), noise_level, device=latents.device, dtype=torch.long
        )
        if use_cfg:
            labels = torch.cat([labels, labels])
    for timestep in scheduler.timesteps:
        model_input = latents if upscaled is None else torch.cat([latents, upscaled], dim=1)
        if use_cfg:
            model_input = torch.cat([model_input, model_input])
        predicted = unet(model_input, timestep, context, class_labels=labels)
        if use_cfg:
            uncond, conditional = predicted.chunk(2)
            channels = latents.shape[1]
            uncond_noise, _ = uncond.split(channels, dim=1)
            conditional_noise, variance = conditional.split(channels, dim=1)
            predicted = torch.cat(
                [uncond_noise + guidance * (conditional_noise - uncond_noise), variance], dim=1
            )
        latents = scheduler.step(predicted, timestep, latents, generator=generator)
    return latents


class DeepFloydIFBackend(TextToImageBackend):
    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        model_id: str | None = None,
    ) -> None:
        super().__init__()
        self.source, self.model_id, self.device = Path(model), model_id, resolve_device(device)
        self.dtype = dtype or torch.float32
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be torch.float32, torch.float16 or torch.bfloat16")
        self.components: IFComponents | None = None
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
            raise TypeError(f"Unknown DeepFloyd IF options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if o["negative_prompt"] is not None and not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "superresolution_steps", "width", "height"):
            if type(o[name]) is not int or o[name] <= 0:
                raise ValueError(f"{name} must be a positive int")
        if o["width"] != 256 or o["height"] != 256:
            raise ValueError("DeepFloyd IF stage I + II output is fixed at 256x256")
        if type(o["noise_level"]) is not int or not 0 <= o["noise_level"] < 1000:
            raise ValueError("noise_level must be an int in [0, 1000)")
        for name in ("guidance", "superresolution_guidance"):
            if (
                isinstance(o[name], bool)
                or not isinstance(o[name], int | float)
                or not math.isfinite(o[name])
                or o[name] < 0
            ):
                raise ValueError(f"{name} must be a finite number >= 0")
        if o["seed"] is not None and (type(o["seed"]) is not int or not 0 <= o["seed"] < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        if o["latents"] is not None and (
            not isinstance(o["latents"], Tensor) or tuple(o["latents"].shape) != (1, 3, 64, 64)
        ):
            raise ValueError("latents must be a tensor of shape (1, 3, 64, 64)")
        return o

    def _load(self) -> None:
        self.components = IFComponents.load(self.source, device=self.device, dtype=self.dtype)
        self.snapshot = read_snapshot(self.source)

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        seed = secrets.randbits(63) if o["seed"] is None else o["seed"]
        generator = torch.Generator("cpu").manual_seed(seed)
        context = encode_prompt(
            self.components,
            inputs.prompt,
            o["negative_prompt"],
            guidance=float(o["guidance"]),
            device=self.device,
        ).to(self.dtype)
        initial = o["latents"]
        if initial is None:
            initial = _noise(
                (1, 3, _STAGE_I_SIZE, _STAGE_I_SIZE), generator, self.dtype, self.device
            )
        stage_i = _denoise(
            self.components.stage_i,
            self.components.stage_i_scheduler,
            context,
            initial.to(self.device, self.dtype),
            steps=o["steps"],
            guidance=float(o["guidance"]),
            generator=generator,
        )
        sr_guidance = float(o["superresolution_guidance"])
        sr_context = encode_prompt(
            self.components,
            inputs.prompt,
            o["negative_prompt"],
            guidance=sr_guidance,
            device=self.device,
        ).to(self.dtype)
        # Stage II prepares its initial image before drawing the upscaled-image noise.
        # Keep this order because both draws continue stage I's generator stream.
        stage_ii_initial = _noise((1, 3, 256, 256), generator, self.dtype, self.device)
        upscaled = F.interpolate(stage_i, size=(256, 256), mode="bilinear", align_corners=True)
        noiser = DDPMScheduler(self.components.image_noising_scheduler)
        upscaled = noiser.add_noise(
            upscaled,
            _noise(tuple(upscaled.shape), generator, self.dtype, self.device),
            torch.tensor([o["noise_level"]], device=self.device),
        )
        result = _denoise(
            self.components.stage_ii,
            self.components.stage_ii_scheduler,
            sr_context,
            stage_ii_initial,
            steps=o["superresolution_steps"],
            guidance=sr_guidance,
            generator=generator,
            upscaled=upscaled,
            noise_level=o["noise_level"],
        )
        pixels = (
            ((result / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu() * 255)
            .round()
            .to(torch.uint8)
        )
        snapshot = self.snapshot or {}
        return ImageResult(
            tuple(Image(pixel) for pixel in pixels),
            {
                "model_id": snapshot.get("model_id"),
                "revision": snapshot.get("revision"),
                "seed": seed,
                "steps": o["steps"],
                "superresolution_steps": o["superresolution_steps"],
                "guidance": float(o["guidance"]),
                "superresolution_guidance": sr_guidance,
                "noise_level": o["noise_level"],
                "width": 256,
                "height": 256,
                "scheduler": "ddpm",
                "device": str(self.device),
                "dtype": str(self.dtype).removeprefix("torch."),
            },
        )
