"""OmniGen text-to-image backend (with an optional input image) over a local diffusers snapshot."""

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
from imagehub.backends.omnigen_inputs import (
    OmniGenInputs,
    build_inputs,
    encode_input_images,
    preprocess_input_image,
    validate_prompt,
)
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image, as_pil
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.omnigen_transformer import OmniGenTransformer2DModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_omnigen import OmniGenTokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "transformer", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 50,
    "guidance": 2.5,
    "image_guidance": 1.6,
    "width": 1024,
    "height": 1024,
    "seed": None,
    "image": None,
    "max_input_image_size": 1024,
    "use_input_image_size_as_output": False,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_PATCH_PIXELS = 16


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "omnigen"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


def omnigen_sigmas(steps: int) -> list[float]:
    """``np.linspace(1, 0, steps + 1)[:steps]`` in float64."""
    step = -1.0 / steps
    return [index * step + 1.0 for index in range(steps)]


@dataclass
class OmniGenComponents:
    tokenizer: OmniGenTokenizer
    transformer: OmniGenTransformer2DModel
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
    ) -> OmniGenComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not an OmniGen snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            class_name = read_json(root / "model_index.json").get("_class_name")
            if class_name != "OmniGenPipeline":
                raise CheckpointError(f"Unsupported OmniGen pipeline class {class_name!r}")
            return cls(
                tokenizer=OmniGenTokenizer(root / "tokenizer"),
                transformer=OmniGenTransformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


@torch.inference_mode()
def denoise(
    components: OmniGenComponents,
    inputs: OmniGenInputs,
    input_img_latents: list[Tensor],
    *,
    steps: int,
    guidance: float,
    image_guidance: float,
    latents: Tensor,
) -> Tensor:
    transformer = components.transformer
    dtype = transformer.patch_embedding.output_image_proj.weight.dtype
    device = latents.device
    use_image = len(inputs.input_pixel_values) > 0
    copies = 3 if use_image else 2
    scheduler = FlowMatchEulerDiscreteScheduler(
        components.scheduler_config, allow_inverted_sigmas=True
    )
    scheduler.set_timesteps(steps, sigmas=omnigen_sigmas(steps))
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    position_ids = inputs.position_ids.to(device)
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents] * copies).to(dtype)
        noise_pred = transformer(
            model_input,
            timestep.expand(model_input.shape[0]),
            input_ids,
            input_img_latents,
            inputs.input_image_sizes,
            attention_mask,
            position_ids,
        )
        if use_image:
            cond, uncond, img_cond = torch.split(noise_pred, len(noise_pred) // 3, dim=0)
            noise_pred = (
                uncond + image_guidance * (img_cond - uncond) + guidance * (cond - img_cond)
            )
        else:
            cond, uncond = torch.split(noise_pred, len(noise_pred) // 2, dim=0)
            noise_pred = uncond + guidance * (cond - uncond)
        latents = scheduler.step(noise_pred, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: Tensor) -> Tensor:
    latents = latents.to(vae.decoder.conv_in.weight.dtype) / vae.scaling_factor
    image = vae.decode(latents)
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


def _finite_number(options: dict[str, Any], name: str) -> None:
    value = options[name]
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite number >= 0")


class OmniGenBackend(TextToImageBackend):
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
        self.components: OmniGenComponents | None = None
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
            raise TypeError(f"Unknown OmniGen options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        for name in ("steps", "width", "height", "max_input_image_size"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        for name in ("guidance", "image_guidance"):
            _finite_number(o, name)
        if type(o["use_input_image_size_as_output"]) is not bool:
            raise TypeError("use_input_image_size_as_output must be a bool")
        seed = o["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        try:
            transformer_config = read_json(self.source / "transformer" / "config.json")
        except FileNotFoundError:
            transformer_config = {}
        try:
            blocks = read_json(self.source / "vae" / "config.json").get(
                "block_out_channels", [1, 1, 1, 1]
            )
        except FileNotFoundError:
            blocks = [1, 1, 1, 1]
        vae_scale = 2 ** (len(blocks) - 1)
        patch = int(transformer_config.get("patch_size", 2))
        pos_max = int(transformer_config.get("pos_embed_max_size", 192))
        multiple = vae_scale * patch
        limit = pos_max * multiple
        for name in ("width", "height"):
            if o[name] % multiple:
                raise ValueError(f"{name} must be a multiple of {multiple}")
            if o[name] > limit:
                raise ValueError(f"{name} must be at most {limit}")
        if o["image"] is not None:
            o["image"] = as_pil(o["image"])
        validate_prompt(inputs.prompt, o["image"] is not None)
        o["pixels"] = None
        if o["image"] is not None:
            pixels = preprocess_input_image(o["image"], o["max_input_image_size"])
            if max(pixels.shape[-2:]) > limit:
                raise ValueError(f"the preprocessed input image must be at most {limit} pixels")
            o["pixels"] = pixels
        elif o["use_input_image_size_as_output"]:
            raise ValueError("use_input_image_size_as_output requires an input image")
        if o["pixels"] is not None and o["use_input_image_size_as_output"]:
            o["height"], o["width"] = (int(x) for x in o["pixels"].shape[-2:])
        o["latent_shape"] = (
            1,
            int(transformer_config.get("in_channels", 4)),
            o["height"] // vae_scale,
            o["width"] // vae_scale,
        )
        return o

    def _load(self) -> None:
        self.components = OmniGenComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the OmniGen architecture, so it was loaded",
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
        pixels = o["pixels"]
        has_image = pixels is not None
        processed = build_inputs(
            self.components.tokenizer,
            inputs.prompt,
            [pixels] if has_image else None,
            height=o["height"],
            width=o["width"],
            use_img_cfg=has_image,
        )
        with torch.inference_mode():
            # The reference samples the input-image posterior from the global generator; here a
            # dedicated generator with the request seed plays that role (the same stream as
            # ``torch.manual_seed(seed)``), consumed once per placeholder occurrence.
            image_generator = torch.Generator("cpu").manual_seed(seed)
            image_latents = [
                x.to(self.device)
                for x in encode_input_images(
                    self.components.vae,
                    [x.to(self.device) for x in processed.input_pixel_values],
                    image_generator,
                )
            ]
            generator = torch.Generator("cpu").manual_seed(seed)
            latents = torch.randn(o["latent_shape"], generator=generator, dtype=torch.float32)
        final = denoise(
            self.components,
            processed,
            image_latents,
            steps=o["steps"],
            guidance=float(o["guidance"]),
            image_guidance=float(o["image_guidance"]),
            latents=latents.to(self.device),
        )
        decoded = decode_latents(self.components.vae, final)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": o["steps"],
            "guidance": float(o["guidance"]),
            "image_guidance": float(o["image_guidance"]) if has_image else None,
            "width": o["width"],
            "height": o["height"],
            "image_input": has_image,
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        if has_image:
            metadata["input_height"], metadata["input_width"] = (int(x) for x in pixels.shape[-2:])
        return ImageResult(tuple(Image(pixel) for pixel in decoded), metadata)
