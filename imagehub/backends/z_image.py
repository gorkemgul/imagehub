"""Z-Image-Turbo text-to-image backend over a local diffusers-layout snapshot."""

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
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.lora_formats import LoRAOption, apply_lora, parse_lora_option
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.qwen3 import Qwen3Model
from imagehub.models.z_image_transformer import ZImageTransformer2DModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler, z_image_sigmas
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 9,
    "guidance": 0.0,
    "cfg_normalization": False,
    "cfg_truncation": 1.0,
    "width": 1024,
    "height": 1024,
    "num_images": 1,
    "seed": None,
    "latents": None,
    "max_sequence_length": 512,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_LATENT_CHANNELS = 16
_LATENT_SCALE = 8


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "z-image-turbo"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class ZImageComponents:
    tokenizer: Qwen2Tokenizer
    text_encoder: Qwen3Model
    transformer: ZImageTransformer2DModel
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
    ) -> ZImageComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Z-Image snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            return cls(
                tokenizer=Qwen2Tokenizer(root / "tokenizer"),
                text_encoder=Qwen3Model.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                transformer=ZImageTransformer2DModel.from_pretrained(
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
def encode_prompt(
    components: ZImageComponents, prompt: str, device: torch.device, max_sequence_length: int
) -> list[Tensor]:
    input_ids, mask = components.tokenizer(
        components.tokenizer.chat_prompt(prompt), max_sequence_length
    )
    input_ids, mask = input_ids.to(device), mask.to(device)
    _, hidden_states = components.text_encoder(input_ids, mask, output_hidden_states=True)
    embeds = hidden_states[-2]
    return [embeds[index][mask[index]] for index in range(len(embeds))]


@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: Tensor) -> Tensor:
    """Decode final latents to uint8 images of shape (batch, height, width, 3) on CPU."""
    latents = latents / vae.scaling_factor + (vae.shift_factor or 0.0)
    image = vae.decode(latents.to(vae.encoder.conv_in.weight.dtype))
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


@torch.inference_mode()
def denoise(
    components: ZImageComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    cfg_normalization: bool,
    cfg_truncation: float,
    latents: Tensor,
    max_sequence_length: int,
) -> Tensor:
    """Run the Z-Image flow-match loop from initial float32 noise."""
    dtype = cast(torch.dtype, components.transformer.all_x_embedder["2-1"].weight.dtype)
    batch = latents.shape[0]
    prompt_embeds = encode_prompt(components, prompt, latents.device, max_sequence_length)
    prompt_embeds = [embed.to(dtype) for embed in prompt_embeds for _ in range(batch)]
    use_cfg = guidance > 0
    if use_cfg:
        negative_embeds = encode_prompt(
            components, negative_prompt or "", latents.device, max_sequence_length
        )
        negative_embeds = [embed.to(dtype) for embed in negative_embeds for _ in range(batch)]
    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    scheduler.set_timesteps(steps, sigmas=z_image_sigmas(steps))
    scheduler.set_begin_index(0)
    for _index, timestep in enumerate(scheduler.timesteps):
        timestep_model = (1000 - timestep).expand(batch) / 1000
        current_guidance = guidance if (1000 - timestep.float()) / 1000 <= cfg_truncation else 0.0
        apply_cfg = use_cfg and current_guidance > 0
        if apply_cfg:
            model_input = latents.to(dtype).repeat(2, 1, 1, 1)
            embeds = prompt_embeds + negative_embeds
            model_timestep = timestep_model.repeat(2)
        else:
            model_input = latents.to(dtype)
            embeds = prompt_embeds
            model_timestep = timestep_model
        outputs = components.transformer(
            list(model_input.unsqueeze(2).unbind(dim=0)), model_timestep, embeds
        )
        if apply_cfg:
            predictions = []
            for positive, negative in zip(outputs[:batch], outputs[batch:], strict=True):
                positive, negative = positive.float(), negative.float()
                prediction = positive + current_guidance * (positive - negative)
                if cfg_normalization:
                    maximum = torch.linalg.vector_norm(positive) * float(cfg_normalization)
                    norm = torch.linalg.vector_norm(prediction)
                    if norm > maximum:
                        prediction = prediction * (maximum / norm)
                predictions.append(prediction)
            noise_pred = torch.stack(predictions)
        else:
            noise_pred = torch.stack([output.float() for output in outputs])
        latents = scheduler.step(-noise_pred.squeeze(2), timestep, latents)
        assert latents.dtype == torch.float32
    return latents


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class ZImageBackend(TextToImageBackend):
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
        self.components: ZImageComponents | None = None
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
            raise TypeError(f"Unknown Z-Image options: {', '.join(unknown)}")
        o = self.option_defaults | merged
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
        if type(o["cfg_normalization"]) is not bool:
            raise TypeError("cfg_normalization must be a bool")
        truncation = o["cfg_truncation"]
        if (
            isinstance(truncation, bool)
            or not isinstance(truncation, int | float)
            or not 0 <= truncation <= 1
        ):
            raise ValueError("cfg_truncation must be a number in [0, 1]")
        seed, latents = o["seed"], o["latents"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        shape = (
            o["num_images"],
            _LATENT_CHANNELS,
            o["height"] // _LATENT_SCALE,
            o["width"] // _LATENT_SCALE,
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
        return o

    def _load(self) -> None:
        self.components = ZImageComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the Z-Image architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )
        if self.lora:
            # ZImagePipeline.load_lora_weights accepts transformer adapters only.
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
            latents = torch.randn(
                (
                    o["num_images"],
                    _LATENT_CHANNELS,
                    o["height"] // _LATENT_SCALE,
                    o["width"] // _LATENT_SCALE,
                ),
                generator=torch.Generator("cpu").manual_seed(seed),
                dtype=torch.float32,
            )
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
            steps=o["steps"],
            guidance=float(o["guidance"]),
            cfg_normalization=o["cfg_normalization"],
            cfg_truncation=float(o["cfg_truncation"]),
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
