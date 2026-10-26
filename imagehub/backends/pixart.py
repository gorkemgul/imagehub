"""PixArt Alpha and Sigma text-to-image backend over a local diffusers snapshot."""

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
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.pixart_transformer import PixArtTransformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.dpmsolver_multistep import DPMSolverMultistepScheduler
from imagehub.schedulers.lcm import LCMScheduler
from imagehub.tokenization_t5 import T5Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 20,
    "guidance": 4.5,
    "width": 1024,
    "height": 1024,
    "seed": None,
    "negative_prompt": None,
    "use_resolution_binning": True,
    "max_sequence_length": 300,
    "latents": None,
}
ASPECT_RATIO_1024_BIN = {
    "0.25": [512.0, 2048.0],
    "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0],
    "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0],
    "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0],
    "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "3.0": [1728.0, 576.0],
    "4.0": [2048.0, 512.0],
}
ASPECT_RATIO_512_BIN = {
    key: [height / 2, width / 2] for key, (height, width) in ASPECT_RATIO_1024_BIN.items()
}
ASPECT_RATIO_256_BIN = {
    key: [height / 4, width / 4] for key, (height, width) in ASPECT_RATIO_1024_BIN.items()
}
ASPECT_RATIO_2048_BIN = {
    "0.25": [1024.0, 4096.0],
    "0.26": [1024.0, 3968.0],
    "0.27": [1024.0, 3840.0],
    "0.28": [1024.0, 3712.0],
    "0.32": [1152.0, 3584.0],
    "0.33": [1152.0, 3456.0],
    "0.35": [1152.0, 3328.0],
    "0.4": [1280.0, 3200.0],
    "0.42": [1280.0, 3072.0],
    "0.48": [1408.0, 2944.0],
    "0.5": [1408.0, 2816.0],
    "0.52": [1408.0, 2688.0],
    "0.57": [1536.0, 2688.0],
    "0.6": [1536.0, 2560.0],
    "0.68": [1664.0, 2432.0],
    "0.72": [1664.0, 2304.0],
    "0.78": [1792.0, 2304.0],
    "0.82": [1792.0, 2176.0],
    "0.88": [1920.0, 2176.0],
    "0.94": [1920.0, 2048.0],
    "1.0": [2048.0, 2048.0],
    "1.07": [2048.0, 1920.0],
    "1.13": [2176.0, 1920.0],
    "1.21": [2176.0, 1792.0],
    "1.29": [2304.0, 1792.0],
    "1.38": [2304.0, 1664.0],
    "1.46": [2432.0, 1664.0],
    "1.67": [2560.0, 1536.0],
    "1.75": [2688.0, 1536.0],
    "2.0": [2816.0, 1408.0],
    "2.09": [2944.0, 1408.0],
    "2.4": [3072.0, 1280.0],
    "2.5": [3200.0, 1280.0],
    "2.89": [3328.0, 1152.0],
    "3.0": [3456.0, 1152.0],
    "3.11": [3584.0, 1152.0],
    "3.62": [3712.0, 1024.0],
    "3.75": [3840.0, 1024.0],
    "3.88": [3968.0, 1024.0],
    "4.0": [4096.0, 1024.0],
}
ASPECT_RATIO_BINS = {
    256: ASPECT_RATIO_2048_BIN,
    128: ASPECT_RATIO_1024_BIN,
    64: ASPECT_RATIO_512_BIN,
    32: ASPECT_RATIO_256_BIN,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def classify_height_width_bin(height: int, width: int, sample_size: int) -> tuple[int, int]:
    try:
        ratios = ASPECT_RATIO_BINS[sample_size]
    except KeyError as error:
        raise ValueError("Invalid sample size") from error
    aspect_ratio = float(height / width)
    closest = min(ratios, key=lambda ratio: abs(float(ratio) - aspect_ratio))
    selected = ratios[closest]
    return int(selected[0]), int(selected[1])


def resize_and_crop_tensor(samples: Tensor, new_width: int, new_height: int) -> Tensor:
    original_height, original_width = samples.shape[2:]
    if original_height == new_height and original_width == new_width:
        return samples
    ratio = max(new_height / original_height, new_width / original_width)
    resized_width = int(original_width * ratio)
    resized_height = int(original_height * ratio)
    samples = F.interpolate(
        samples, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )
    start_x = (resized_width - new_width) // 2
    start_y = (resized_height - new_height) // 2
    return samples[:, :, start_y : start_y + new_height, start_x : start_x + new_width]


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "pixart-sigma-xl-1024"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class PixArtComponents:
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel
    transformer: PixArtTransformer2DModel
    vae: AutoencoderKL
    scheduler_config: dict[str, Any]
    variant: str
    scheduler: str

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> PixArtComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a PixArt snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            class_name = read_json(root / "model_index.json").get("_class_name")
            variants = {"PixArtSigmaPipeline": "sigma", "PixArtAlphaPipeline": "alpha"}
            if class_name not in variants:
                raise CheckpointError(f"Unsupported PixArt pipeline class {class_name!r}")
            scheduler_config = read_json(root / "scheduler" / "scheduler_config.json")
            schedulers = {
                "DPMSolverMultistepScheduler": "dpm-solver++",
                "LCMScheduler": "lcm",
            }
            scheduler_class = scheduler_config.get("_class_name")
            if scheduler_class not in schedulers:
                raise CheckpointError(f"Unsupported PixArt scheduler {scheduler_class!r}")
            return cls(
                tokenizer=T5Tokenizer(root / "tokenizer"),
                text_encoder=T5EncoderModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                transformer=PixArtTransformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=scheduler_config,
                variant=variants[class_name],
                scheduler=schedulers[scheduler_class],
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: str,
    max_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    input_ids, attention_mask = tokenizer([prompt.lower().strip()], max_length)
    encoded = text_encoder(input_ids.to(device), attention_mask.to(device))
    embeds = encoded[0] if isinstance(encoded, tuple) else encoded
    return embeds.to(dtype=dtype), attention_mask.to(device)


@torch.inference_mode()
def encode_prompt(
    components: PixArtComponents,
    prompt: str,
    negative_prompt: str | None,
    max_sequence_length: int,
    device: torch.device,
    do_classifier_free_guidance: bool,
) -> tuple[Tensor, Tensor]:
    dtype = components.transformer.pos_embed.proj.weight.dtype
    prompt_embeds, prompt_mask = _prompt_embeds(
        components.tokenizer,
        components.text_encoder,
        prompt,
        max_sequence_length,
        device,
        dtype,
    )
    if not do_classifier_free_guidance:
        return prompt_embeds, prompt_mask
    negative_embeds, negative_mask = _prompt_embeds(
        components.tokenizer,
        components.text_encoder,
        negative_prompt or "",
        max_sequence_length,
        device,
        dtype,
    )
    return torch.cat([negative_embeds, prompt_embeds]), torch.cat([negative_mask, prompt_mask])


def size_conditions(
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    batch_size: int,
    do_cfg: bool,
) -> tuple[Tensor, Tensor]:
    resolution = torch.tensor([height, width]).repeat(batch_size, 1)
    aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size, 1)
    resolution = resolution.to(dtype=dtype, device=device)
    aspect_ratio = aspect_ratio.to(dtype=dtype, device=device)
    if do_cfg:
        resolution = torch.cat([resolution, resolution])
        aspect_ratio = torch.cat([aspect_ratio, aspect_ratio])
    return resolution, aspect_ratio


@torch.inference_mode()
def denoise(
    components: PixArtComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    max_sequence_length: int,
    latents: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    use_cfg = guidance > 1.0
    prompt_embeds, prompt_mask = encode_prompt(
        components,
        prompt,
        negative_prompt,
        max_sequence_length,
        latents.device,
        use_cfg,
    )
    lcm_scheduler: LCMScheduler | None = None
    dpm_scheduler: DPMSolverMultistepScheduler | None = None
    if components.scheduler == "lcm":
        lcm_scheduler = LCMScheduler(components.scheduler_config)
        lcm_scheduler.set_timesteps(steps)
        timesteps = lcm_scheduler.timesteps
    else:
        dpm_scheduler = DPMSolverMultistepScheduler(components.scheduler_config)
        dpm_scheduler.set_timesteps(steps)
        timesteps = dpm_scheduler.timesteps
    added_cond_kwargs = None
    if components.variant == "alpha" and components.transformer.model_config["sample_size"] == 128:
        resolution, aspect_ratio = size_conditions(
            height,
            width,
            prompt_embeds.dtype,
            latents.device,
            batch_size=latents.shape[0],
            do_cfg=use_cfg,
        )
        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}
    latent_channels = int(components.transformer.model_config["in_channels"])
    for timestep in timesteps:
        latent_model_input = torch.cat([latents] * 2) if use_cfg else latents
        if lcm_scheduler is not None:
            latent_model_input = lcm_scheduler.scale_model_input(latent_model_input, timestep)
        else:
            assert dpm_scheduler is not None
            latent_model_input = dpm_scheduler.scale_model_input(latent_model_input, timestep)
        current_timestep = timestep[None].to(latent_model_input.device)
        current_timestep = current_timestep.expand(latent_model_input.shape[0])
        noise_pred = components.transformer(
            latent_model_input,
            prompt_embeds,
            current_timestep,
            prompt_mask,
            added_cond_kwargs,
        )
        if use_cfg:
            noise_unconditional, noise_text = noise_pred.chunk(2)
            noise_pred = noise_unconditional + guidance * (noise_text - noise_unconditional)
        if components.transformer.out_channels // 2 == latent_channels:
            noise_pred = noise_pred.chunk(2, dim=1)[0]
        if lcm_scheduler is not None:
            latents, _ = lcm_scheduler.step(noise_pred, timestep, latents, generator=generator)
        else:
            assert dpm_scheduler is not None
            latents = dpm_scheduler.step(noise_pred, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(
    vae: AutoencoderKL,
    latents: Tensor,
    *,
    output_width: int,
    output_height: int,
    resize: bool,
) -> Tensor:
    latents = latents / vae.scaling_factor
    image = vae.decode(latents.to(vae.decoder.conv_in.weight.dtype))
    if resize:
        image = resize_and_crop_tensor(image, output_width, output_height)
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class PixArtBackend(TextToImageBackend):
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
        self.components: PixArtComponents | None = None
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
            raise TypeError(f"Unknown PixArt options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if o["negative_prompt"] is not None and not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        if type(o["use_resolution_binning"]) is not bool:
            raise TypeError("use_resolution_binning must be a bool")
        for name in ("steps", "width", "height", "max_sequence_length"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        if (
            isinstance(o["guidance"], bool)
            or not isinstance(o["guidance"], int | float)
            or not math.isfinite(o["guidance"])
            or o["guidance"] < 0
        ):
            raise ValueError("guidance must be a finite number >= 0")
        seed = o["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        o["output_width"], o["output_height"] = o["width"], o["height"]
        try:
            transformer_config = read_json(self.source / "transformer" / "config.json")
        except FileNotFoundError:
            transformer_config = {"in_channels": 4, "sample_size": 128}
        if o["use_resolution_binning"]:
            o["height"], o["width"] = classify_height_width_bin(
                o["height"], o["width"], int(transformer_config.get("sample_size", 128))
            )
        for name in ("width", "height"):
            if o[name] % 8:
                raise ValueError(f"{name} must be a multiple of 8")
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

    def size_conditions(
        self, options: dict[str, Any], *, batch_size: int, do_cfg: bool
    ) -> tuple[Tensor, Tensor]:
        return size_conditions(
            options["height"],
            options["width"],
            self.dtype,
            self.device,
            batch_size=batch_size,
            do_cfg=do_cfg,
        )

    def _load(self) -> None:
        self.components = PixArtComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the PixArt architecture, so it was loaded",
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
            width=o["width"],
            height=o["height"],
            max_sequence_length=o["max_sequence_length"],
            latents=latents.to(self.dtype).to(self.device),
            generator=generator,
        )
        pixels = decode_latents(
            self.components.vae,
            final,
            output_width=o["output_width"],
            output_height=o["output_height"],
            resize=o["use_resolution_binning"],
        )
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": o["steps"],
            "guidance": float(o["guidance"]),
            "width": o["output_width"],
            "height": o["output_height"],
            "negative_prompt": o["negative_prompt"],
            "scheduler": self.components.scheduler,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
