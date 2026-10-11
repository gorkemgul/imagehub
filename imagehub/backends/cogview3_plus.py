"""CogView3-Plus text-to-image backend over a local diffusers-layout snapshot."""

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
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.cogview3_plus_transformer import CogView3PlusTransformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.ddim_cogvideox import CogVideoXDDIMScheduler
from imagehub.tokenization_t5 import T5Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 50,
    "guidance": 5.0,
    "width": 1024,
    "height": 1024,
    "seed": None,
    "negative_prompt": None,
    "max_sequence_length": 224,
    "original_size": None,
    "crops_coords_top_left": (0, 0),
    "latents": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "cogview3-plus-3b"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class CogView3PlusComponents:
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel
    transformer: CogView3PlusTransformer2DModel
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
    ) -> CogView3PlusComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a CogView3-Plus snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            class_name = read_json(root / "model_index.json").get("_class_name")
            if class_name != "CogView3PlusPipeline":
                raise CheckpointError(f"Unsupported CogView3-Plus pipeline class {class_name!r}")
            return cls(
                tokenizer=T5Tokenizer(root / "tokenizer"),
                text_encoder=T5EncoderModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                transformer=CogView3PlusTransformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _t5_embeds(
    components: CogView3PlusComponents,
    prompt: str,
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    input_ids, _ = components.tokenizer([prompt], max_sequence_length)
    # The reference pipeline passes no attention mask: padding tokens are attended to.
    embeds = components.text_encoder(input_ids.to(device))[0]
    return embeds.to(dtype=dtype, device=device)


@torch.inference_mode()
def encode_prompt(
    components: CogView3PlusComponents,
    prompt: str,
    negative_prompt: str | None,
    max_sequence_length: int,
    device: torch.device,
    do_classifier_free_guidance: bool,
) -> tuple[Tensor, Tensor | None]:
    dtype = cast(torch.dtype, components.text_encoder.shared.weight.dtype)
    prompt_embeds = _t5_embeds(components, prompt, max_sequence_length, device, dtype)
    negative_prompt_embeds = None
    if do_classifier_free_guidance:
        if negative_prompt is None:
            negative_prompt_embeds = prompt_embeds.new_zeros(prompt_embeds.shape)
        else:
            negative_prompt_embeds = _t5_embeds(
                components, negative_prompt, max_sequence_length, device, dtype
            )
    return prompt_embeds, negative_prompt_embeds


@torch.inference_mode()
def denoise(
    components: CogView3PlusComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    latents: Tensor,
    max_sequence_length: int = 224,
    original_size: tuple[int, int] | None = None,
    crops_coords_top_left: tuple[int, int] = (0, 0),
) -> Tensor:
    use_cfg = guidance > 1.0
    device = latents.device
    prompt_embeds, negative_prompt_embeds = encode_prompt(
        components, prompt, negative_prompt, max_sequence_length, device, use_cfg
    )
    if use_cfg:
        assert negative_prompt_embeds is not None
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    scheduler = CogVideoXDDIMScheduler(components.scheduler_config)
    scheduler.set_timesteps(steps)
    dtype = prompt_embeds.dtype
    original = torch.tensor([original_size or (height, width)], dtype=dtype)
    target = torch.tensor([(height, width)], dtype=dtype)
    crops = torch.tensor([crops_coords_top_left], dtype=dtype)
    if use_cfg:
        original = torch.cat([original, original])
        target = torch.cat([target, target])
        crops = torch.cat([crops, crops])
    original, target, crops = original.to(device), target.to(device), crops.to(device)
    for timestep in scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2) if use_cfg else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
        current_timestep = timestep.expand(latent_model_input.shape[0])
        noise_pred = components.transformer(
            latent_model_input,
            prompt_embeds,
            current_timestep,
            original,
            target,
            crops,
        )
        noise_pred = noise_pred.float()
        if use_cfg:
            noise_unconditional, noise_text = noise_pred.chunk(2)
            noise_pred = noise_unconditional + guidance * (noise_text - noise_unconditional)
        latents = scheduler.step(noise_pred, timestep, latents)
        latents = latents.to(dtype)
    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: Tensor) -> Tensor:
    image = vae.decode(latents / vae.scaling_factor)
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


def _int_pair(value: Any, name: str, *, minimum: int) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(item) is not int or item < minimum for item in value)
    ):
        bound = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a pair of {bound} ints")
    return (value[0], value[1])


class CogView3PlusBackend(TextToImageBackend):
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
        self.components: CogView3PlusComponents | None = None
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults

    def _geometry(self) -> tuple[int, int, int, int]:
        """Latent channels, VAE scale, patch size and the largest patch grid."""
        channels, vae_scale, patch_size, max_grid = 16, 8, 2, 128
        transformer_config = self.source / "transformer" / "config.json"
        vae_config = self.source / "vae" / "config.json"
        if transformer_config.is_file():
            config = read_json(transformer_config)
            channels = int(config.get("in_channels", channels))
            patch_size = int(config.get("patch_size", patch_size))
            max_grid = int(config.get("pos_embed_max_size", max_grid))
        if vae_config.is_file():
            blocks = read_json(vae_config).get("block_out_channels", [1, 1, 1, 1])
            vae_scale = 2 ** (len(blocks) - 1)
        return channels, vae_scale, patch_size, max_grid

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown CogView3-Plus options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if o["negative_prompt"] is not None and not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "width", "height", "max_sequence_length"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        channels, vae_scale, patch_size, max_grid = self._geometry()
        unit = vae_scale * patch_size
        for name in ("width", "height"):
            if o[name] % unit:
                raise ValueError(f"{name} must be a multiple of {unit}")
            if o[name] > unit * max_grid:
                raise ValueError(f"{name} must be at most {unit * max_grid}")
        guidance = o["guidance"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, int | float)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError("guidance must be a finite number >= 0")
        o["original_size"] = (
            None
            if o["original_size"] is None
            else _int_pair(o["original_size"], "original_size", minimum=1)
        )
        o["crops_coords_top_left"] = _int_pair(
            o["crops_coords_top_left"], "crops_coords_top_left", minimum=0
        )
        seed, latents = o["seed"], o["latents"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        shape = (1, channels, o["height"] // vae_scale, o["width"] // vae_scale)
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
        self.components = CogView3PlusComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the CogView3-Plus architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
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
            width=o["width"],
            height=o["height"],
            latents=latents.to(self.dtype).to(self.device),
            max_sequence_length=o["max_sequence_length"],
            original_size=o["original_size"],
            crops_coords_top_left=o["crops_coords_top_left"],
        )
        pixels = decode_latents(self.components.vae, final)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": o["steps"],
            "guidance": float(o["guidance"]),
            "negative_prompt": o["negative_prompt"],
            "width": o["width"],
            "height": o["height"],
            "scheduler": "ddim_cogvideox",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
