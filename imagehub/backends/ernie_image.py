"""ERNIE-Image text-to-image backend over a local diffusers-layout snapshot."""

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
from imagehub.backends.flux2 import decode_latents as decode_vae_latents
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl_flux2 import (
    AutoencoderKLFlux2,
    denormalize_latents,
    unpatchify_latents,
)
from imagehub.models.ernie_image_transformer import ErnieImageTransformer2DModel
from imagehub.models.mistral3_text import Mistral3TextModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_mistral import MistralTokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": "",
    "steps": 50,
    "guidance": 4.0,
    "width": 1024,
    "height": 1024,
    "num_images": 1,
    "seed": None,
    "latents": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_DEFAULT_VAE_SCALE = 16
# The reference pipeline normalises with this constant instead of the VAE's configured epsilon.
_BATCH_NORM_EPS = 1e-5


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "ernie-image"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class ErnieImageComponents:
    tokenizer: MistralTokenizer
    text_encoder: Mistral3TextModel
    transformer: ErnieImageTransformer2DModel
    vae: AutoencoderKLFlux2
    scheduler_config: dict[str, Any]

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> ErnieImageComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not an ERNIE-Image snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            return cls(
                tokenizer=MistralTokenizer(root / "tokenizer"),
                text_encoder=Mistral3TextModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                transformer=ErnieImageTransformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderKLFlux2.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


@torch.inference_mode()
def encode_prompt(components: ErnieImageComponents, prompt: str, device: torch.device) -> Tensor:
    """Return the second-to-last hidden state of the text encoder, shape ``(tokens, width)``."""
    ids = components.tokenizer(prompt)
    input_ids = torch.tensor([ids], device=device)
    _, hidden_states = components.text_encoder(input_ids, output_hidden_states=True)
    return hidden_states[-2][0]


def pad_text(
    hiddens: list[Tensor], device: torch.device, dtype: torch.dtype, width: int
) -> tuple[Tensor, Tensor]:
    """Stack variable-length text states into a zero-padded batch and their lengths."""
    lengths = torch.tensor([h.shape[0] for h in hiddens], device=device, dtype=torch.long)
    padded = torch.zeros((len(hiddens), int(lengths.max().item()), width), device=device, dtype=dtype)
    for index, hidden in enumerate(hiddens):
        padded[index, : hidden.shape[0], :] = hidden.to(device).to(dtype)
    return padded, lengths


@torch.inference_mode()
def denoise(
    components: ErnieImageComponents,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    latents: Tensor,
) -> Tensor:
    """Run the flow-match loop from initial noise; returns the normalised patchified latents."""
    device, dtype = latents.device, latents.dtype
    batch = latents.shape[0]
    use_cfg = guidance > 1.0
    width = components.transformer.config["text_in_dim"]
    positive = [encode_prompt(components, prompt, device)] * batch
    if use_cfg:
        negative = [encode_prompt(components, negative_prompt, device)] * batch
        text, lengths = pad_text(negative + positive, device, dtype, width)
    else:
        text, lengths = pad_text(positive, device, dtype, width)

    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    scheduler.set_timesteps(steps, sigmas=torch.linspace(1.0, 0.0, steps + 1)[:-1].tolist())
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents, latents], dim=0) if use_cfg else latents
        timesteps = torch.full((model_input.shape[0],), timestep.item(), device=device, dtype=dtype)
        prediction = components.transformer(model_input, timesteps, text, lengths)
        if use_cfg:
            uncond, cond = prediction.chunk(2, dim=0)
            prediction = uncond + guidance * (cond - uncond)
        latents = scheduler.step(prediction, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderKLFlux2, latents: Tensor) -> Tensor:
    """Decode final latents to uint8 images of shape (batch, height, width, 3) on CPU."""
    assert vae.bn.running_mean is not None and vae.bn.running_var is not None
    latents = denormalize_latents(
        latents, vae.bn.running_mean, vae.bn.running_var, _BATCH_NORM_EPS
    )
    return decode_vae_latents(vae, unpatchify_latents(latents))


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class ErnieImageBackend(TextToImageBackend):
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
        self.components: ErnieImageComponents | None = None
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults

    def _vae_scale(self) -> int:
        """Pixels per latent cell of the patchified latents: ``2 ** len(block_out_channels)``."""
        vae_config = self.source / "vae" / "config.json"
        if vae_config.is_file():
            return 2 ** len(read_json(vae_config).get("block_out_channels", [1] * 4))
        return _DEFAULT_VAE_SCALE

    def _latent_shape(self, options: dict[str, Any]) -> tuple[int, int, int, int]:
        channels = 128
        transformer_config = self.source / "transformer" / "config.json"
        if transformer_config.is_file():
            channels = int(read_json(transformer_config).get("in_channels", channels))
        scale = self._vae_scale()
        return (options["num_images"], channels, options["height"] // scale, options["width"] // scale)

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown ERNIE-Image options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if o["negative_prompt"] is None:
            o["negative_prompt"] = ""
        if not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "num_images", "width", "height"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        scale = self._vae_scale()
        for name in ("width", "height"):
            if o[name] % scale:
                raise ValueError(f"{name} must be a multiple of {scale}")
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
        shape = self._latent_shape(o)
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
        self.components = ErnieImageComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the ERNIE-Image architecture, so it was loaded",
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
                self._latent_shape(o),
                generator=torch.Generator("cpu").manual_seed(seed),
                dtype=self.dtype,
            )
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
            steps=o["steps"],
            guidance=float(o["guidance"]),
            latents=latents.to(device=self.device, dtype=self.dtype),
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
            "prompt_enhancer": False,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
