"""BAGEL text-to-image backend."""

from __future__ import annotations

import math
import os
import secrets
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.checkpoints import load_weights, read_json, read_snapshot
from imagehub.image import Image
from imagehub.models.bagel import BagelAutoencoder, BagelModel
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "width": 1024,
    "height": 1024,
    "guidance": 4.0,
    "steps": 50,
    "timestep_shift": 3.0,
    "cfg_interval": (0.4, 1.0),
    "cfg_renorm_min": 0.0,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_BOS, _EOS, _IMAGE_START, _IMAGE_END = 151644, 151645, 151652, 151653


class BagelComponents:
    def __init__(self, tokenizer: Qwen2Tokenizer, model: BagelModel, vae: BagelAutoencoder) -> None:
        self.tokenizer, self.model, self.vae = tokenizer, model, vae

    @classmethod
    def load(cls, root: Path, *, device: torch.device, dtype: torch.dtype) -> BagelComponents:
        config = dict(read_json(root / "config.json"))
        config["max_latent_size"] = 64
        with torch.device("meta"):
            model = BagelModel(config)
            vae = BagelAutoencoder()
        model.load_checkpoint(load_weights(root / "ema.safetensors"))
        model.init_non_persistent_buffers()
        vae.load_checkpoint(load_weights(root / "ae.safetensors"))
        return cls(
            Qwen2Tokenizer(root),
            model.to(device=device, dtype=dtype).eval(),
            vae.to(device=device, dtype=dtype).eval(),
        )


def _prompt_cache(model: BagelModel, tokenizer: Qwen2Tokenizer, prompt: str) -> list[Any]:
    ids = torch.tensor(
        [_BOS, *tokenizer.tokenize(prompt), _EOS], device=model.vae2llm.weight.device
    )
    value = model.language_model.model.embed_tokens(ids)
    positions = torch.arange(ids.numel(), device=ids.device)
    generated = torch.zeros(ids.numel(), dtype=torch.bool, device=ids.device)
    _, cache = model.language_model.model(value, positions, generated, causal=True)
    return cache


def _flow_prediction(
    model: BagelModel,
    latent: Tensor,
    timestep: Tensor,
    positions: Tensor,
    latent_positions: Tensor,
    cache: list[Any] | None,
) -> Tensor:
    device = latent.device
    ids = torch.tensor([_IMAGE_START, _IMAGE_END], device=device)
    text = model.language_model.model.embed_tokens(ids)
    value = text.new_zeros((latent.shape[0] + 2, text.shape[-1]))
    value[0], value[-1] = text[0], text[1]
    value[1:-1] = (
        model.vae2llm(latent.to(model.vae2llm.weight.dtype))
        + model.time_embedder(timestep.expand(latent.shape[0]))
        + model.latent_pos_embed(latent_positions)
    )
    generated = torch.zeros(value.shape[0], dtype=torch.bool, device=device)
    generated[1:-1] = True
    output, _ = model.language_model.model(value, positions, generated, cache, causal=False)
    return model.llm2vae(output[1:-1])


def _to_pixels(image: Tensor) -> Tensor:
    return (image * 0.5 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)


@torch.inference_mode()
def generate(
    components: BagelComponents,
    prompt: str,
    *,
    width: int,
    height: int,
    guidance: float,
    steps: int,
    timestep_shift: float,
    cfg_interval: tuple[float, float],
    cfg_renorm_min: float,
    generator: torch.Generator,
) -> Tensor:
    model = components.model
    device = model.vae2llm.weight.device
    cache = _prompt_cache(model, components.tokenizer, prompt)
    latent_height, latent_width = height // 16, width // 16
    token_dim = model.vae2llm.in_features
    latent = torch.randn(
        latent_height * latent_width,
        token_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    grid = (
        torch.arange(latent_height, device=device)[:, None] * 64
        + torch.arange(latent_width, device=device)[None]
    ).flatten()
    prompt_length = cache[0][0].shape[0]
    positions = torch.full((latent.shape[0] + 2,), prompt_length, device=device, dtype=torch.long)
    unconditioned_positions = torch.zeros_like(positions)
    times = torch.linspace(1, 0, steps, device=device)
    times = timestep_shift * times / (1 + (timestep_shift - 1) * times)
    for current, following in zip(times[:-1], times[1:], strict=True):
        packed_positions = positions.clone()
        packed_positions[1:-1] = prompt_length
        velocity = _flow_prediction(model, latent, current, packed_positions, grid, cache)
        if guidance > 1 and current > cfg_interval[0] and current <= cfg_interval[1]:
            unconditional = _flow_prediction(
                model, latent, current, unconditioned_positions, grid, None
            )
            guided = unconditional + guidance * (velocity - unconditional)
            scale = (velocity.norm() / (guided.norm() + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
            velocity = guided * scale
        latent = latent - velocity.to(latent) * (current - following)
    patch = 2
    latent = latent.reshape(1, latent_height, latent_width, patch, patch, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent).reshape(
        1, 16, latent_height * patch, latent_width * patch
    )
    vae_dtype = next(components.vae.parameters()).dtype
    image = components.vae.decode(latent.to(vae_dtype))
    return _to_pixels(image)


class BagelBackend(TextToImageBackend):
    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        model_id: str | None = None,
    ) -> None:
        super().__init__()
        self.source, self.model_id = Path(model), model_id
        self.device = resolve_device(device)
        self.dtype = dtype or torch.bfloat16
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be torch.float32, torch.float16 or torch.bfloat16")
        self.components: BagelComponents | None = None
        self.snapshot: dict[str, Any] | None = None

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown BAGEL options: {', '.join(unknown)}")
        result = DEFAULTS | merged
        if type(result["seed"]) is not int and result["seed"] is not None:
            raise ValueError("seed must be an int or None")
        if result["seed"] is not None and not 0 <= result["seed"] < 2**64:
            raise ValueError("seed must be in [0, 2**64)")
        for name in ("width", "height"):
            if type(result[name]) is not int or not 16 <= result[name] <= 1024 or result[name] % 16:
                raise ValueError(f"{name} must be a multiple of 16 from 16 through 1024")
        if not isinstance(result["guidance"], int | float) or not math.isfinite(result["guidance"]):
            raise ValueError("guidance must be a finite number")
        if result["guidance"] < 1:
            raise ValueError("guidance must be at least 1")
        if type(result["steps"]) is not int or result["steps"] < 2:
            raise ValueError("steps must be an int of at least 2")
        if not isinstance(result["timestep_shift"], int | float) or result["timestep_shift"] <= 0:
            raise ValueError("timestep_shift must be positive")
        interval = result["cfg_interval"]
        if (
            not isinstance(interval, tuple | list)
            or len(interval) != 2
            or not all(isinstance(value, int | float) for value in interval)
            or not 0 <= interval[0] <= interval[1] <= 1
        ):
            raise ValueError("cfg_interval must contain two ordered numbers in [0, 1]")
        if (
            not isinstance(result["cfg_renorm_min"], int | float)
            or not 0 <= result["cfg_renorm_min"] <= 1
        ):
            raise ValueError("cfg_renorm_min must be in [0, 1]")
        result["cfg_interval"] = (float(interval[0]), float(interval[1]))
        return result

    def _load(self) -> None:
        self.components = BagelComponents.load(self.source, device=self.device, dtype=self.dtype)
        self.snapshot = read_snapshot(self.source)

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed = options["seed"] if options["seed"] is not None else secrets.randbits(63)
        pixels = generate(
            self.components,
            inputs.prompt,
            width=options["width"],
            height=options["height"],
            guidance=float(options["guidance"]),
            steps=options["steps"],
            timestep_shift=float(options["timestep_shift"]),
            cfg_interval=options["cfg_interval"],
            cfg_renorm_min=float(options["cfg_renorm_min"]),
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id", self.model_id),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": options["steps"],
            "guidance": float(options["guidance"]),
            "timestep_shift": float(options["timestep_shift"]),
            "width": options["width"],
            "height": options["height"],
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        images = pixels.permute(0, 2, 3, 1).cpu()
        return ImageResult(tuple(Image(image) for image in images), metadata)
