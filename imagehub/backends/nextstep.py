"""NextStep-1 text-to-image backend."""

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
from imagehub.models.nextstep import NextStepAutoencoder, NextStepModel
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "width": 256,
    "height": 256,
    "guidance": 1.0,
    "negative_prompt": "",
    "steps": 20,
    "timestep_shift": 1.0,
    "sde": True,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _left_pad(rows: list[list[int]], pad: int, device: torch.device) -> tuple[Tensor, Tensor]:
    length = max(map(len, rows))
    ids = [[pad] * (length - len(row)) + row for row in rows]
    mask = [[False] * (length - len(row)) + [True] * len(row) for row in rows]
    return torch.tensor(ids, device=device), torch.tensor(mask, device=device)


class NextStepComponents:
    def __init__(
        self,
        tokenizer: Qwen2Tokenizer,
        model: NextStepModel,
        vae: NextStepAutoencoder,
    ) -> None:
        self.tokenizer, self.model, self.vae = tokenizer, model, vae

    @classmethod
    def load(cls, root: Path, *, device: torch.device, dtype: torch.dtype) -> NextStepComponents:
        config = read_json(root / "config.json")
        vae_config = read_json(root / "vae" / "config.json")
        with torch.device("meta"):
            model = NextStepModel(config)
            vae = NextStepAutoencoder(vae_config)
        model_state = load_weights(root)
        if config.get("torch_dtype") == "bfloat16":
            model_state = {
                name: value.to(torch.bfloat16) if value.is_floating_point() else value
                for name, value in model_state.items()
            }
        model.load_checkpoint(model_state)
        model.init_non_persistent_buffers()
        vae.load_checkpoint(load_weights(root / "vae" / "checkpoint.pt"))
        return cls(
            Qwen2Tokenizer(root),
            model.to(device=device, dtype=dtype).eval(),
            vae.to(device=device, dtype=dtype).eval(),
        )


@torch.inference_mode()
def generate(
    components: NextStepComponents,
    prompt: str,
    *,
    width: int,
    height: int,
    guidance: float,
    negative_prompt: str,
    steps: int,
    timestep_shift: float,
    sde: bool,
    generator: torch.Generator,
) -> Tensor:
    model, tokenizer = components.model, components.tokenizer
    config = model.config
    down_factor = 8 * config["latent_patch_size"]
    latent_height, latent_width = height // down_factor, width // down_factor
    area = f"<|image_area|>{latent_height}*{latent_width}"
    captions = [prompt] if guidance == 1 else [prompt, negative_prompt]
    rows = [
        [
            *([config["bos_token_id"]] if getattr(tokenizer, "add_bos_token", False) else []),
            *tokenizer.tokenize(text),
            *tokenizer.tokenize(area),
            config["boi"],
        ]
        for text in captions
    ]
    device = next(model.parameters()).device
    ids, mask = _left_pad(rows, tokenizer.pad_id, device)
    hidden, cache = model(model.embed(ids), mask)
    condition = hidden[:, -1:]
    tokens: list[Tensor] = []
    token_dim = config["latent_channels"] * config["latent_patch_size"] ** 2
    for _ in range(latent_height * latent_width):
        projected = model.image_out_projector(condition).squeeze(1)
        noise = torch.randn((1, token_dim), generator=generator, dtype=projected.dtype).to(device)
        token = model.image_head.sample(
            projected,
            noise,
            guidance=guidance,
            steps=steps,
            shift=timestep_shift,
            generator=generator,
            sde=sde,
        )
        tokens.append(token)
        embedded = model.image_in_projector(token[:, None])
        if guidance != 1:
            embedded = torch.cat((embedded, embedded))
        mask = torch.cat((mask, torch.ones(mask.shape[0], 1, device=device, dtype=torch.bool)), -1)
        hidden, cache = model(embedded, mask, cache)
        condition = hidden[:, -1:]
    packed = torch.stack(tokens, 1)
    latents = model.unpatchify(packed, latent_height, latent_width)
    image = components.vae.decode(latents).float()
    return (image / 2 + 0.5).clamp(0, 1).mul(255).round().to(torch.uint8)


class NextStepBackend(TextToImageBackend):
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
        self.dtype = dtype or torch.float32
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be torch.float32, torch.float16 or torch.bfloat16")
        self.components: NextStepComponents | None = None
        self.snapshot: dict[str, Any] | None = None

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown NextStep-1 options: {', '.join(unknown)}")
        result = DEFAULTS | merged
        if type(result["seed"]) is not int and result["seed"] is not None:
            raise ValueError("seed must be an int or None")
        if result["seed"] is not None and not 0 <= result["seed"] < 2**64:
            raise ValueError("seed must be in [0, 2**64)")
        for name in ("width", "height"):
            if type(result[name]) is not int or result[name] < 16 or result[name] % 16:
                raise ValueError(f"{name} must be a positive multiple of 16")
        if not isinstance(result["guidance"], int | float) or not math.isfinite(result["guidance"]):
            raise ValueError("guidance must be a finite number")
        if result["guidance"] < 1:
            raise ValueError("guidance must be at least 1")
        if not isinstance(result["negative_prompt"], str):
            raise ValueError("negative_prompt must be a string")
        if type(result["steps"]) is not int or result["steps"] < 1:
            raise ValueError("steps must be a positive int")
        if not isinstance(result["timestep_shift"], int | float) or result["timestep_shift"] <= 0:
            raise ValueError("timestep_shift must be positive")
        if type(result["sde"]) is not bool:
            raise ValueError("sde must be a bool")
        return result

    def _load(self) -> None:
        self.components = NextStepComponents.load(self.source, device=self.device, dtype=self.dtype)
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
            negative_prompt=options["negative_prompt"],
            steps=options["steps"],
            timestep_shift=float(options["timestep_shift"]),
            sde=options["sde"],
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id", self.model_id),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": options["steps"],
            "guidance": float(options["guidance"]),
            "width": options["width"],
            "height": options["height"],
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        images = pixels.permute(0, 2, 3, 1).cpu()
        return ImageResult(tuple(Image(image) for image in images), metadata)
