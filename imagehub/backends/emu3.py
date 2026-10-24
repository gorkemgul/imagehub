"""Emu3-Gen text-to-image backend."""

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
from imagehub.checkpoints import load_weights, read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.emu3 import Emu3ForCausalLM, KVCache
from imagehub.models.emu3_vq import Emu3VisionVQ
from imagehub.tokenization_emu3 import Emu3Tokenizer
from imagehub.types import ImageResult, T2IRequest

POSITIVE_SUFFIX = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
    "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, "
    "username, blurry."
)
OPTION_DEFAULTS: dict[str, Any] = {
    "guidance": 3.0,
    "temperature": 1.0,
    "top_k": 2048,
    "width": 720,
    "height": 720,
    "seed": None,
    "negative_prompt": NEGATIVE_PROMPT,
    "positive_suffix": POSITIVE_SUFFIX,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def prompt_ids(tokenizer: Emu3Tokenizer, prompt: str, height: int, width: int) -> Tensor:
    text = f"<|extra_203|>{prompt}<|image start|>{height}*{width}<|image token|>"
    return torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0)


def _sample(logits: Tensor, top_k: int, temperature: float, generator: torch.Generator) -> Tensor:
    logits = logits.float() / temperature
    if top_k:
        cutoff = torch.topk(logits, min(top_k, logits.shape[-1])).values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, -torch.inf)
    probabilities = logits.softmax(-1).cpu()
    return torch.multinomial(probabilities, 1, generator=generator).to(logits.device)


@torch.inference_mode()
def generate_tokens(
    model: Emu3ForCausalLM,
    tokenizer: Emu3Tokenizer,
    conditional_ids: Tensor,
    unconditional_ids: Tensor | None,
    *,
    height: int,
    width: int,
    guidance: float,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
) -> Tensor:
    """Generate a constrained ``(1, height, width)`` visual-token grid."""
    conditional_logits, conditional_cache = model(conditional_ids)
    unconditional_cache: KVCache | None = None
    unconditional_logits: Tensor | None = None
    if unconditional_ids is not None:
        unconditional_logits, unconditional_cache = model(unconditional_ids)
    visual: list[Tensor] = []
    offset = 1
    while True:
        if offset == (width + 1) * height + 3:
            break
        if offset % (width + 1) == 0:
            next_token = torch.tensor([[tokenizer.eol_id]], device=conditional_ids.device)
        elif offset == (width + 1) * height + 1:
            next_token = torch.tensor([[tokenizer.eof_id]], device=conditional_ids.device)
        elif offset == (width + 1) * height + 2:
            next_token = torch.tensor([[tokenizer.eoi_id]], device=conditional_ids.device)
        else:
            logits = conditional_logits[:, -1]
            if unconditional_logits is not None:
                logits = unconditional_logits[:, -1] + guidance * (
                    logits - unconditional_logits[:, -1]
                )
            logits = logits[:, tokenizer.visual_start : tokenizer.visual_end + 1]
            sampled = _sample(logits, top_k, temperature, generator)
            visual.append(sampled)
            next_token = sampled + tokenizer.visual_start
        conditional_logits, conditional_cache = model(next_token, conditional_cache)
        if unconditional_cache is not None:
            unconditional_logits, unconditional_cache = model(next_token, unconditional_cache)
        offset += 1
    return torch.cat(visual, dim=1).reshape(1, height, width)


@dataclass
class Emu3Components:
    tokenizer: Emu3Tokenizer
    model: Emu3ForCausalLM
    vision_tokenizer: Emu3VisionVQ

    @classmethod
    def load(
        cls, directory: str | os.PathLike[str], *, device: torch.device, dtype: torch.dtype
    ) -> Emu3Components:
        root = Path(directory)
        required = (
            "config.json",
            "model.safetensors.index.json",
            "emu3.tiktoken",
            "emu3_vision_tokens.txt",
            "vision_tokenizer/config.json",
            "vision_tokenizer/model.safetensors",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise CheckpointError(
                f"{root} is not an Emu3-Gen snapshot (missing {', '.join(missing)})"
            )
        tokenizer = Emu3Tokenizer(root)
        with torch.device("meta"):
            model = Emu3ForCausalLM(read_json(root / "config.json"))
            vision = Emu3VisionVQ(read_json(root / "vision_tokenizer/config.json"))
        model.load_checkpoint(load_weights(root))
        model.to(device=device, dtype=dtype).eval()
        vision.load_checkpoint(load_weights(root / "vision_tokenizer"))
        vision.to(device=device, dtype=dtype).eval()
        return cls(tokenizer, model, vision)


class Emu3Backend(TextToImageBackend):
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
        self.components: Emu3Components | None = None
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
            raise TypeError(f"Unknown Emu3 options: {', '.join(unknown)}")
        values = self.option_defaults | merged
        for name in ("width", "height"):
            if type(values[name]) is not int or values[name] <= 0 or values[name] % 8:
                raise ValueError(f"{name} must be a positive multiple of 8")
        if values["width"] * values["height"] > 518400:
            raise ValueError("width * height must not exceed the published 518400-pixel area")
        for name in ("guidance", "temperature"):
            if isinstance(values[name], bool) or not isinstance(values[name], int | float):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(values[name]):
                raise ValueError(f"{name} must be a finite number")
        if values["guidance"] < 0 or values["temperature"] <= 0:
            raise ValueError("guidance must be >= 0 and temperature must be > 0")
        if type(values["top_k"]) is not int or not 0 <= values["top_k"] <= 32768:
            raise ValueError("top_k must be an int in [0, 32768]")
        if not isinstance(values["negative_prompt"], str) or not isinstance(
            values["positive_suffix"], str
        ):
            raise ValueError("negative_prompt and positive_suffix must be strings")
        seed = values["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        return values

    def _load(self) -> None:
        self.components = Emu3Components.load(self.source, device=self.device, dtype=self.dtype)
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    @torch.inference_mode()
    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed = options["seed"] if options["seed"] is not None else secrets.randbits(63)
        generator = torch.Generator("cpu").manual_seed(seed)
        grid_height, grid_width = options["height"] // 8, options["width"] // 8
        conditional = prompt_ids(
            self.components.tokenizer,
            inputs.prompt + options["positive_suffix"],
            grid_height,
            grid_width,
        ).to(self.device)
        unconditional = None
        if options["guidance"] != 1:
            unconditional = prompt_ids(
                self.components.tokenizer,
                options["negative_prompt"],
                grid_height,
                grid_width,
            ).to(self.device)
        tokens = generate_tokens(
            self.components.model,
            self.components.tokenizer,
            conditional,
            unconditional,
            height=grid_height,
            width=grid_width,
            guidance=float(options["guidance"]),
            temperature=float(options["temperature"]),
            top_k=options["top_k"],
            generator=generator,
        )
        decoded = self.components.vision_tokenizer.decode(tokens)
        pixels = decoded.float().add(1).div(2).clamp(0, 1).mul(255).to(torch.uint8)
        pixels = pixels.permute(0, 2, 3, 1)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "guidance": float(options["guidance"]),
            "temperature": float(options["temperature"]),
            "top_k": options["top_k"],
            "width": options["width"],
            "height": options["height"],
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
