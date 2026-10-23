"""Lumina-DiMOO text-to-image backend with discrete image-token diffusion."""

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
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.llada import LLaDAForMultiModalGeneration
from imagehub.models.vq_model import VQModel
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

MASK_TOKEN_ID = 126336
NEWLINE_TOKEN_ID = 126084
IMAGE_TOKEN_OFFSET = 126356
ANSWER_START_ID = 126354
ANSWER_END_ID = 126355
BEGIN_IMAGE_ID = 126349
END_IMAGE_ID = 126350
CODEBOOK_SIZE = 8192
SYSTEM_PROMPT = "Generate an image according to the text prompt."
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 64,
    "guidance": 4.0,
    "temperature": 1.0,
    "width": 1024,
    "height": 1024,
    "seed": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def format_prompt(prompt: str) -> tuple[str, str]:
    prefix = f"<system>{SYSTEM_PROMPT}</system><user>"
    return prefix + prompt + "</user>", prefix + "<uncondition></user>"


def prompt_ids(
    tokenizer: Qwen2Tokenizer, prompt: str, *, height: int, width: int, device: torch.device
) -> tuple[Tensor, Tensor, int]:
    conditional, unconditional = format_prompt(prompt)
    conditional_ids = tokenizer.tokenize(conditional)
    unconditional_ids = tokenizer.tokenize(unconditional)
    grid_height, grid_width = height // 16, width // 16
    image: list[int] = []
    for _ in range(grid_height):
        image.extend([MASK_TOKEN_ID] * grid_width)
        image.append(NEWLINE_TOKEN_ID)
    suffix = [ANSWER_START_ID, BEGIN_IMAGE_ID, *image, END_IMAGE_ID, ANSWER_END_ID]
    code_start = len(conditional_ids) + 2
    return (
        torch.tensor([conditional_ids + suffix], dtype=torch.long, device=device),
        torch.tensor([unconditional_ids], dtype=torch.long, device=device),
        code_start,
    )


def _gumbel(values: Tensor, generator: torch.Generator) -> Tensor:
    uniform = torch.rand(values.shape, dtype=values.dtype, generator=generator, device="cpu")
    return -torch.log(-torch.log(uniform + 1e-20) + 1e-20).to(values.device)


def mask_by_random_topk(
    count: Tensor, probabilities: Tensor, temperature: float, generator: torch.Generator
) -> Tensor:
    confidence = torch.log(probabilities.clamp_min(1e-20)) + temperature * _gumbel(
        probabilities, generator
    )
    indices = count.long().unsqueeze(1).clamp(0, probabilities.shape[1] - 1)
    cutoff = torch.gather(torch.sort(confidence, dim=-1).values, 1, indices)
    return confidence < cutoff


@torch.inference_mode()
def generate_tokens(
    model: LLaDAForMultiModalGeneration,
    conditional_ids: Tensor,
    unconditional_ids: Tensor,
    *,
    code_start: int,
    image_tokens: int,
    steps: int,
    guidance: float,
    temperature: float,
    generator: torch.Generator,
) -> Tensor:
    sequence = conditional_ids.clone()
    mask = sequence == MASK_TOKEN_ID
    initial_unknown = mask.sum(dim=1, keepdim=True)
    unknown = initial_unknown.clone()
    for step in range(steps):
        if not bool(unknown.item()):
            break
        if step + 1 < steps:
            ratio = torch.tensor([(step + 1) / steps], device=sequence.device)
            keep = (initial_unknown.float() * torch.cos(0.5 * math.pi * ratio)).floor()
            keep = keep.clamp_min(1).long()
        else:
            keep = torch.zeros_like(unknown)
        conditional_logits = model(sequence)[
            ..., IMAGE_TOKEN_OFFSET : IMAGE_TOKEN_OFFSET + CODEBOOK_SIZE
        ]
        if guidance > 0:
            unconditioned = torch.cat((unconditional_ids, sequence[:, code_start - 2 :]), dim=1)
            unconditioned_mask = torch.cat(
                (
                    torch.zeros_like(unconditional_ids, dtype=torch.bool),
                    mask[:, code_start - 2 :],
                ),
                dim=1,
            )
            cond = conditional_logits[mask].view(1, -1, CODEBOOK_SIZE)
            uncond = model(unconditioned)[
                ..., IMAGE_TOKEN_OFFSET : IMAGE_TOKEN_OFFSET + CODEBOOK_SIZE
            ][unconditioned_mask].view(1, -1, CODEBOOK_SIZE)
            logits = (1 + guidance) * cond - guidance * uncond
        else:
            logits = conditional_logits[:, mask[0]]
        sampled = (
            logits.argmax(-1)
            if temperature == 0
            else (logits / temperature + _gumbel(logits, generator)).argmax(-1)
        )
        probabilities = logits.softmax(-1)
        confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        positions = mask.nonzero(as_tuple=False)[:, 1]
        sequence.view(-1)[positions] = (sampled + IMAGE_TOKEN_OFFSET).view(-1)
        selected = mask_by_random_topk(keep.squeeze(1), confidence, temperature, generator)
        sequence.view(-1)[positions[selected.view(-1)]] = MASK_TOKEN_ID
        mask = sequence == MASK_TOKEN_ID
        unknown = mask.sum(dim=1, keepdim=True)
    tokens = sequence[0, code_start:-2]
    return tokens[tokens != NEWLINE_TOKEN_ID].view(1, image_tokens) - IMAGE_TOKEN_OFFSET


def decode_tokens(vqvae: VQModel, tokens: Tensor, height: int, width: int) -> Tensor:
    latent_height, latent_width = height // 16, width // 16
    upcast = vqvae.post_quant_conv.weight.dtype == torch.float16 and vqvae.config["force_upcast"]
    if upcast:
        vqvae.float()
    try:
        decoded = vqvae.decode(
            tokens,
            force_not_quantize=True,
            shape=(tokens.shape[0], latent_height, latent_width, vqvae.config["latent_channels"]),
        ).clip(0, 1)
    finally:
        if upcast:
            vqvae.half()
    pixels = decoded.permute(0, 2, 3, 1).float().cpu().mul(255).round()
    return pixels.to(torch.uint8)


@dataclass
class LuminaDiMOOComponents:
    tokenizer: Qwen2Tokenizer
    model: LLaDAForMultiModalGeneration
    vqvae: VQModel

    @classmethod
    def load(
        cls, directory: str | os.PathLike[str], *, device: torch.device, dtype: torch.dtype
    ) -> LuminaDiMOOComponents:
        root = Path(directory)
        required = (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "vqvae/config.json",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Lumina-DiMOO snapshot (missing {', '.join(missing)})"
            )
        class_name = read_json(root / "model_index.json").get("_class_name")
        if class_name != "LuminaDiMOOPipeline":
            raise CheckpointError(f"Unsupported Lumina-DiMOO pipeline class {class_name!r}")
        return cls(
            Qwen2Tokenizer(root),
            LLaDAForMultiModalGeneration.from_pretrained(root, device=device, dtype=dtype),
            VQModel.from_pretrained(root / "vqvae", device=device, dtype=dtype),
        )


class LuminaDiMOOBackend(TextToImageBackend):
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
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults
        self.components: LuminaDiMOOComponents | None = None
        self.snapshot: dict[str, Any] | None = None

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown Lumina-DiMOO options: {', '.join(unknown)}")
        values = self.option_defaults | merged
        if type(values["steps"]) is not int or not 1 <= values["steps"] <= 1000:
            raise ValueError("steps must be an int in [1, 1000]")
        for name in ("guidance", "temperature"):
            value = values[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if values["guidance"] < 0 or values["temperature"] < 0:
            raise ValueError("guidance and temperature must be >= 0")
        for name in ("width", "height"):
            if type(values[name]) is not int or values[name] <= 0 or values[name] % 16:
                raise ValueError(f"{name} must be a positive multiple of 16")
        seed = values["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        return values

    def _load(self) -> None:
        self.components = LuminaDiMOOComponents.load(
            self.source, device=self.device, dtype=self.dtype
        )
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
        conditional, unconditional, code_start = prompt_ids(
            self.components.tokenizer,
            inputs.prompt,
            height=options["height"],
            width=options["width"],
            device=self.device,
        )
        image_tokens = options["height"] // 16 * (options["width"] // 16)
        tokens = generate_tokens(
            self.components.model,
            conditional,
            unconditional,
            code_start=code_start,
            image_tokens=image_tokens,
            steps=options["steps"],
            guidance=float(options["guidance"]),
            temperature=float(options["temperature"]),
            generator=generator,
        )
        pixels = decode_tokens(self.components.vqvae, tokens, options["height"], options["width"])
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": options["steps"],
            "guidance": float(options["guidance"]),
            "temperature": float(options["temperature"]),
            "width": options["width"],
            "height": options["height"],
            "scheduler": "cosine-maskgit",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
