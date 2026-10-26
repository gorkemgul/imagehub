"""Infinity-2B text-to-image backend."""

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
from imagehub.checkpoints import load_weights, read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.infinity import InfinityTransformer, ScaleSchedule, TextCondition
from imagehub.models.infinity_vae import InfinityVAE
from imagehub.models.t5 import T5EncoderModel
from imagehub.tokenization_t5 import T5Tokenizer
from imagehub.types import ImageResult, T2IRequest

TRANSFORMER_FILE = "infinity_2b_reg.pth"
VAE_FILE = "infinity_vae_d32reg.pth"
TEXT_ENCODER_DIR = "text_encoder"
SQUARE_SCALES = (
    (1, 1),
    (2, 2),
    (4, 4),
    (6, 6),
    (8, 8),
    (12, 12),
    (16, 16),
    (20, 20),
    (24, 24),
    (32, 32),
    (40, 40),
    (48, 48),
    (64, 64),
)
SCHEDULE_LENGTHS = {256: 7, 512: 10, 1024: 13}
OPTION_DEFAULTS: dict[str, Any] = {
    "cfg_scale": 3.0,
    "temperature": 0.5,
    "top_k": 900,
    "top_p": 0.97,
    "width": 1024,
    "height": 1024,
    "seed": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def scale_schedule(size: int) -> tuple[tuple[int, int, int], ...]:
    return tuple((1, height, width) for height, width in SQUARE_SCALES[: SCHEDULE_LENGTHS[size]])


def _load_native(
    model: InfinityTransformer | InfinityVAE,
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    container_key: str | None = None,
) -> None:
    model.load_checkpoint(load_weights(path, container_key=container_key))
    model.init_non_persistent_buffers()
    model.to(device=device, dtype=dtype).eval()


@dataclass
class InfinityComponents:
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel
    transformer: InfinityTransformer
    vae: InfinityVAE

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> InfinityComponents:
        root = Path(directory)
        required = (
            TRANSFORMER_FILE,
            VAE_FILE,
            f"{TEXT_ENCODER_DIR}/config.json",
            f"{TEXT_ENCODER_DIR}/tokenizer.json",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise CheckpointError(
                f"{root} is not an Infinity-2B snapshot (missing {', '.join(missing)}). "
                f"Download it with: python -m imagehub.pretrained infinity-2b {root}"
            )
        tokenizer = T5Tokenizer(root / TEXT_ENCODER_DIR)
        text_encoder = T5EncoderModel.from_pretrained(
            root / TEXT_ENCODER_DIR, device=device, dtype=dtype
        )
        overrides = (
            read_json(root / "infinity_config.json")
            if (root / "infinity_config.json").is_file()
            else {}
        )
        with torch.device("meta"):
            transformer = InfinityTransformer(overrides.get("transformer"))
            vae = InfinityVAE(overrides.get("vae"))
        _load_native(transformer, root / TRANSFORMER_FILE, device=device, dtype=dtype)
        _load_native(vae, root / VAE_FILE, device=device, dtype=dtype, container_key="vae")
        return cls(tokenizer, text_encoder, transformer, vae)


@torch.inference_mode()
def encode_prompt(
    components: InfinityComponents, prompt: str, device: torch.device
) -> TextCondition:
    input_ids, mask = components.tokenizer([prompt], 512)
    input_ids = input_ids.to(device)
    mask = mask.to(device)
    features = components.text_encoder(input_ids, mask)[0].float()
    lengths = mask.sum(-1).tolist()
    compact = torch.cat([row[: int(length)] for row, length in zip(features, lengths, strict=True)])
    cumulative = F.pad(mask.sum(-1, dtype=torch.int32).cumsum(0), (1, 0))
    return compact, [int(length) for length in lengths], cumulative, max(lengths)


def sample_bits(
    logits: Tensor,
    *,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
) -> Tensor:
    batch, length, choices = logits.shape
    if top_k > 0:
        kept = min(top_k, choices)
        threshold = logits.topk(kept, sorted=False, dim=-1).values.amin(-1, keepdim=True)
        logits.masked_fill_(logits < threshold, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=False)
        remove = sorted_logits.softmax(-1).cumsum(-1) <= 1 - top_p
        remove[..., -1:] = False
        logits.masked_fill_(remove.scatter(-1, sorted_indices, remove), -torch.inf)
    sampled = torch.multinomial(
        logits.softmax(-1).view(-1, choices).cpu(),
        1,
        replacement=True,
        generator=generator,
    )
    return sampled.view(batch, length).to(logits.device)


@torch.inference_mode()
def generate(
    components: InfinityComponents,
    condition: TextCondition,
    schedule: ScaleSchedule,
    *,
    seed: int,
    cfg_scale: float,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Tensor:
    transformer, vae = components.transformer, components.vae
    guidance = cfg_scale != 1.0
    stage, adaptive, pooled, projected = transformer.begin_generation(condition, schedule, guidance)
    batch = 1
    generator = torch.Generator("cpu").manual_seed(seed)
    final_size = schedule[-1]
    accumulated: Tensor | int = 0
    try:
        for index, (_, height, width) in enumerate(schedule):
            stage, logits = transformer.stage_logits(
                stage, adaptive, pooled, projected, schedule, index
            )
            logits = logits / temperature
            if guidance:
                logits = cfg_scale * logits[:batch] + (1.0 - cfg_scale) * logits[batch:]
            logits = logits.reshape(batch, height * width * vae.config["latent_channels"], 2)
            bits = sample_bits(logits, top_k=top_k, top_p=top_p, generator=generator).reshape(
                batch, 1, height, width, vae.config["latent_channels"]
            )
            codes = vae.bits_to_codes(bits).movedim(2, 2)
            if index + 1 < len(schedule):
                accumulated = accumulated + F.interpolate(
                    codes, size=final_size, mode="trilinear", align_corners=False
                )
                next_codes = F.interpolate(
                    accumulated,
                    size=schedule[index + 1],
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(2)
                next_codes = next_codes.flatten(2).transpose(1, 2)
                stage = transformer.next_stage(next_codes, 2 if guidance else 1)
            else:
                accumulated = accumulated + codes
    finally:
        transformer.end_generation()
    assert isinstance(accumulated, Tensor)
    image = vae.decode(accumulated.squeeze(2))
    # The reference converts RGB decoder output to BGR for OpenCV; reverse it back for Image.
    return ((image + 1) / 2).permute(0, 2, 3, 1).mul(255).to(torch.uint8)


def _finite(value: Any, name: str, *, positive: bool = False) -> None:
    valid = not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)
    if not valid or value < 0 or (positive and value == 0):
        relation = ">" if positive else ">="
        raise ValueError(f"{name} must be a finite number {relation} 0")


class InfinityBackend(TextToImageBackend):
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
        self.components: InfinityComponents | None = None
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
            raise TypeError(f"Unknown Infinity options: {', '.join(unknown)}")
        values = self.option_defaults | merged
        _finite(values["cfg_scale"], "cfg_scale")
        _finite(values["temperature"], "temperature", positive=True)
        if type(values["top_k"]) is not int or values["top_k"] < 0:
            raise ValueError("top_k must be an int >= 0")
        _finite(values["top_p"], "top_p", positive=True)
        if values["top_p"] > 1:
            raise ValueError("top_p must be at most 1")
        if values["width"] != values["height"] or values["width"] not in SCHEDULE_LENGTHS:
            raise ValueError("width and height must be equal and one of 256, 512 or 1024")
        seed = values["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("seed must be an int in [0, 2**63)")
        return values

    def _load(self) -> None:
        self.components = InfinityComponents.load(self.source, device=self.device, dtype=self.dtype)
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r}, not {self.model_id!r}",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    @torch.inference_mode()
    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed = options["seed"]
        if seed is None:
            seed = secrets.randbits(63)
        condition = encode_prompt(self.components, inputs.prompt, self.device)
        pixels = generate(
            self.components,
            condition,
            scale_schedule(options["width"]),
            seed=seed,
            cfg_scale=float(options["cfg_scale"]),
            temperature=float(options["temperature"]),
            top_k=options["top_k"],
            top_p=float(options["top_p"]),
        )
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "cfg_scale": float(options["cfg_scale"]),
            "temperature": float(options["temperature"]),
            "top_k": options["top_k"],
            "top_p": float(options["top_p"]),
            "width": options["width"],
            "height": options["height"],
            "scales": len(scale_schedule(options["width"])),
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
