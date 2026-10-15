"""HiDream-O1 direct-pixel text-to-image backend."""

from __future__ import annotations

import math
import os
import secrets
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.hidream_o1 import HiDreamO1Model
from imagehub.schedulers.flow_unipc import FlowUniPCMultistepScheduler
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

PATCH_SIZE = 32
NOISE_SCALE = 8.0
T_EPS = 0.001
DEFAULT_TIMESTEPS = (
    999,
    987,
    974,
    960,
    945,
    929,
    913,
    895,
    877,
    857,
    836,
    814,
    790,
    764,
    737,
    707,
    675,
    640,
    602,
    560,
    515,
    464,
    409,
    347,
    278,
    199,
    110,
    8,
)
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 50,
    "guidance": 5.0,
    "shift": 3.0,
    "width": 2560,
    "height": 1440,
    "num_images": 1,
    "seed": None,
    "noise": None,
    "timesteps": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def get_rope_index_fix_point(
    input_ids: Tensor,
    image_grid_thw: Tensor,
    *,
    image_token_id: int,
    vision_start_token_id: int,
    fix_point: int = 4096,
) -> Tensor:
    """Build the fixed-point multimodal positions used for target pixel patches."""
    positions = torch.ones(
        3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
    )
    for batch, row in enumerate(input_ids):
        starts = torch.argwhere(row == vision_start_token_id).flatten()
        image_starts = [int(index) + 1 for index in starts if row[int(index) + 1] == image_token_id]
        pieces: list[Tensor] = []
        cursor = 0
        offset = 0
        for image_index, end in enumerate(image_starts):
            text_length = max(0, end - cursor - 1)
            if text_length:
                text = torch.arange(text_length, device=row.device) + offset
                pieces.append(text.view(1, -1).expand(3, -1))
                offset += text_length
            time, height, width = (int(value) for value in image_grid_thw[image_index])
            t = (
                torch.arange(time, device=row.device)
                .view(-1, 1)
                .expand(-1, height * width)
                .flatten()
            )
            h = (
                torch.arange(height, device=row.device)
                .view(1, -1, 1)
                .expand(time, -1, width)
                .flatten()
            )
            w = (
                torch.arange(width, device=row.device)
                .view(1, 1, -1)
                .expand(time, height, -1)
                .flatten()
            )
            pieces.append(torch.stack((t, h, w)) + fix_point)
            fix_point = 0
            cursor = end + time * height * width
            offset = int(pieces[-1].max()) + 1
        if cursor < len(row):
            text = torch.arange(len(row) - cursor, device=row.device) + offset
            pieces.append(text.view(1, -1).expand(3, -1))
        positions[:, batch] = torch.cat(pieces, dim=1)
    return positions


def build_text_sample(
    prompt: str,
    height: int,
    width: int,
    tokenizer: Qwen2Tokenizer,
    config: dict[str, Any],
) -> dict[str, Tensor]:
    """Tokenize one prompt and append the target patch sequence."""
    rendered = tokenizer.chat_prompt(prompt) + "<|boi_token|><|tms_token|>"
    input_ids = torch.tensor([tokenizer.tokenize(rendered)], dtype=torch.long)
    image_length = (height // PATCH_SIZE) * (width // PATCH_SIZE)
    vision_tokens = torch.full((1, image_length), int(config["image_token_id"]), dtype=torch.long)
    vision_tokens[0, 0] = int(config["vision_start_token_id"])
    padded = torch.cat((input_ids, vision_tokens), dim=-1)
    grid = torch.tensor([[1, height // PATCH_SIZE, width // PATCH_SIZE]], dtype=torch.long)
    position_ids = get_rope_index_fix_point(
        padded,
        grid,
        image_token_id=int(config["image_token_id"]),
        vision_start_token_id=int(config["vision_start_token_id"]),
    )
    token_types = torch.zeros_like(padded)
    begin = input_ids.shape[1] - 1
    token_types[:, begin : begin + image_length + 1] = 1
    vinput_mask = torch.zeros_like(padded, dtype=torch.bool)
    vinput_mask[:, input_ids.shape[1] :] = True
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_types": token_types,
        "vinput_mask": vinput_mask,
    }


@dataclass
class HiDreamO1Components:
    tokenizer: Qwen2Tokenizer
    model: HiDreamO1Model

    @classmethod
    def load(
        cls, directory: Path, *, device: torch.device, dtype: torch.dtype, model_id: str | None
    ) -> HiDreamO1Components:
        required = ("config.json", "tokenizer.json", "tokenizer_config.json")
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            selected = model_id or "hidream-o1-image"
            raise CheckpointError(
                f"{directory} is not a HiDream-O1 snapshot (missing {', '.join(missing)}). "
                f"Download it with: python -m imagehub.pretrained {selected} {directory}"
            )
        return cls(
            tokenizer=Qwen2Tokenizer(directory),
            model=HiDreamO1Model.from_pretrained(directory, device=device, dtype=dtype),
        )


@torch.inference_mode()
def denoise(
    components: HiDreamO1Components,
    *,
    prompt: str,
    height: int,
    width: int,
    steps: int,
    guidance: float,
    shift: float,
    noise: Tensor,
    timesteps: list[int] | None,
) -> Tensor:
    model = components.model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    conditional = build_text_sample(prompt, height, width, components.tokenizer, model.config)
    samples = [conditional]
    if guidance > 1:
        samples.append(build_text_sample(" ", height, width, components.tokenizer, model.config))
    samples = [{name: value.to(device) for name, value in sample.items()} for sample in samples]
    patches = (
        noise.to(device=device, dtype=dtype)
        .unfold(2, PATCH_SIZE, PATCH_SIZE)
        .unfold(3, PATCH_SIZE, PATCH_SIZE)
    )
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(noise.shape[0], -1, 3 * PATCH_SIZE**2)
    scheduler = FlowUniPCMultistepScheduler(shift=shift)
    scheduler.set_timesteps(steps, device=device, timesteps=timesteps)
    for timestep in scheduler.timesteps:
        model_timestep = 1.0 - timestep.float() / 1000.0
        sigma = (timestep.float() / 1000.0).clamp_min(T_EPS)
        predictions = []
        for sample in samples:
            batch = patches.shape[0]
            expanded = {
                name: value.expand(-1, batch, -1)
                if name == "position_ids"
                else value.expand(batch, -1)
                for name, value in sample.items()
            }
            autocast = (
                torch.autocast(device.type, dtype=dtype, cache_enabled=False)
                if dtype != torch.float32 and device.type in ("cpu", "cuda")
                else nullcontext()
            )
            with autocast:
                output = model(
                    expanded["input_ids"],
                    expanded["position_ids"],
                    patches,
                    model_timestep.expand(batch),
                    expanded["token_types"],
                )
            mask = expanded["vinput_mask"]
            predictions.append(output[mask].reshape(batch, -1, output.shape[-1]))
        conditional_velocity = (predictions[0].float() - patches.float()) / sigma
        if len(predictions) == 2:
            unconditional_velocity = (predictions[1].float() - patches.float()) / sigma
            velocity = unconditional_velocity + guidance * (
                conditional_velocity - unconditional_velocity
            )
        else:
            velocity = conditional_velocity
        patches = scheduler.step(-velocity, timestep.float(), patches.float()).to(dtype)
    return patches


def decode_patches(patches: Tensor, height: int, width: int) -> Tensor:
    """Rearrange clean RGB patches and apply the official uint8 conversion."""
    batch = patches.shape[0]
    image = patches.reshape(
        batch, height // PATCH_SIZE, width // PATCH_SIZE, 3, PATCH_SIZE, PATCH_SIZE
    )
    image = image.permute(0, 1, 4, 2, 5, 3).reshape(batch, height, width, 3)
    image = (image + 1) / 2
    return (image.float().cpu().clamp(0, 1) * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class HiDreamO1Backend(TextToImageBackend):
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
        self.components: HiDreamO1Components | None = None
        self.snapshot: dict[str, Any] | None = None

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown HiDream-O1 options: {', '.join(unknown)}")
        selected = self.option_defaults | merged
        for name in ("steps", "width", "height", "num_images"):
            _positive_int(selected, name)
        if selected["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        for name in ("width", "height"):
            if selected[name] % PATCH_SIZE:
                raise ValueError(f"{name} must be a multiple of {PATCH_SIZE}")
        for name in ("guidance", "shift"):
            value = selected[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if selected["guidance"] < 0 or selected["shift"] <= 0:
            raise ValueError("guidance must be >= 0 and shift must be > 0")
        seed, noise = selected["seed"], selected["noise"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63 - 1):
            raise ValueError("seed must be an int in [0, 2**63 - 1)")
        shape = (selected["num_images"], 3, selected["height"], selected["width"])
        if noise is not None:
            if seed is not None:
                raise ValueError("Pass either seed or noise, not both")
            if (
                not isinstance(noise, Tensor)
                or not noise.is_floating_point()
                or tuple(noise.shape) != shape
            ):
                raise ValueError(f"noise must be a floating-point tensor of shape {shape}")
        timesteps = selected["timesteps"]
        if timesteps is not None:
            if (
                not isinstance(timesteps, list)
                or not timesteps
                or not all(type(x) is int for x in timesteps)
            ):
                raise TypeError("timesteps must be a nonempty list of ints or None")
            if any(x < 0 or x >= 1000 for x in timesteps):
                raise ValueError("timesteps must be in [0, 1000)")
            selected["steps"] = len(timesteps)
        return selected

    def _load(self) -> None:
        self.components = HiDreamO1Components.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
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

    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed, noise = options["seed"], options["noise"]
        if noise is None:
            if seed is None:
                seed = secrets.randbits(63)
            noise = NOISE_SCALE * torch.randn(
                (options["num_images"], 3, options["height"], options["width"]),
                generator=torch.Generator("cpu").manual_seed(seed + 1),
            )
        patches = denoise(
            self.components,
            prompt=inputs.prompt,
            height=options["height"],
            width=options["width"],
            steps=options["steps"],
            guidance=float(options["guidance"]),
            shift=float(options["shift"]),
            noise=noise,
            timesteps=options["timesteps"],
        )
        pixels = decode_patches(patches, options["height"], options["width"])
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": options["steps"],
            "guidance": float(options["guidance"]),
            "shift": float(options["shift"]),
            "width": options["width"],
            "height": options["height"],
            "scheduler": "flow_unipc",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
