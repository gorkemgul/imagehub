"""PixelDiT text-to-image backend."""

# Derivative of NVlabs PixelDiT under NSCLv1; non-commercial. See licenses/pixeldit.

from __future__ import annotations

import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.checkpoints import load_weights, read_json, read_snapshot
from imagehub.image import Image
from imagehub.models.gemma2 import Gemma2Model
from imagehub.models.pixeldit import PixelDiT
from imagehub.tokenization_gemma import GemmaTokenizer
from imagehub.types import ImageResult, T2IRequest

CHI_PROMPT: tuple[str, ...] = (
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual '
    "descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, "
    "and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly "
    "without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round "
    "shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red "
    "flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, "
    "featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a "
    "double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any "
    "additional commentary or evaluations:",
    "User Prompt: ",
)
DEFAULT_NEGATIVE_PROMPT = "low quality, worst quality, over-saturated, blurry, deformed, watermark"
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 50,
    "guidance": 2.75,
    "width": 1024,
    "height": 1024,
    "seed": 0,
    "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    "interval_guidance": (0.0, 1.0),
    "use_chi_prompt": True,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@dataclass
class PixelDiTComponents:
    tokenizer: GemmaTokenizer
    text_encoder: Gemma2Model
    transformer: PixelDiT

    @classmethod
    def load(
        cls, directory: str | os.PathLike[str], *, device: torch.device, dtype: torch.dtype
    ) -> PixelDiTComponents:
        root = Path(directory)
        text_root = root / "text_encoder"
        text_config = read_json(text_root / "config.json")
        model_config = read_json(root / "config.json")
        with torch.device("meta"):
            text_encoder = Gemma2Model(text_config)
            transformer = PixelDiT(model_config)
        text_state = load_weights(text_root, skip_prefixes=("lm_head.",))
        text_encoder.load_checkpoint(
            {
                name.removeprefix("model."): value
                for name, value in text_state.items()
                if name.startswith("model.")
            }
        )
        text_encoder.init_non_persistent_buffers()
        text_encoder.to(device=device, dtype=dtype).eval()
        transformer.load_checkpoint(
            load_weights(root / "pixeldit_t2i_v1.pth", container_key="state_dict")
        )
        transformer.to(device=device, dtype=dtype).eval()
        return cls(GemmaTokenizer(text_root), text_encoder, transformer)


@torch.inference_mode()
def encode_prompt(
    components: PixelDiTComponents,
    prompt: str,
    negative_prompt: str,
    device: torch.device,
    *,
    use_chi_prompt: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    header = "\n".join(CHI_PROMPT) if use_chi_prompt else ""
    max_length = len(components.tokenizer.tokenize(header)) + 300 - 2 if use_chi_prompt else 300
    input_ids, attention_mask = components.tokenizer(
        header + prompt.strip(), max_length=max_length, padding_side="right"
    )
    negative_ids, negative_mask = components.tokenizer(
        negative_prompt, max_length=300, padding_side="right"
    )
    select = [0, *range(-299, 0)]
    positive = components.text_encoder(input_ids.to(device), attention_mask.to(device))[:, select]
    negative = components.text_encoder(negative_ids.to(device), negative_mask.to(device))[:, select]
    return (
        positive,
        attention_mask[:, select].to(device),
        negative,
        negative_mask[:, select].to(device),
    )


def _lambda(timestep: Tensor) -> Tensor:
    return torch.log(1 - timestep) - torch.log(timestep)


def _data_prediction(
    transformer: PixelDiT,
    sample: Tensor,
    timestep: Tensor,
    condition: Tensor,
    unconditional: Tensor,
    guidance: float,
    interval: tuple[float, float],
) -> Tensor:
    continuous = timestep.expand(sample.shape[0])
    model_timestep = continuous * 1000
    if guidance == 1.0 or not interval[0] < float(timestep) < interval[1]:
        velocity = transformer(sample, model_timestep, condition)
        noise = (1 - continuous).view(-1, 1, 1, 1).to(sample) * velocity + sample
    else:
        doubled = torch.cat([sample, sample])
        times = torch.cat([model_timestep, model_timestep])
        text = torch.cat([unconditional, condition])
        velocity = transformer(doubled, times, text)
        sigma = torch.cat([continuous, continuous])
        noise = (1 - sigma).view(-1, 1, 1, 1).to(sample) * velocity + doubled
        noise_unconditional, noise_conditional = noise.chunk(2)
        noise = noise_unconditional + guidance * (noise_conditional - noise_unconditional)
    return (sample - timestep * noise) / (1 - timestep)


def _first_order(sample: Tensor, model: Tensor, start: Tensor, end: Tensor) -> Tensor:
    step = _lambda(end) - _lambda(start)
    sigma_start, sigma_end = start, end
    alpha_end = 1 - end
    return sigma_end / sigma_start * sample - alpha_end * torch.expm1(-step) * model


def _second_order(
    sample: Tensor,
    previous: Tensor,
    current: Tensor,
    previous_time: Tensor,
    current_time: Tensor,
    end: Tensor,
) -> Tensor:
    lambda_previous = _lambda(previous_time)
    lambda_current = _lambda(current_time)
    step = _lambda(end) - lambda_current
    ratio = (lambda_current - lambda_previous) / step
    derivative = (1.0 / ratio) * (current - previous)
    phi = torch.expm1(-step)
    return (
        end / current_time * sample - (1 - end) * phi * current - 0.5 * (1 - end) * phi * derivative
    )


@torch.inference_mode()
def sample_flow_dpm(
    transformer: PixelDiT,
    noise: Tensor,
    condition: Tensor,
    unconditional: Tensor,
    *,
    steps: int,
    guidance: float,
    flow_shift: float = 4.0,
    interval_guidance: tuple[float, float] = (0.0, 1.0),
) -> Tensor:
    beta = torch.linspace(1.0, 0.001, steps + 1, device=noise.device)
    sigma = 1 - beta
    timesteps = (flow_shift * sigma / (1 + (flow_shift - 1) * sigma)).flip(0)
    sample = noise
    previous_time = timesteps[0]
    previous_model = _data_prediction(
        transformer,
        sample,
        previous_time,
        condition,
        unconditional,
        guidance,
        interval_guidance,
    )
    older_time, older_model = previous_time, previous_model
    for index in range(1, steps + 1):
        current_time = timesteps[index]
        if index == 1 or index == steps:
            sample = _first_order(sample, previous_model, previous_time, current_time)
        else:
            sample = _second_order(
                sample,
                older_model,
                previous_model,
                older_time,
                previous_time,
                current_time,
            )
        if index < steps:
            older_time, older_model = previous_time, previous_model
            previous_time = current_time
            previous_model = _data_prediction(
                transformer,
                sample,
                previous_time,
                condition,
                unconditional,
                guidance,
                interval_guidance,
            )
    return sample


class PixelDiTBackend(TextToImageBackend):
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
        self.components: PixelDiTComponents | None = None
        self.snapshot: dict[str, Any] | None = None

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown PixelDiT options: {', '.join(unknown)}")
        result = OPTION_DEFAULTS | merged
        for name in ("steps", "width", "height"):
            if type(result[name]) is not int or result[name] <= 0:
                raise ValueError(f"{name} must be a positive int")
        if result["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        for name in ("width", "height"):
            if result[name] % 16:
                raise ValueError(f"{name} must be divisible by 16")
        guidance = result["guidance"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, int | float)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError("guidance must be a finite number >= 0")
        if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**64:
            raise ValueError("seed must be an int in [0, 2**64)")
        if not isinstance(result["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string")
        interval = result["interval_guidance"]
        if (
            not isinstance(interval, list | tuple)
            or len(interval) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int | float) for value in interval
            )
            or not 0 <= interval[0] <= interval[1] <= 1
        ):
            raise ValueError("interval_guidance must be two numbers in increasing order in [0, 1]")
        result["interval_guidance"] = (float(interval[0]), float(interval[1]))
        if type(result["use_chi_prompt"]) is not bool:
            raise TypeError("use_chi_prompt must be a bool")
        return result

    def _load(self) -> None:
        self.components = PixelDiTComponents.load(self.source, device=self.device, dtype=self.dtype)
        self.snapshot = read_snapshot(self.source)

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    @torch.inference_mode()
    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed = options["seed"] if options["seed"] is not None else secrets.randbits(63)
        condition, _, unconditional, _ = encode_prompt(
            self.components,
            inputs.prompt,
            options["negative_prompt"],
            self.device,
            use_chi_prompt=options["use_chi_prompt"],
        )
        generator = torch.Generator("cpu").manual_seed(seed)
        noise = torch.randn(
            (1, 3, options["height"], options["width"]),
            generator=generator,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.dtype)
        sample = sample_flow_dpm(
            self.components.transformer,
            noise,
            condition.to(self.dtype),
            unconditional.to(self.dtype),
            steps=options["steps"],
            guidance=float(options["guidance"]),
            interval_guidance=options["interval_guidance"],
        )
        pixels = ((sample + 1) / 2).clamp(0, 1).mul(255).round().permute(0, 2, 3, 1)
        images = pixels.to(device="cpu", dtype=torch.uint8)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id", self.model_id),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": options["steps"],
            "guidance": float(options["guidance"]),
            "width": options["width"],
            "height": options["height"],
            "sampler": "flow_dpm-solver",
            "flow_shift": 4.0,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(image) for image in images), metadata)
