"""Logit-normal schedule and Euler flow-matching sampler for Ideogram 4."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LogitNormalSchedule:
    mean: float
    std: float = 1.0
    logsnr_min: float = -15.0
    logsnr_max: float = 18.0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float64)
        z = torch.special.ndtri(t)
        y = self.mean + self.std * z
        t_ = torch.special.expit(y)
        t_ = 1 - t_
        t_min = 1.0 / (1 + math.exp(0.5 * self.logsnr_max))
        t_max = 1.0 / (1 + math.exp(0.5 * self.logsnr_min))
        return t_.clamp(t_min, t_max).to(torch.float32)


def get_schedule_for_resolution(
    image_resolution: tuple[int, int],
    known_resolution: tuple[int, int] = (512, 512),
    known_mean: float = 1.0,
    std: float = 1.0,
) -> LogitNormalSchedule:
    """Return the resolution-aware evaluation schedule."""
    num_pixels = image_resolution[0] * image_resolution[1]
    known_pixels = known_resolution[0] * known_resolution[1]
    mean = known_mean + 0.5 * math.log(num_pixels / known_pixels)
    return LogitNormalSchedule(mean=mean, std=std)


def make_step_intervals(num_steps: int) -> torch.Tensor:
    """Return the linear step schedule used by the evaluation config."""
    return torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32)


@dataclass(frozen=True, kw_only=True)
class SamplerParameters:
    num_steps: int
    guidance_schedule: tuple[float, ...]
    mu: float
    std: float = 1.0

    def __post_init__(self) -> None:
        if len(self.guidance_schedule) != self.num_steps:
            raise ValueError(
                f"guidance_schedule has length {len(self.guidance_schedule)}, "
                f"expected num_steps={self.num_steps}"
            )


PRESETS: dict[str, SamplerParameters] = {
    "V4_QUALITY_48": SamplerParameters(
        num_steps=48,
        guidance_schedule=(3.0,) * 3 + (7.0,) * 45,
        mu=0.0,
        std=1.5,
    ),
    "V4_DEFAULT_20": SamplerParameters(
        num_steps=20,
        guidance_schedule=(3.0,) * 2 + (7.0,) * 18,
        mu=0.0,
        std=1.75,
    ),
    "V4_TURBO_12": SamplerParameters(
        num_steps=12,
        guidance_schedule=(3.0,) * 1 + (7.0,) * 11,
        mu=0.5,
        std=1.75,
    ),
}


def ideogram4_step(
    sample: torch.Tensor,
    positive_output: torch.Tensor,
    negative_output: torch.Tensor,
    guidance_scale: float | torch.Tensor,
    t: float,
    s: float,
) -> torch.Tensor:
    """Apply one classifier-free-guided Euler flow-matching update."""
    velocity = guidance_scale * positive_output + (1.0 - guidance_scale) * negative_output
    delta = s - t
    return sample + velocity * delta
