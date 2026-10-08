"""Wuerstchen DDPM scheduler used by the Stable Cascade prior and decoder."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

WUERSTCHEN_DEFAULTS: dict[str, Any] = {"s": 0.008, "scaler": 1.0}


class DDPMWuerstchenScheduler:
    init_noise_sigma = 1.0

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in WUERSTCHEN_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported Wuerstchen scheduler setting {unknown[0]}")
        self.config = WUERSTCHEN_DEFAULTS | {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        if self.config["scaler"] <= 0:
            raise ValueError("Wuerstchen scheduler scaler must be positive")
        self.scaler = self.config["scaler"]
        self.s = torch.tensor([self.config["s"]])
        self._init_alpha_cumprod = torch.cos(self.s / (1 + self.s) * math.pi * 0.5) ** 2
        self.timesteps = torch.linspace(1.0, 0.0, 1)

    def _alpha_cumprod(self, t: Tensor, device: torch.device) -> Tensor:
        if self.scaler > 1:
            t = 1 - (1 - t) ** self.scaler
        elif self.scaler < 1:
            t = t**self.scaler
        s = self.s.to(device)
        alpha_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2 / (
            self._init_alpha_cumprod.to(device)
        )
        return alpha_cumprod.clamp(0.0001, 0.9999)

    def scale_model_input(self, sample: Tensor, timestep: Tensor | None = None) -> Tensor:
        return sample

    def set_timesteps(self, num_inference_steps: int) -> None:
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1)

    def previous_timestep(self, timestep: Tensor) -> Tensor:
        index = int((self.timesteps - timestep[0]).abs().argmin().item())
        return self.timesteps[index + 1][None].expand(timestep.shape[0])

    def step(
        self,
        model_output: Tensor,
        timestep: Tensor,
        sample: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        dtype = model_output.dtype
        device = model_output.device
        prev_t = self.previous_timestep(timestep)
        trailing = [1] * (sample.ndim - 1)
        alpha_cumprod = self._alpha_cumprod(timestep, device).view(timestep.size(0), *trailing)
        alpha_cumprod_prev = self._alpha_cumprod(prev_t, device).view(prev_t.size(0), *trailing)
        alpha = alpha_cumprod / alpha_cumprod_prev
        mu = (1.0 / alpha).sqrt() * (
            sample - (1 - alpha) * model_output / (1 - alpha_cumprod).sqrt()
        )
        noise_device = device
        if generator is not None and generator.device.type == "cpu":
            noise_device = torch.device("cpu")
        std_noise = torch.randn(
            mu.shape, generator=generator, device=noise_device, dtype=model_output.dtype
        ).to(device)
        std = ((1 - alpha) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)).sqrt() * std_noise
        pred = mu + std * (prev_t != 0).float().view(prev_t.size(0), *trailing)
        return pred.to(dtype)
