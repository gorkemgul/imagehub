"""UnCLIP scheduler used by the Kandinsky 2.2 prior."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

UNCLIP_DEFAULTS: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "variance_type": "fixed_small_log",
    "clip_sample": True,
    "clip_sample_range": 1.0,
    "prediction_type": "epsilon",
    "beta_schedule": "squaredcos_cap_v2",
}


def _betas_for_alpha_bar(num_diffusion_timesteps: int) -> Tensor:
    def alpha_bar(time: float) -> float:
        return math.cos((time + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for index in range(num_diffusion_timesteps):
        time_1 = index / num_diffusion_timesteps
        time_2 = (index + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(time_2) / alpha_bar(time_1), 0.999))
    return torch.tensor(betas, dtype=torch.float32)


class UnCLIPScheduler:
    init_noise_sigma = 1.0

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in UNCLIP_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported UnCLIP setting {unknown[0]}")
        self.config = UNCLIP_DEFAULTS | {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        if self.config["beta_schedule"] != "squaredcos_cap_v2":
            raise ValueError("UnCLIPScheduler only supports squaredcos_cap_v2")
        self.betas = _betas_for_alpha_bar(self.config["num_train_timesteps"])
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.num_inference_steps = 0
        self.timesteps = torch.arange(self.config["num_train_timesteps"] - 1, -1, -1)
        self.variance_type = self.config["variance_type"]

    def scale_model_input(self, sample: Tensor, timestep: int | None = None) -> Tensor:
        return sample

    def set_timesteps(self, num_inference_steps: int) -> None:
        self.num_inference_steps = num_inference_steps
        step_ratio = (self.config["num_train_timesteps"] - 1) / (num_inference_steps - 1)
        timesteps = torch.arange(num_inference_steps, dtype=torch.float64) * step_ratio
        self.timesteps = timesteps.round().flip(0).to(torch.int64)

    def _get_variance(self, timestep: int, prev_timestep: int) -> Tensor:
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        beta = (
            self.betas[timestep]
            if prev_timestep == timestep - 1
            else 1 - alpha_prod_t / alpha_prod_t_prev
        )
        variance = beta_prod_t_prev / beta_prod_t * beta
        variance = torch.log(torch.clamp(variance, min=1e-20))
        return torch.exp(0.5 * variance)

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        *,
        prev_timestep: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if self.num_inference_steps == 0:
            raise ValueError("Call set_timesteps before step")
        if prev_timestep is None:
            prev_timestep = timestep - 1
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        if prev_timestep == timestep - 1:
            beta = self.betas[timestep]
            alpha = self.alphas[timestep]
        else:
            beta = 1 - alpha_prod_t / alpha_prod_t_prev
            alpha = 1 - beta
        if self.config["prediction_type"] == "epsilon":
            pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        elif self.config["prediction_type"] == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(
                f"Unsupported UnCLIP prediction_type {self.config['prediction_type']!r}"
            )
        if self.config["clip_sample"]:
            pred_original_sample = torch.clamp(
                pred_original_sample,
                -self.config["clip_sample_range"],
                self.config["clip_sample_range"],
            )
        pred_original_sample_coeff = alpha_prod_t_prev**0.5 * beta / beta_prod_t
        current_sample_coeff = alpha**0.5 * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        )
        variance: Tensor | int = 0
        if timestep > 0:
            noise_device = sample.device
            if generator is not None and generator.device.type == "cpu":
                noise_device = torch.device("cpu")
            variance_noise = torch.randn(
                model_output.shape,
                dtype=model_output.dtype,
                generator=generator,
                device=noise_device,
            ).to(model_output.device)
            variance = self._get_variance(timestep, prev_timestep) * variance_noise
        return pred_prev_sample + variance
