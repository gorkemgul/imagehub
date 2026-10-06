"""Latent Consistency Model scheduler for the Dreamshaper v7 release configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

_DEFAULTS: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "trained_betas": None,
    "original_inference_steps": 50,
    "clip_sample": False,
    "clip_sample_range": 1.0,
    "set_alpha_to_one": True,
    "steps_offset": 0,
    "prediction_type": "epsilon",
    "thresholding": False,
    "dynamic_thresholding_ratio": 0.995,
    "sample_max_value": 1.0,
    "timestep_spacing": "leading",
    "timestep_scaling": 10.0,
    "rescale_betas_zero_snr": False,
}

_IGNORED = frozenset({"steps_offset"})


class LCMScheduler:
    """Stateful LCM sampler. Create one per trajectory and set its timesteps first."""

    init_noise_sigma = 1.0

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(key for key in config if not key.startswith("_") and key not in _DEFAULTS)
        if unknown:
            raise ValueError(f"Unsupported LCM setting {unknown[0]}")
        cfg = _DEFAULTS | {key: value for key, value in config.items() if not key.startswith("_")}
        for key, expected in _DEFAULTS.items():
            # The reference declares steps_offset but never uses it in LCM sampling.
            if key in _IGNORED:
                continue
            if cfg[key] != expected:
                raise ValueError(f"Unsupported LCM setting {key}={cfg[key]!r}")
        self.config = cfg
        count = cfg["num_train_timesteps"]
        betas = (
            torch.linspace(
                cfg["beta_start"] ** 0.5,
                cfg["beta_end"] ** 0.5,
                count,
                dtype=torch.float32,
            )
            ** 2
        )
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0)
        self.timesteps = torch.arange(count - 1, -1, -1, dtype=torch.int64)
        self.num_inference_steps = 0
        self._step_index: int | None = None

    def set_timesteps(self, num_inference_steps: int) -> None:
        original_steps = self.config["original_inference_steps"]
        if type(num_inference_steps) is not int or not 0 < num_inference_steps <= original_steps:
            raise ValueError(f"num_inference_steps must be an int in [1, {original_steps}]")
        count = self.config["num_train_timesteps"]
        skip = count // original_steps
        original = torch.arange(1, original_steps + 1, dtype=torch.int64) * skip - 1
        original = original.flip(0)
        indices = [
            math.floor(i * original_steps / num_inference_steps) for i in range(num_inference_steps)
        ]
        self.timesteps = original[torch.tensor(indices)]
        self.num_inference_steps = num_inference_steps
        self._step_index = None

    def _init_step_index(self, timestep: Tensor | int) -> None:
        indices = (self.timesteps == torch.as_tensor(timestep)).nonzero()
        if not len(indices):
            raise ValueError(f"timestep {timestep} is not in the current schedule")
        position = 1 if len(indices) > 1 else 0
        self._step_index = int(indices[position].item())

    def get_scalings_for_boundary_condition_discrete(
        self, timestep: Tensor | int
    ) -> tuple[Tensor, Tensor]:
        scaled_timestep = torch.as_tensor(timestep) * self.config["timestep_scaling"]
        sigma_data = 0.5
        c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
        return c_skip, c_out

    def step(
        self,
        model_output: Tensor,
        timestep: Tensor | int,
        sample: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.num_inference_steps == 0:
            raise ValueError("Call set_timesteps before step")
        if self._step_index is None:
            self._init_step_index(timestep)
        assert self._step_index is not None
        previous_index = self._step_index + 1
        previous_timestep = (
            self.timesteps[previous_index]
            if previous_index < len(self.timesteps)
            else torch.as_tensor(timestep)
        )
        timestep_index = int(torch.as_tensor(timestep).item())
        previous_timestep_index = int(previous_timestep.item())
        alpha_prod_t = self.alphas_cumprod[timestep_index].to(sample.device)
        alpha_prod_t_prev = self.alphas_cumprod[previous_timestep_index].to(sample.device)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)
        c_skip = c_skip.to(sample.device)
        c_out = c_out.to(sample.device)
        predicted_original = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        denoised = c_out * predicted_original + c_skip * sample
        if self._step_index != self.num_inference_steps - 1:
            random_device = model_output.device
            if generator is not None and generator.device.type == "cpu":
                random_device = torch.device("cpu")
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=random_device,
                dtype=denoised.dtype,
            ).to(model_output.device)
            previous = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
        else:
            previous = denoised
        self._step_index += 1
        return previous, denoised
