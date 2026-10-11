"""Euler ancestral scheduler for the SDXL Turbo release configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from imagehub.schedulers.euler import scaled_linear_training_sigmas

_DEFAULTS: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "trained_betas": None,
    "prediction_type": "epsilon",
    "timestep_spacing": "linspace",
    "steps_offset": 0,
    "rescale_betas_zero_snr": False,
    "clip_sample": False,
    "interpolation_type": "linear",
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    # Published by some releases; diffusers' constructor does not read them, so only their
    # inert values are accepted.
    "sigma_min": None,
    "sigma_max": None,
    "timestep_type": "discrete",
    "use_karras_sigmas": False,
}
_REQUIRED = {
    "beta_schedule": "scaled_linear",
    "trained_betas": None,
    "prediction_type": "epsilon",
    "rescale_betas_zero_snr": False,
    "clip_sample": False,
    "interpolation_type": "linear",
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "sigma_min": None,
    "sigma_max": None,
    "timestep_type": "discrete",
    "use_karras_sigmas": False,
}


class EulerAncestralDiscreteScheduler:
    """Stateful Euler ancestral sampler for scaled-linear SDXL schedules."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(key for key in config if not key.startswith("_") and key not in _DEFAULTS)
        if unknown:
            raise ValueError(f"Unsupported Euler ancestral setting {unknown[0]}")
        cfg = _DEFAULTS | {key: value for key, value in config.items() if not key.startswith("_")}
        for key, expected in _REQUIRED.items():
            if cfg[key] != expected:
                raise ValueError(f"Unsupported Euler ancestral setting {key}={cfg[key]!r}")
        if cfg["timestep_spacing"] not in ("leading", "trailing"):
            raise ValueError(
                f"Unsupported Euler ancestral setting timestep_spacing={cfg['timestep_spacing']!r}"
            )
        self.config = cfg
        self._training_sigmas = scaled_linear_training_sigmas(cfg)
        self.timesteps = torch.empty(0, dtype=torch.float32)
        self.sigmas = torch.empty(0, dtype=torch.float32)
        self.num_inference_steps = 0
        self._step_index: int | None = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def init_noise_sigma(self) -> Tensor:
        if self.num_inference_steps == 0:
            raise RuntimeError("Call set_timesteps before reading init_noise_sigma")
        maximum = self.sigmas.max()
        if self.config["timestep_spacing"] == "trailing":
            return maximum
        return (maximum**2 + 1) ** 0.5

    def set_timesteps(self, num_inference_steps: int) -> None:
        count = self.config["num_train_timesteps"]
        if type(num_inference_steps) is not int or not 0 < num_inference_steps <= count:
            raise ValueError(f"num_inference_steps must be an int in [1, {count}]")
        self.num_inference_steps = num_inference_steps
        if self.config["timestep_spacing"] == "leading":
            ratio = count // num_inference_steps
            timesteps = (
                torch.arange(num_inference_steps, dtype=torch.float64).mul(ratio).round().flip(0)
                + self.config["steps_offset"]
            )
        else:
            ratio = count / num_inference_steps
            timesteps = torch.arange(count, 0, -ratio, dtype=torch.float64).round() - 1
        self.timesteps = timesteps.to(torch.float32)
        indices = self.timesteps.to(torch.int64).clamp_max(count - 1)
        selected = self._training_sigmas[indices]
        self.sigmas = torch.cat([selected, torch.zeros(1, dtype=torch.float32)])
        self._step_index = None

    def _init_step_index(self, timestep: Tensor | float) -> None:
        indices = (self.timesteps == torch.as_tensor(timestep)).nonzero()
        if not len(indices):
            raise ValueError(f"timestep {timestep} is not in the current schedule")
        position = 1 if len(indices) > 1 else 0
        self._step_index = int(indices[position].item())

    def scale_model_input(self, sample: Tensor, timestep: Tensor | float) -> Tensor:
        if self.num_inference_steps == 0:
            raise RuntimeError("Call set_timesteps before scale_model_input")
        if self._step_index is None:
            self._init_step_index(timestep)
        assert self._step_index is not None
        sigma = self.sigmas[self._step_index]
        return sample / ((sigma**2 + 1) ** 0.5)

    def step(
        self,
        model_output: Tensor,
        timestep: Tensor | float,
        sample: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if self.num_inference_steps == 0:
            raise RuntimeError("Call set_timesteps before step")
        if self._step_index is None:
            self._init_step_index(timestep)
        assert self._step_index is not None
        sigma = self.sigmas[self._step_index]
        sample = sample.to(torch.float32)
        pred_original_sample = sample - sigma * model_output
        sigma_from = self.sigmas[self._step_index]
        sigma_to = self.sigmas[self._step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
        derivative = (sample - pred_original_sample) / sigma
        previous = sample + derivative * (sigma_down - sigma)

        device = model_output.device
        random_device = device
        if generator is not None and generator.device.type == "cpu":
            random_device = torch.device("cpu")
        noise = torch.randn(
            model_output.shape,
            generator=generator,
            device=random_device,
            dtype=model_output.dtype,
        ).to(device)
        previous = previous + noise * sigma_up
        self._step_index += 1
        return previous.to(model_output.dtype)
