"""Euler discrete scheduler for the Stable Diffusion XL release configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

EULER_DEFAULTS: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "prediction_type": "epsilon",
    "interpolation_type": "linear",
    "use_karras_sigmas": False,
    "use_exponential_sigmas": False,
    "use_beta_sigmas": False,
    "sigma_min": None,
    "sigma_max": None,
    "timestep_spacing": "linspace",
    "timestep_type": "discrete",
    "steps_offset": 0,
    "trained_betas": None,
    "rescale_betas_zero_snr": False,
    "final_sigmas_type": "zero",
}
_REQUIRED = {
    "beta_schedule": "scaled_linear",
    "prediction_type": "epsilon",
    "interpolation_type": "linear",
    "use_karras_sigmas": False,
    "use_exponential_sigmas": False,
    "use_beta_sigmas": False,
    "sigma_min": None,
    "sigma_max": None,
    "timestep_spacing": "leading",
    "timestep_type": "discrete",
    "trained_betas": None,
    "rescale_betas_zero_snr": False,
    "final_sigmas_type": "zero",
}
_IGNORED_COMPATIBILITY_KEYS = {
    "clip_sample",
    "sample_max_value",
    "set_alpha_to_one",
    "skip_prk_steps",
}


class EulerDiscreteScheduler:
    """Stateful Euler sampler. Create one per trajectory and set its timesteps first."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key
            for key in config
            if not key.startswith("_")
            and key not in EULER_DEFAULTS
            and key not in _IGNORED_COMPATIBILITY_KEYS
        )
        if unknown:
            raise ValueError(f"Unsupported Euler setting {unknown[0]}")
        cfg = EULER_DEFAULTS | {k: v for k, v in config.items() if not k.startswith("_")}
        for key, expected in _REQUIRED.items():
            if cfg[key] != expected:
                raise ValueError(f"Unsupported Euler setting {key}={cfg[key]!r}")
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
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self._training_sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.config = cfg
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
        return (maximum**2 + 1) ** 0.5

    def set_timesteps(self, num_inference_steps: int) -> None:
        count = self.config["num_train_timesteps"]
        if type(num_inference_steps) is not int or not 0 < num_inference_steps <= count:
            raise ValueError(f"num_inference_steps must be an int in [1, {count}]")
        self.num_inference_steps = num_inference_steps
        ratio = count // num_inference_steps
        self.timesteps = (
            torch.arange(num_inference_steps, dtype=torch.float32).mul(ratio).flip(0)
            + self.config["steps_offset"]
        )
        # The supported leading schedule has integer timesteps. Selecting from the float32
        # training schedule exactly reproduces NumPy's float64 interpolation followed by
        # the reference's float32 cast, without returning float64 tensors.
        indices = self.timesteps.to(torch.int64).clamp_max(count - 1)
        selected = self._training_sigmas[indices]
        self.sigmas = torch.cat([selected, torch.zeros(1, dtype=torch.float32)])
        self._step_index = None

    def _init_step_index(self, timestep: Tensor | float) -> None:
        indices = (self.timesteps == torch.as_tensor(timestep)).nonzero()
        if not len(indices):
            raise ValueError(f"timestep {timestep} is not in the current schedule")
        self._step_index = int(indices[0].item())

    def scale_model_input(self, sample: Tensor, timestep: Tensor | float) -> Tensor:
        if self.num_inference_steps == 0:
            raise RuntimeError("Call set_timesteps before scale_model_input")
        if self._step_index is None:
            self._init_step_index(timestep)
        assert self._step_index is not None
        sigma = self.sigmas[self._step_index]
        return sample / ((sigma**2 + 1) ** 0.5)

    def step(self, model_output: Tensor, timestep: Tensor | float, sample: Tensor) -> Tensor:
        if self.num_inference_steps == 0:
            raise RuntimeError("Call set_timesteps before step")
        if self._step_index is None:
            self._init_step_index(timestep)
        assert self._step_index is not None
        sample = sample.to(torch.float32)
        sigma = self.sigmas[self._step_index]
        pred_original_sample = sample - sigma * model_output
        derivative = (sample - pred_original_sample) / sigma
        dt = self.sigmas[self._step_index + 1] - sigma
        previous = sample + derivative * dt
        self._step_index += 1
        return previous.to(model_output.dtype)
