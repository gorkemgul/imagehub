"""Second-order DEIS scheduler for published Stable Diffusion configurations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

DEIS_DEFAULTS: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "trained_betas": None,
    "solver_order": 2,
    "prediction_type": "epsilon",
    "thresholding": False,
    "dynamic_thresholding_ratio": 0.995,
    "sample_max_value": 1.0,
    "algorithm_type": "deis",
    "solver_type": "logrho",
    "lower_order_final": True,
    "use_karras_sigmas": False,
    "use_exponential_sigmas": False,
    "use_beta_sigmas": False,
    "use_flow_sigmas": False,
    "flow_shift": 1.0,
    "timestep_spacing": "linspace",
    "steps_offset": 0,
    "use_dynamic_shifting": False,
    "time_shift_type": "exponential",
}
_IGNORED_COMPATIBILITY_KEYS = {
    "clip_sample",
    "clip_sample_range",
    "set_alpha_to_one",
    "skip_prk_steps",
}


class DEISMultistepScheduler:
    """The second-order log-rho DEIS mode used by Small-SD and DreamShaper."""

    init_noise_sigma = 1.0

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key
            for key in config
            if not key.startswith("_")
            and key not in DEIS_DEFAULTS
            and key not in _IGNORED_COMPATIBILITY_KEYS
        )
        if unknown:
            raise ValueError(f"Unsupported DEIS setting {unknown[0]}")
        cfg = DEIS_DEFAULTS | {
            key: value
            for key, value in config.items()
            if not key.startswith("_") and key in DEIS_DEFAULTS
        }
        required = {
            "trained_betas": None,
            "solver_order": 2,
            "prediction_type": "epsilon",
            "thresholding": False,
            "algorithm_type": "deis",
            "solver_type": "logrho",
            "use_karras_sigmas": False,
            "use_exponential_sigmas": False,
            "use_beta_sigmas": False,
            "use_flow_sigmas": False,
            "flow_shift": 1.0,
            "use_dynamic_shifting": False,
            "time_shift_type": "exponential",
        }
        for key, expected in required.items():
            if cfg[key] != expected:
                raise ValueError(f"Unsupported DEIS setting {key}={cfg[key]!r}")
        if cfg["timestep_spacing"] not in ("linspace", "leading"):
            raise ValueError(
                f"Unsupported DEIS setting timestep_spacing={cfg['timestep_spacing']!r}"
            )
        if cfg["beta_schedule"] == "scaled_linear":
            betas = (
                torch.linspace(
                    cfg["beta_start"] ** 0.5,
                    cfg["beta_end"] ** 0.5,
                    cfg["num_train_timesteps"],
                    dtype=torch.float32,
                )
                ** 2
            )
        elif cfg["beta_schedule"] == "linear":
            betas = torch.linspace(
                cfg["beta_start"],
                cfg["beta_end"],
                cfg["num_train_timesteps"],
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unsupported DEIS setting beta_schedule={cfg['beta_schedule']!r}")
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self._training_sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.config = cfg
        self.timesteps = torch.empty(0, dtype=torch.int64)
        self.sigmas = torch.empty(0, dtype=torch.float32)
        self.model_outputs: list[Tensor | None] = [None, None]
        self.lower_order_nums = 0
        self.num_inference_steps: int | None = None
        self._step_index: int | None = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_timesteps(self, num_inference_steps: int) -> None:
        count = self.config["num_train_timesteps"]
        if type(num_inference_steps) is not int or not 0 < num_inference_steps <= count:
            raise ValueError(f"num_inference_steps must be an int in [1, {count}]")
        if self.config["timestep_spacing"] == "linspace":
            timesteps = (
                torch.linspace(0, count - 1, num_inference_steps + 1, dtype=torch.float64)
                .round()
                .flip(0)[:-1]
                .to(torch.int64)
            )
        elif self.config["timestep_spacing"] == "leading":
            step_ratio = count // (num_inference_steps + 1)
            timesteps = (
                torch.arange(num_inference_steps + 1, dtype=torch.int64).flip(0)[:-1] * step_ratio
                + self.config["steps_offset"]
            )
        else:
            raise ValueError(
                f"Unsupported DEIS setting timestep_spacing={self.config['timestep_spacing']!r}"
            )
        sigmas = self._training_sigmas[timesteps]
        sigma_last = self._training_sigmas[0]
        self.timesteps = timesteps
        self.sigmas = torch.cat([sigmas, sigma_last.reshape(1)]).to(torch.float32)
        self.model_outputs = [None, None]
        self.lower_order_nums = 0
        self.num_inference_steps = len(timesteps)
        self._step_index = None

    def _init_step_index(self, timestep: int | Tensor) -> None:
        matches = (self.timesteps == timestep).nonzero()
        self._step_index = int(matches[1 if len(matches) > 1 else 0].item())

    @staticmethod
    def _sigma_to_alpha_sigma_t(sigma: Tensor) -> tuple[Tensor, Tensor]:
        alpha = 1 / ((sigma**2 + 1) ** 0.5)
        return alpha, sigma * alpha

    def _convert_model_output(self, model_output: Tensor, sample: Tensor) -> Tensor:
        assert self.step_index is not None
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(self.sigmas[self.step_index])
        x0 = (sample - sigma_t * model_output) / alpha_t
        return (sample - alpha_t * x0) / sigma_t

    def _first_order_update(self, model_output: Tensor, sample: Tensor) -> Tensor:
        assert self.step_index is not None
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(self.sigmas[self.step_index + 1])
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(self.sigmas[self.step_index])
        h = (torch.log(alpha_t) - torch.log(sigma_t)) - (torch.log(alpha_s) - torch.log(sigma_s))
        return (alpha_t / alpha_s) * sample - sigma_t * (torch.exp(h) - 1.0) * model_output

    def _second_order_update(self, sample: Tensor) -> Tensor:
        assert self.step_index is not None
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(self.sigmas[self.step_index + 1])
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(self.sigmas[self.step_index])
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(self.sigmas[self.step_index - 1])
        m0, m1 = self.model_outputs[-1], self.model_outputs[-2]
        assert m0 is not None and m1 is not None
        rho_t, rho_s0, rho_s1 = sigma_t / alpha_t, sigma_s0 / alpha_s0, sigma_s1 / alpha_s1

        def integral(t: Tensor, b: Tensor, c: Tensor) -> Tensor:
            def log(value: Tensor) -> Tensor:
                return value.new_tensor(math.log(value.item()))

            return t * (-log(c) + log(t) - 1) / (log(b) - log(c))

        coef1 = integral(rho_t, rho_s0, rho_s1) - integral(rho_s0, rho_s0, rho_s1)
        coef2 = integral(rho_t, rho_s1, rho_s0) - integral(rho_s0, rho_s1, rho_s0)
        return alpha_t * (sample / alpha_s0 + coef1 * m0 + coef2 * m1)

    def step(self, model_output: Tensor, timestep: int | Tensor, sample: Tensor) -> Tensor:
        if self.num_inference_steps is None:
            raise RuntimeError("Call set_timesteps before step")
        if self.step_index is None:
            self._init_step_index(timestep)
        assert self.step_index is not None
        lower_order_final = (
            self.step_index == len(self.timesteps) - 1
            and self.config["lower_order_final"]
            and len(self.timesteps) < 15
        )
        converted = self._convert_model_output(model_output, sample)
        self.model_outputs[0] = self.model_outputs[1]
        self.model_outputs[1] = converted
        if self.lower_order_nums < 1 or lower_order_final:
            previous = self._first_order_update(converted, sample)
        else:
            previous = self._second_order_update(sample)
        if self.lower_order_nums < 2:
            self.lower_order_nums += 1
        assert self._step_index is not None
        self._step_index += 1
        return previous

    @staticmethod
    def scale_model_input(sample: Tensor, timestep: int | Tensor | None = None) -> Tensor:
        return sample
