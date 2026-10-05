"""EDM DPM-Solver++ scheduler for the Playground v2.5 release configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

EDM_DPMSOLVER_DEFAULTS: dict[str, Any] = {
    "sigma_min": 0.002,
    "sigma_max": 80.0,
    "sigma_data": 0.5,
    "sigma_schedule": "karras",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "rho": 7.0,
    "solver_order": 2,
    "thresholding": False,
    "dynamic_thresholding_ratio": 0.995,
    "sample_max_value": 1.0,
    "algorithm_type": "dpmsolver++",
    "solver_type": "midpoint",
    "lower_order_final": True,
    "euler_at_final": False,
    "final_sigmas_type": "zero",
}
_REQUIRED = dict(EDM_DPMSOLVER_DEFAULTS)


class EDMDPMSolverMultistepScheduler:
    """Second-order EDM DPM-Solver++ for the published Playground v2.5 config."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in EDM_DPMSOLVER_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported EDM DPM-Solver setting {unknown[0]}")
        cfg = EDM_DPMSOLVER_DEFAULTS | {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        for key, expected in _REQUIRED.items():
            if cfg[key] != expected:
                raise ValueError(f"Unsupported EDM DPM-Solver setting {key}={cfg[key]!r}")
        self.config = cfg
        ramp = torch.linspace(0, 1, cfg["num_train_timesteps"])
        sigmas = self._compute_karras_sigmas(ramp)
        self.timesteps = self.precondition_noise(sigmas)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)]).to("cpu")
        self.num_inference_steps: int | None = None
        self.model_outputs: list[Tensor | None] = [None] * cfg["solver_order"]
        self.lower_order_nums = 0
        self._step_index: int | None = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def init_noise_sigma(self) -> float:
        return (self.config["sigma_max"] ** 2 + 1) ** 0.5

    def _compute_karras_sigmas(self, ramp: Tensor) -> Tensor:
        rho = self.config["rho"]
        min_inv_rho = self.config["sigma_min"] ** (1 / rho)
        max_inv_rho = self.config["sigma_max"] ** (1 / rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

    @staticmethod
    def precondition_noise(sigma: Tensor | float) -> Tensor:
        if not isinstance(sigma, Tensor):
            sigma = torch.tensor([sigma])
        return 0.25 * torch.log(sigma)

    def set_timesteps(self, num_inference_steps: int) -> None:
        if type(num_inference_steps) is not int or num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be a positive int")
        self.num_inference_steps = num_inference_steps
        ramp = torch.linspace(0, 1, num_inference_steps)
        sigmas = self._compute_karras_sigmas(ramp).to(dtype=torch.float32)
        self.timesteps = self.precondition_noise(sigmas)
        self.sigmas = torch.cat([sigmas, torch.tensor([0], dtype=torch.float32)]).to("cpu")
        self.model_outputs = [None] * self.config["solver_order"]
        self.lower_order_nums = 0
        self._step_index = None

    def index_for_timestep(self, timestep: Tensor | float) -> int:
        candidates = (self.timesteps == timestep).nonzero()
        if len(candidates) == 0:
            return len(self.timesteps) - 1
        if len(candidates) > 1:
            return int(candidates[1].item())
        return int(candidates[0].item())

    def _init_step_index(self, timestep: Tensor | float) -> None:
        if isinstance(timestep, Tensor):
            timestep = timestep.to(self.timesteps.device)
        self._step_index = self.index_for_timestep(timestep)

    def scale_model_input(self, sample: Tensor, timestep: Tensor | float) -> Tensor:
        if self.num_inference_steps is None:
            raise RuntimeError("Call set_timesteps before scale_model_input")
        if self.step_index is None:
            self._init_step_index(timestep)
        assert self.step_index is not None
        sigma = self.sigmas[self.step_index]
        c_in = 1 / ((sigma**2 + self.config["sigma_data"] ** 2) ** 0.5)
        return sample * c_in

    def _convert_model_output(self, model_output: Tensor, sample: Tensor) -> Tensor:
        assert self.step_index is not None
        sigma = self.sigmas[self.step_index]
        sigma_data = self.config["sigma_data"]
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip * sample + c_out * model_output

    @staticmethod
    def _sigma_to_alpha_sigma_t(sigma: Tensor) -> tuple[Tensor, Tensor]:
        return torch.tensor(1), sigma

    def _first_order_update(self, model_output: Tensor, sample: Tensor) -> Tensor:
        assert self.step_index is not None
        sigma_t = self.sigmas[self.step_index + 1]
        sigma_s = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s
        return (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output

    def _second_order_update(self, sample: Tensor) -> Tensor:
        assert self.step_index is not None
        sigma_t = self.sigmas[self.step_index + 1]
        sigma_s0 = self.sigmas[self.step_index]
        sigma_s1 = self.sigmas[self.step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
        m0, m1 = self.model_outputs[-1], self.model_outputs[-2]
        assert m0 is not None and m1 is not None
        h = lambda_t - lambda_s0
        h_0 = lambda_s0 - lambda_s1
        r0 = h_0 / h
        d0 = m0
        d1 = (1.0 / r0) * (m0 - m1)
        return (
            (sigma_t / sigma_s0) * sample
            - (alpha_t * (torch.exp(-h) - 1.0)) * d0
            - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * d1
        )

    def step(self, model_output: Tensor, timestep: Tensor | float, sample: Tensor) -> Tensor:
        if self.num_inference_steps is None:
            raise RuntimeError("Call set_timesteps before step")
        if self.step_index is None:
            self._init_step_index(timestep)
        assert self.step_index is not None
        lower_order_final = (
            self.step_index == len(self.timesteps) - 1
            and self.config["final_sigmas_type"] == "zero"
        )
        converted = self._convert_model_output(model_output, sample)
        self.model_outputs[0] = self.model_outputs[1]
        self.model_outputs[1] = converted
        if self.lower_order_nums < 1 or lower_order_final:
            previous = self._first_order_update(converted, sample)
        else:
            previous = self._second_order_update(sample)
        if self.lower_order_nums < self.config["solver_order"]:
            self.lower_order_nums += 1
        assert self._step_index is not None
        self._step_index += 1
        return previous
