"""UniPC multistep sampler for HiDream-I1 Full's flow-prediction schedule."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

DEFAULTS: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "solver_order": 2,
    "prediction_type": "epsilon",
    "thresholding": False,
    "predict_x0": True,
    "solver_type": "bh2",
    "lower_order_final": True,
    "disable_corrector": [],
    "use_flow_sigmas": False,
    "flow_shift": 1.0,
    "timestep_spacing": "linspace",
    "steps_offset": 0,
    "final_sigmas_type": "zero",
    "use_karras_sigmas": False,
    "use_exponential_sigmas": False,
    "use_beta_sigmas": False,
    "rescale_betas_zero_snr": False,
    "trained_betas": None,
    "solver_p": None,
    "dynamic_thresholding_ratio": 0.995,
    "sample_max_value": 1.0,
}


class UniPCMultistepScheduler:
    """The published HiDream Full UniPC configuration (flow sigmas, UniPC-2 bh2)."""

    order = 1

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(key for key in config if not key.startswith("_") and key not in DEFAULTS)
        if unknown:
            raise ValueError(f"Unsupported UniPC setting {unknown[0]}")
        self.config = DEFAULTS | {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        required = {
            "beta_schedule": "linear",
            "solver_order": 2,
            "prediction_type": "flow_prediction",
            "thresholding": False,
            "predict_x0": True,
            "solver_type": "bh2",
            "lower_order_final": True,
            "use_flow_sigmas": True,
            "timestep_spacing": "linspace",
            "final_sigmas_type": "zero",
            "use_karras_sigmas": False,
            "use_exponential_sigmas": False,
            "use_beta_sigmas": False,
            "rescale_betas_zero_snr": False,
            "trained_betas": None,
            "solver_p": None,
        }
        for key, value in required.items():
            if self.config[key] != value:
                raise ValueError(f"Unsupported UniPC setting {key}={self.config[key]!r}")
        self.num_inference_steps: int | None = None
        self.timesteps = torch.arange(int(self.config["num_train_timesteps"]) - 1, -1, -1)
        self.sigmas = torch.empty(0)
        self.model_outputs: list[Tensor | None] = [None, None]
        self.lower_order_nums = 0
        self.last_sample: Tensor | None = None
        self._step_index: int | None = None
        self.this_order = 1

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_timesteps(self, steps: int) -> None:
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive int")
        count = int(self.config["num_train_timesteps"])
        start, stop = 1.0, 1.0 / count
        delta = (stop - start) / steps
        base = [start + index * delta for index in range(steps)]
        shift = float(self.config["flow_shift"])
        sigmas = [shift * value / (1 + (shift - 1) * value) for value in base]
        if abs(sigmas[0] - 1) < 1e-6:
            sigmas[0] -= 1e-6
        self.timesteps = torch.tensor([value * count for value in sigmas], dtype=torch.int64)
        self.sigmas = torch.tensor([*sigmas, 0.0], dtype=torch.float32)
        self.num_inference_steps = steps
        self.model_outputs = [None, None]
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None
        self.this_order = 1

    def _init_step_index(self, timestep: Tensor | int) -> None:
        candidates = (self.timesteps == timestep).nonzero()
        self._step_index = (
            int(candidates[1 if len(candidates) > 1 else 0].item())
            if len(candidates)
            else len(self.timesteps) - 1
        )

    @staticmethod
    def _alpha_sigma(sigma: Tensor) -> tuple[Tensor, Tensor]:
        return 1 - sigma, sigma

    def _update(self, sample: Tensor, order: int) -> Tensor:
        assert self._step_index is not None and self.model_outputs[-1] is not None
        index = self._step_index
        device = sample.device
        sigma_t, sigma_s = self.sigmas[index + 1].to(device), self.sigmas[index].to(device)
        alpha_t, sigma_t = self._alpha_sigma(sigma_t)
        alpha_s, sigma_s = self._alpha_sigma(sigma_s)
        h = (torch.log(alpha_t) - torch.log(sigma_t)) - (torch.log(alpha_s) - torch.log(sigma_s))
        phi1, bh = torch.expm1(-h), torch.expm1(-h)
        output = self.model_outputs[-1]
        prediction = sigma_t / sigma_s * sample - alpha_t * phi1 * output
        if order == 2:
            previous = self.model_outputs[-2]
            assert previous is not None
            alpha_p, sigma_p = self._alpha_sigma(self.sigmas[index - 1].to(device))
            rk = (
                (torch.log(alpha_p) - torch.log(sigma_p))
                - (torch.log(alpha_s) - torch.log(sigma_s))
            ) / h
            prediction -= alpha_t * bh * (0.5 * ((previous - output) / rk))
        return prediction.to(sample.dtype)

    def _correct(self, model_output: Tensor, sample: Tensor, order: int) -> Tensor:
        assert self._step_index is not None and self.last_sample is not None
        previous = self.model_outputs[-1]
        assert previous is not None
        index, device = self._step_index, sample.device
        sigma_t, sigma_s = self.sigmas[index].to(device), self.sigmas[index - 1].to(device)
        alpha_t, sigma_t = self._alpha_sigma(sigma_t)
        alpha_s, sigma_s = self._alpha_sigma(sigma_s)
        h = (torch.log(alpha_t) - torch.log(sigma_t)) - (torch.log(alpha_s) - torch.log(sigma_s))
        phi1, bh = torch.expm1(-h), torch.expm1(-h)
        base = sigma_t / sigma_s * self.last_sample - alpha_t * phi1 * previous
        if order == 1:
            return (base - alpha_t * bh * (0.5 * (model_output - previous))).to(sample.dtype)
        older = self.model_outputs[-2]
        assert older is not None
        alpha_o, sigma_o = self._alpha_sigma(self.sigmas[index - 2].to(device))
        rk = (
            (torch.log(alpha_o) - torch.log(sigma_o)) - (torch.log(alpha_s) - torch.log(sigma_s))
        ) / h
        d1 = (older - previous) / rk
        rhos = torch.linalg.solve(
            torch.stack([torch.ones(2, device=device), torch.tensor([rk, 1.0], device=device)]),
            torch.stack([(phi1 / -h - 1) / bh, ((phi1 / -h - 1) / -h - 0.5) * 2 / bh]),
        ).to(sample.dtype)
        return (base - alpha_t * bh * (rhos[0] * d1 + rhos[1] * (model_output - previous))).to(
            sample.dtype
        )

    def step(
        self,
        model_output: Tensor,
        timestep: Tensor | int,
        sample: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        del generator
        if self.num_inference_steps is None:
            raise RuntimeError("Call set_timesteps before step")
        if self._step_index is None:
            self._init_step_index(timestep)
        assert self._step_index is not None
        converted = sample - self.sigmas[self._step_index].to(sample.device) * model_output
        if self._step_index > 0 and self.last_sample is not None:
            sample = self._correct(converted, sample, self.this_order)
        for index in range(1):
            self.model_outputs[index] = self.model_outputs[index + 1]
        self.model_outputs[-1] = converted
        remaining = len(self.timesteps) - self._step_index
        order = min(2, remaining, self.lower_order_nums + 1)
        result = self._update(sample, order)
        self.this_order = order
        self.last_sample = sample
        self.lower_order_nums = min(self.lower_order_nums + 1, 2)
        self._step_index += 1
        return result
