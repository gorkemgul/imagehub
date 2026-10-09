"""DPM-Solver++ multistep scheduler for the PixArt (epsilon) and Sana (flow) configurations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

DPMSOLVER_MULTISTEP_DEFAULTS: dict[str, Any] = {
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
    "algorithm_type": "dpmsolver++",
    "solver_type": "midpoint",
    "lower_order_final": True,
    "euler_at_final": False,
    "use_karras_sigmas": False,
    "use_exponential_sigmas": False,
    "use_beta_sigmas": False,
    "use_lu_lambdas": False,
    "use_flow_sigmas": False,
    "flow_shift": 1.0,
    "final_sigmas_type": "zero",
    "lambda_min_clipped": -float("inf"),
    "variance_type": None,
    "timestep_spacing": "linspace",
    "steps_offset": 0,
    "rescale_betas_zero_snr": False,
    "use_dynamic_shifting": False,
    "time_shift_type": "exponential",
}
# Settings the two supported modes share; ``prediction_type``, ``use_flow_sigmas`` and
# ``flow_shift`` are validated separately.
_MODE_KEYS = frozenset({"prediction_type", "use_flow_sigmas", "flow_shift"})
_REQUIRED = {
    key: value for key, value in DPMSOLVER_MULTISTEP_DEFAULTS.items() if key not in _MODE_KEYS
}


class DPMSolverMultistepScheduler:
    """Second-order DPM-Solver++: epsilon prediction (PixArt) or flow sigmas (Sana)."""

    init_noise_sigma = 1.0

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key
            for key in config
            if not key.startswith("_") and key not in DPMSOLVER_MULTISTEP_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported DPM-Solver setting {unknown[0]}")
        cfg = DPMSOLVER_MULTISTEP_DEFAULTS | {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        for key, expected in _REQUIRED.items():
            if cfg[key] != expected:
                raise ValueError(f"Unsupported DPM-Solver setting {key}={cfg[key]!r}")
        self.use_flow_sigmas = cfg["use_flow_sigmas"]
        if (cfg["prediction_type"], self.use_flow_sigmas) not in {
            ("epsilon", False),
            ("flow_prediction", True),
        }:
            raise ValueError(
                "Unsupported DPM-Solver setting "
                f"prediction_type={cfg['prediction_type']!r} with "
                f"use_flow_sigmas={self.use_flow_sigmas!r}"
            )
        if not self.use_flow_sigmas and cfg["flow_shift"] != 1.0:
            raise ValueError(f"Unsupported DPM-Solver setting flow_shift={cfg['flow_shift']!r}")
        if self.use_flow_sigmas and not float(cfg["flow_shift"]) > 0.0:
            raise ValueError(f"Unsupported DPM-Solver setting flow_shift={cfg['flow_shift']!r}")
        self.config = cfg
        count = cfg["num_train_timesteps"]
        self.betas = torch.linspace(cfg["beta_start"], cfg["beta_end"], count, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        self.sigmas = (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5).cpu()
        self.timesteps = torch.linspace(count - 1, 0, count, dtype=torch.float32)
        self.num_inference_steps: int | None = None
        self.model_outputs: list[Tensor | None] = [None] * cfg["solver_order"]
        self.lower_order_nums = 0
        self._step_index: int | None = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_timesteps(self, num_inference_steps: int) -> None:
        count = self.config["num_train_timesteps"]
        if type(num_inference_steps) is not int or not 0 < num_inference_steps <= count:
            raise ValueError(f"num_inference_steps must be an int in [1, {count}]")
        if self.use_flow_sigmas:
            # NumPy's linspace (float64): arange * step + start, with the endpoint pinned.
            stop = 1.0 / count
            step = (stop - 1.0) / num_inference_steps
            alphas = [index * step + 1.0 for index in range(num_inference_steps)] + [stop]
            shift = float(self.config["flow_shift"])
            base = [1.0 - alpha for alpha in alphas]
            shifted = [shift * sigma / (1 + (shift - 1) * sigma) for sigma in base]
            flow_sigmas = shifted[::-1][:-1]
            self.sigmas = torch.tensor(flow_sigmas + [0.0], dtype=torch.float32)
            self.timesteps = torch.tensor(
                [sigma * count for sigma in flow_sigmas], dtype=torch.float64
            ).to(torch.int64)
        else:
            timesteps = (
                torch.linspace(0, count - 1, num_inference_steps + 1, dtype=torch.float64)
                .round()
                .flip(0)[:-1]
                .to(torch.int64)
            )
            training_sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
            sigmas = training_sigmas[timesteps].tolist()
            self.sigmas = torch.tensor(sigmas + [0.0], dtype=torch.float32)
            self.timesteps = timesteps
        self.num_inference_steps = len(self.timesteps)
        self.model_outputs = [None] * self.config["solver_order"]
        self.lower_order_nums = 0
        self._step_index = None

    def index_for_timestep(self, timestep: int | Tensor) -> int:
        candidates = (self.timesteps == timestep).nonzero()
        if len(candidates) == 0:
            return len(self.timesteps) - 1
        if len(candidates) > 1:
            return int(candidates[1].item())
        return int(candidates[0].item())

    def _init_step_index(self, timestep: int | Tensor) -> None:
        if isinstance(timestep, Tensor):
            timestep = timestep.to(self.timesteps.device)
        self._step_index = self.index_for_timestep(timestep)

    def _sigma_to_alpha_sigma_t(self, sigma: Tensor) -> tuple[Tensor, Tensor]:
        if self.use_flow_sigmas:
            return 1 - sigma, sigma
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        sigma_t = sigma * alpha_t
        return alpha_t, sigma_t

    def _convert_model_output(self, model_output: Tensor, sample: Tensor) -> Tensor:
        assert self.step_index is not None
        sigma = self.sigmas[self.step_index]
        if self.config["prediction_type"] == "flow_prediction":
            return sample - sigma * model_output
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        return (sample - sigma_t * model_output) / alpha_t

    def _first_order_update(self, model_output: Tensor, sample: Tensor) -> Tensor:
        assert self.step_index is not None
        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
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
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        d0, d1 = m0, (1.0 / r0) * (m0 - m1)
        return (
            (sigma_t / sigma_s0) * sample
            - (alpha_t * (torch.exp(-h) - 1.0)) * d0
            - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * d1
        )

    def step(self, model_output: Tensor, timestep: int | Tensor, sample: Tensor) -> Tensor:
        if self.num_inference_steps is None:
            raise RuntimeError("Call set_timesteps before step")
        if self.step_index is None:
            self._init_step_index(timestep)
        assert self.step_index is not None
        lower_order_final = self.step_index == len(self.timesteps) - 1
        converted = self._convert_model_output(model_output, sample)
        self.model_outputs[0] = self.model_outputs[1]
        self.model_outputs[1] = converted
        sample = sample.to(torch.float32)
        if self.lower_order_nums < 1 or lower_order_final:
            previous = self._first_order_update(converted, sample)
        else:
            previous = self._second_order_update(sample)
        if self.lower_order_nums < self.config["solver_order"]:
            self.lower_order_nums += 1
        previous = previous.to(model_output.dtype)
        assert self._step_index is not None
        self._step_index += 1
        return previous

    @staticmethod
    def scale_model_input(sample: Tensor, timestep: int | Tensor | None = None) -> Tensor:
        return sample
