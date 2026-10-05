"""PNDM scheduler in PLMS-only mode (``skip_prk_steps=True``), as configured by SD 1.x releases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

PNDM_DEFAULTS: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "set_alpha_to_one": False,
    "skip_prk_steps": False,
    "steps_offset": 0,
    "prediction_type": "epsilon",
    "timestep_spacing": "leading",
    "trained_betas": None,
}
_REQUIRED = {
    "skip_prk_steps": True,
    "prediction_type": "epsilon",
    "timestep_spacing": "leading",
    "trained_betas": None,
}


class PNDMScheduler:
    """Stateful sampler. Create one per trajectory and call :meth:`set_timesteps` first."""

    init_noise_sigma = 1.0

    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = PNDM_DEFAULTS | {k: v for k, v in config.items() if not k.startswith("_")}
        for key, expected in _REQUIRED.items():
            if cfg[key] != expected:
                raise ValueError(f"Unsupported PNDM setting {key}={cfg[key]!r}")
        n = cfg["num_train_timesteps"]
        if cfg["beta_schedule"] == "scaled_linear":
            betas = (
                torch.linspace(
                    cfg["beta_start"] ** 0.5, cfg["beta_end"] ** 0.5, n, dtype=torch.float32
                )
                ** 2
            )
        elif cfg["beta_schedule"] == "linear":
            betas = torch.linspace(cfg["beta_start"], cfg["beta_end"], n, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported beta_schedule {cfg['beta_schedule']!r}")
        self.config = cfg
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if cfg["set_alpha_to_one"] else self.alphas_cumprod[0]
        )
        self.timesteps = torch.empty(0, dtype=torch.int64)
        self.num_inference_steps = 0
        self._ets: list[Tensor] = []
        self._cur_sample: Tensor | None = None
        self._counter = 0

    def set_timesteps(self, num_inference_steps: int) -> None:
        n = self.config["num_train_timesteps"]
        if type(num_inference_steps) is not int or not 0 < num_inference_steps <= n:
            raise ValueError(f"num_inference_steps must be an int in [1, {n}]")
        self.num_inference_steps = num_inference_steps
        ratio = n // num_inference_steps
        base = [i * ratio + self.config["steps_offset"] for i in range(num_inference_steps)]
        # PLMS repeats the second-to-last timestep: num_inference_steps + 1 model evaluations.
        plms = base[:-1] + base[-2:-1] + base[-1:]
        self.timesteps = torch.tensor(plms[::-1], dtype=torch.int64)
        self._ets, self._cur_sample, self._counter = [], None, 0

    def step(self, model_output: Tensor, timestep: Tensor | int, sample: Tensor) -> Tensor:
        if self.num_inference_steps == 0:
            raise RuntimeError("Call set_timesteps before step")
        t = torch.as_tensor(timestep)
        ratio = self.config["num_train_timesteps"] // self.num_inference_steps
        prev_t = t - ratio
        if self._counter != 1:
            self._ets = self._ets[-3:]
            self._ets.append(model_output)
        else:
            prev_t = t
            t = t + ratio
        ets = self._ets
        if len(ets) == 1 and self._counter == 0:
            self._cur_sample = sample
        elif len(ets) == 1 and self._counter == 1:
            model_output = (model_output + ets[-1]) / 2
            assert self._cur_sample is not None
            sample, self._cur_sample = self._cur_sample, None
        elif len(ets) == 2:
            model_output = (3 * ets[-1] - ets[-2]) / 2
        elif len(ets) == 3:
            model_output = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12
        else:
            model_output = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
        self._counter += 1
        return self._previous(sample, t, prev_t, model_output)

    def _previous(self, sample: Tensor, t: Tensor, prev_t: Tensor, eps: Tensor) -> Tensor:
        """PNDM paper eq. (9), evaluated in the same order as diffusers."""
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        beta_t, beta_prev = 1 - alpha_t, 1 - alpha_prev
        sample_coeff = (alpha_prev / alpha_t) ** 0.5
        denom = alpha_t * beta_prev**0.5 + (alpha_t * beta_t * alpha_prev) ** 0.5
        return sample_coeff * sample - (alpha_prev - alpha_t) * eps / denom
