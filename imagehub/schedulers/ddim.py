"""Deterministic DDIM sampler (``eta=0``) for epsilon, sample and v-prediction models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor

DDIM_DEFAULTS: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "trained_betas": None,
    "clip_sample": True,
    "set_alpha_to_one": True,
    "steps_offset": 0,
    "prediction_type": "epsilon",
    "thresholding": False,
    "dynamic_thresholding_ratio": 0.995,
    "clip_sample_range": 1.0,
    "sample_max_value": 1.0,
    "timestep_spacing": "leading",
    "rescale_betas_zero_snr": False,
}
_REQUIRED = {"thresholding": False, "trained_betas": None, "rescale_betas_zero_snr": False}
_PREDICTION_TYPES = ("epsilon", "sample", "v_prediction")
_SPACINGS = ("leading", "linspace", "trailing")
# Keys that older releases write but the sampler never reads.
_LEGACY_KEYS = {"skip_prk_steps"}


class DDIMScheduler:
    """Stateless apart from the timestep table; call :meth:`set_timesteps` before :meth:`step`."""

    init_noise_sigma = 1.0

    def __init__(self, config: Mapping[str, Any]) -> None:
        extra = {k: v for k, v in config.items() if not k.startswith("_")}
        unknown = sorted(set(extra) - set(DDIM_DEFAULTS) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(f"Unsupported DDIM setting {unknown[0]}")
        cfg = DDIM_DEFAULTS | {k: v for k, v in extra.items() if k in DDIM_DEFAULTS}
        for key, expected in _REQUIRED.items():
            if cfg[key] != expected:
                raise ValueError(f"Unsupported DDIM setting {key}={cfg[key]!r}")
        if cfg["prediction_type"] not in _PREDICTION_TYPES:
            raise ValueError(f"Unsupported DDIM setting prediction_type={cfg['prediction_type']!r}")
        if cfg["timestep_spacing"] not in _SPACINGS:
            raise ValueError(
                f"Unsupported DDIM setting timestep_spacing={cfg['timestep_spacing']!r}"
            )
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

    def set_timesteps(self, num_inference_steps: int) -> None:
        n = self.config["num_train_timesteps"]
        if type(num_inference_steps) is not int or not 0 < num_inference_steps <= n:
            raise ValueError(f"num_inference_steps must be an int in [1, {n}]")
        self.num_inference_steps = num_inference_steps
        spacing = self.config["timestep_spacing"]
        if spacing == "linspace":
            steps = np.linspace(0, n - 1, num_inference_steps).round()[::-1].copy()
            timesteps = steps.astype(np.int64)
        elif spacing == "leading":
            ratio = n // num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * ratio).round()[::-1].copy()
            timesteps = timesteps.astype(np.int64) + self.config["steps_offset"]
        else:
            ratio_f = n / num_inference_steps
            timesteps = np.round(np.arange(n, 0, -ratio_f)).astype(np.int64) - 1
        self.timesteps = torch.from_numpy(timesteps)

    def scale_model_input(self, sample: Tensor, timestep: Tensor | int | None = None) -> Tensor:
        return sample

    def step(self, model_output: Tensor, timestep: Tensor | int, sample: Tensor) -> Tensor:
        if self.num_inference_steps == 0:
            raise RuntimeError("Call set_timesteps before step")
        cfg = self.config
        prev_timestep = timestep - cfg["num_train_timesteps"] // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        if cfg["prediction_type"] == "epsilon":
            pred_original = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif cfg["prediction_type"] == "sample":
            pred_original = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original) / beta_prod_t ** (0.5)
        else:
            pred_original = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        if cfg["clip_sample"]:
            pred_original = pred_original.clamp(-cfg["clip_sample_range"], cfg["clip_sample_range"])
        # eta = 0: the variance term is zero, so only the deterministic direction remains.
        direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        return alpha_prod_t_prev ** (0.5) * pred_original + direction
