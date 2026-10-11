"""Deterministic DDIM sampler with shifted SNR (the CogVideoX variant), ``eta = 0`` only."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

COGVIDEOX_DDIM_DEFAULTS: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.0120,
    "beta_schedule": "scaled_linear",
    "trained_betas": None,
    "clip_sample": True,
    "set_alpha_to_one": True,
    "steps_offset": 0,
    "prediction_type": "epsilon",
    "clip_sample_range": 1.0,
    "sample_max_value": 1.0,
    "timestep_spacing": "leading",
    "rescale_betas_zero_snr": False,
    "snr_shift_scale": 3.0,
}
# ``clip_sample``, ``clip_sample_range`` and ``sample_max_value`` are accepted because releases
# write them, but the reference ``step`` never reads them.
_PREDICTION_TYPES = ("epsilon", "sample", "v_prediction")
_SPACINGS = ("leading", "linspace", "trailing")
_SCHEDULES = ("linear", "scaled_linear", "squaredcos_cap_v2")


def betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999) -> Tensor:
    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def rescale_zero_terminal_snr(alphas_cumprod: Tensor) -> Tensor:
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_t = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= alphas_bar_sqrt_t
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_t)
    return alphas_bar_sqrt**2


def _linspace(stop: int, num: int) -> list[float]:
    """``numpy.linspace(0, stop, num)`` in float64, including the exact endpoint."""
    if num == 1:
        return [0.0]
    step = stop / (num - 1)
    values = [i * step for i in range(num)]
    values[-1] = float(stop)
    return values


class CogVideoXDDIMScheduler:
    """Call :meth:`set_timesteps` before :meth:`step`."""

    init_noise_sigma = 1.0

    def __init__(self, config: Mapping[str, Any]) -> None:
        extra = {k: v for k, v in config.items() if not k.startswith("_")}
        unknown = sorted(set(extra) - set(COGVIDEOX_DDIM_DEFAULTS))
        if unknown:
            raise ValueError(f"Unsupported CogVideoX DDIM setting {unknown[0]}")
        cfg = COGVIDEOX_DDIM_DEFAULTS | extra
        if cfg["trained_betas"] is not None:
            raise ValueError("Unsupported CogVideoX DDIM setting trained_betas")
        if cfg["prediction_type"] not in _PREDICTION_TYPES:
            raise ValueError(
                f"Unsupported CogVideoX DDIM setting prediction_type={cfg['prediction_type']!r}"
            )
        if cfg["timestep_spacing"] not in _SPACINGS:
            raise ValueError(
                f"Unsupported CogVideoX DDIM setting timestep_spacing={cfg['timestep_spacing']!r}"
            )
        if cfg["beta_schedule"] not in _SCHEDULES:
            raise ValueError(f"Unsupported beta_schedule {cfg['beta_schedule']!r}")
        n = cfg["num_train_timesteps"]
        if cfg["beta_schedule"] == "linear":
            betas = torch.linspace(cfg["beta_start"], cfg["beta_end"], n, dtype=torch.float32)
        elif cfg["beta_schedule"] == "scaled_linear":
            betas = (
                torch.linspace(
                    cfg["beta_start"] ** 0.5, cfg["beta_end"] ** 0.5, n, dtype=torch.float64
                )
                ** 2
            )
        else:
            betas = betas_for_alpha_bar(n)
        self.config = cfg
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        shift = cfg["snr_shift_scale"]
        alphas_cumprod = alphas_cumprod / (shift + (1 - shift) * alphas_cumprod)
        if cfg["rescale_betas_zero_snr"]:
            alphas_cumprod = rescale_zero_terminal_snr(alphas_cumprod)
        self.alphas_cumprod = alphas_cumprod
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if cfg["set_alpha_to_one"] else self.alphas_cumprod[0]
        )
        self.timesteps = torch.arange(n - 1, -1, -1, dtype=torch.int64)
        self.num_inference_steps = 0

    def set_timesteps(self, num_inference_steps: int) -> None:
        n = self.config["num_train_timesteps"]
        if type(num_inference_steps) is not int or not 0 < num_inference_steps <= n:
            raise ValueError(f"num_inference_steps must be an int in [1, {n}]")
        self.num_inference_steps = num_inference_steps
        spacing = self.config["timestep_spacing"]
        if spacing == "linspace":
            values = _linspace(n - 1, num_inference_steps)
            timesteps = [int(round(v)) for v in reversed(values)]
        elif spacing == "leading":
            ratio = n // num_inference_steps
            offset = self.config["steps_offset"]
            timesteps = [i * ratio + offset for i in reversed(range(num_inference_steps))]
        else:
            ratio_f = n / num_inference_steps
            # numpy.arange derives its stride from the first two elements, not from the raw step.
            delta = (n - ratio_f) - n
            count = math.ceil(-n / -ratio_f)
            timesteps = [int(round(n + i * delta)) - 1 for i in range(count)]
        self.timesteps = torch.tensor(timesteps, dtype=torch.int64)

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
        elif cfg["prediction_type"] == "sample":
            pred_original = model_output
        else:
            pred_original = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        a_t = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5
        b_t = alpha_prod_t_prev**0.5 - alpha_prod_t**0.5 * a_t
        return a_t * sample + b_t * pred_original
