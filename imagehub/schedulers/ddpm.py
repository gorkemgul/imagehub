"""Denoising Diffusion Probabilistic Model scheduler for HunyuanDiT."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

DDPM_DEFAULTS: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "trained_betas": None,
    "variance_type": "fixed_small",
    "clip_sample": True,
    "prediction_type": "epsilon",
    "thresholding": False,
    "dynamic_thresholding_ratio": 0.995,
    "clip_sample_range": 1.0,
    "sample_max_value": 1.0,
    "timestep_spacing": "leading",
    "steps_offset": 0,
    "rescale_betas_zero_snr": False,
}
_IGNORED = frozenset({"set_alpha_to_one", "skip_prk_steps"})
_HUNYUAN_CONFIG = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.03,
    "beta_schedule": "scaled_linear",
    "trained_betas": None,
    "variance_type": "fixed_small",
    "clip_sample": False,
    "thresholding": False,
    "dynamic_thresholding_ratio": 0.995,
    "clip_sample_range": 1.0,
    "sample_max_value": 1.0,
    "timestep_spacing": "leading",
    "steps_offset": 1,
    "rescale_betas_zero_snr": False,
}


class DDPMScheduler:
    """Stateful DDPM sampler for the HunyuanDiT release configuration."""

    init_noise_sigma = 1.0

    def __init__(self, config: Mapping[str, Any]) -> None:
        supported = DDPM_DEFAULTS.keys() | _IGNORED
        unknown = sorted(key for key in config if not key.startswith("_") and key not in supported)
        if unknown:
            raise ValueError(f"Unsupported DDPM setting {unknown[0]}")
        cfg = DDPM_DEFAULTS | {
            key: value
            for key, value in config.items()
            if not key.startswith("_") and key not in _IGNORED
        }
        is_hunyuan = all(cfg[key] == expected for key, expected in _HUNYUAN_CONFIG.items())
        is_kandinsky = (
            cfg["beta_schedule"] == "linear"
            and cfg["clip_sample"] is True
            and cfg["prediction_type"] == "epsilon"
            and cfg["variance_type"] == "learned_range"
            and cfg["steps_offset"] == 0
        )
        if not is_hunyuan and not is_kandinsky:
            changed = next(
                (key for key, expected in _HUNYUAN_CONFIG.items() if cfg[key] != expected),
                "configuration",
            )
            raise ValueError(f"Unsupported DDPM setting {changed}={cfg.get(changed)!r}")
        if cfg["trained_betas"] is not None:
            raise ValueError("Unsupported DDPM setting trained_betas")
        if cfg["beta_schedule"] not in ("linear", "scaled_linear"):
            raise ValueError(f"Unsupported DDPM setting beta_schedule={cfg['beta_schedule']!r}")
        if cfg["variance_type"] not in ("fixed_small", "learned_range"):
            raise ValueError(f"Unsupported DDPM setting variance_type={cfg['variance_type']!r}")
        if cfg["thresholding"]:
            raise ValueError("Unsupported DDPM setting thresholding=True")
        if cfg["rescale_betas_zero_snr"]:
            raise ValueError("Unsupported DDPM setting rescale_betas_zero_snr=True")
        if cfg["timestep_spacing"] != "leading":
            raise ValueError(
                f"Unsupported DDPM setting timestep_spacing={cfg['timestep_spacing']!r}"
            )
        if cfg["prediction_type"] not in ("epsilon", "v_prediction"):
            raise ValueError(f"Unsupported DDPM setting prediction_type={cfg['prediction_type']!r}")
        self.config = cfg
        count = cfg["num_train_timesteps"]
        if cfg["beta_schedule"] == "scaled_linear":
            self.betas = (
                torch.linspace(
                    cfg["beta_start"] ** 0.5,
                    cfg["beta_end"] ** 0.5,
                    count,
                    dtype=torch.float32,
                )
                ** 2
            )
        else:
            self.betas = torch.linspace(
                cfg["beta_start"], cfg["beta_end"], count, dtype=torch.float32
            )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.timesteps = torch.arange(count - 1, -1, -1, dtype=torch.int64)
        self.num_inference_steps = 0

    def scale_model_input(self, sample: Tensor, timestep: Tensor | int | None = None) -> Tensor:
        return sample

    def set_timesteps(self, num_inference_steps: int) -> None:
        count = self.config["num_train_timesteps"]
        if type(num_inference_steps) is not int or not 0 < num_inference_steps <= count:
            raise ValueError(f"num_inference_steps must be an int in [1, {count}]")
        self.num_inference_steps = num_inference_steps
        step_ratio = count // num_inference_steps
        self.timesteps = (
            torch.arange(num_inference_steps - 1, -1, -1, dtype=torch.int64) * step_ratio
            + self.config["steps_offset"]
        )

    def previous_timestep(self, timestep: Tensor | int) -> Tensor:
        indices = (self.timesteps == torch.as_tensor(timestep)).nonzero(as_tuple=True)[0]
        if not len(indices):
            raise ValueError(f"timestep {timestep} is not in the current schedule")
        index = int(indices[0].item())
        if index == self.timesteps.shape[0] - 1:
            return torch.tensor(-1)
        return self.timesteps[index + 1]

    def _get_variance(
        self, timestep: Tensor | int, predicted_variance: Tensor | None = None
    ) -> Tensor:
        previous_timestep = self.previous_timestep(timestep)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[previous_timestep] if previous_timestep >= 0 else self.one
        )
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)
        if self.config["variance_type"] == "learned":
            assert predicted_variance is not None
            return predicted_variance
        if self.config["variance_type"] == "learned_range":
            assert predicted_variance is not None
            minimum = torch.log(variance)
            maximum = torch.log(current_beta_t)
            fraction = (predicted_variance + 1) / 2
            return fraction * maximum + (1 - fraction) * minimum
        return variance

    def step(
        self,
        model_output: Tensor,
        timestep: Tensor | int,
        sample: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if self.num_inference_steps == 0:
            raise ValueError("Call set_timesteps before step")
        t = timestep
        previous_timestep = self.previous_timestep(t)
        if model_output.shape[1] == sample.shape[1] * 2 and self.config["variance_type"] in (
            "learned",
            "learned_range",
        ):
            model_output, predicted_variance = model_output.split(sample.shape[1], dim=1)
        else:
            predicted_variance = None
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[previous_timestep] if previous_timestep >= 0 else self.one
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.config["prediction_type"] == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        else:
            pred_original_sample = alpha_prod_t**0.5 * sample - beta_prod_t**0.5 * model_output

        if self.config["clip_sample"]:
            limit = self.config["clip_sample_range"]
            pred_original_sample = pred_original_sample.clamp(-limit, limit)

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        )

        variance: Tensor | int = 0
        if t > 0:
            random_device = model_output.device
            if generator is not None and generator.device.type == "cpu":
                random_device = torch.device("cpu")
            variance_noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=random_device,
                dtype=model_output.dtype,
            ).to(model_output.device)
            variance_value = self._get_variance(t, predicted_variance)
            if self.config["variance_type"] == "learned_range":
                variance = torch.exp(0.5 * variance_value) * variance_noise
            else:
                variance = variance_value**0.5 * variance_noise
        return pred_prev_sample + variance
