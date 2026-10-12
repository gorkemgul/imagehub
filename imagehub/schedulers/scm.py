"""SCM scheduler for Sana-Sprint (trigflow parameterization)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

SCM_DEFAULTS: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "prediction_type": "trigflow",
    "sigma_data": 0.5,
}


class SCMScheduler:
    """Single-step consistent model scheduler used by Sana-Sprint.

    Only the published ``trigflow`` configuration is supported.
    """

    init_noise_sigma = 1.0
    order = 1

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in SCM_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported SCM scheduler setting {unknown[0]}")
        cfg = SCM_DEFAULTS | {k: v for k, v in config.items() if not k.startswith("_")}
        if cfg["prediction_type"] != "trigflow":
            raise ValueError(
                f"Unsupported SCM scheduler setting prediction_type={cfg['prediction_type']!r}"
            )
        if not isinstance(cfg["sigma_data"], int | float) or cfg["sigma_data"] <= 0:
            raise ValueError(f"Unsupported SCM scheduler setting sigma_data={cfg['sigma_data']!r}")
        self.config = cfg
        self.num_inference_steps: int | None = None
        self.timesteps = torch.arange(cfg["num_train_timesteps"] - 1, -1, -1, dtype=torch.int64)
        self._step_index: int | None = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        max_timesteps: float = 1.57080,
        intermediate_timesteps: float | None = None,
    ) -> None:
        count = self.config["num_train_timesteps"]
        if type(num_inference_steps) is not int or not 0 < num_inference_steps <= count:
            raise ValueError(f"num_inference_steps must be an int in [1, {count}]")
        self.num_inference_steps = num_inference_steps
        if intermediate_timesteps is not None and num_inference_steps != 2:
            raise ValueError(
                "Intermediate timesteps for SCM is not supported when num_inference_steps != 2"
            )
        if intermediate_timesteps is not None:
            self.timesteps = torch.tensor(
                [max_timesteps, intermediate_timesteps, 0.0], dtype=torch.float32
            )
        else:
            self.timesteps = torch.linspace(
                max_timesteps, 0, num_inference_steps + 1, dtype=torch.float32
            )
        self._step_index = None

    def _index_for_timestep(self, timestep: float | Tensor) -> int:
        if isinstance(timestep, Tensor):
            timestep = timestep.item()
        matches = (self.timesteps == timestep).nonzero(as_tuple=False)
        pos = 1 if len(matches) > 1 else 0
        return int(matches[pos].item())

    def _init_step_index(self, timestep: float | Tensor) -> None:
        self._step_index = self._index_for_timestep(timestep)

    def step(
        self,
        model_output: Tensor,
        timestep: float | Tensor,
        sample: Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.num_inference_steps is None:
            raise RuntimeError("Call set_timesteps before step")
        if self._step_index is None:
            self._init_step_index(timestep)
        assert self._step_index is not None
        s = self.timesteps[self._step_index]
        t = self.timesteps[self._step_index + 1]
        if self.config["prediction_type"] == "trigflow":
            pred_x0 = torch.cos(s) * sample - torch.sin(s) * model_output
        else:
            raise ValueError(f"Unsupported SCM prediction_type {self.config['prediction_type']!r}")
        if len(self.timesteps) > 1:
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                dtype=model_output.dtype,
                device=model_output.device,
            )
            noise = noise * self.config["sigma_data"]
            prev_sample = torch.cos(t) * pred_x0 + torch.sin(t) * noise
        else:
            prev_sample = pred_x0
        self._step_index += 1
        return prev_sample, pred_x0

    def __len__(self) -> int:
        return self.config["num_train_timesteps"]

    @staticmethod
    def scale_model_input(sample: Tensor, timestep: float | Tensor | None = None) -> Tensor:
        return sample
