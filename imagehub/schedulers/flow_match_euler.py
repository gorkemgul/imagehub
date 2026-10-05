"""Flow-match Euler scheduler for Z-Image-Turbo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

FLOW_MATCH_DEFAULTS: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": False,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
    "invert_sigmas": False,
    "shift_terminal": None,
    "use_karras_sigmas": False,
    "use_exponential_sigmas": False,
    "use_beta_sigmas": False,
    "time_shift_type": "exponential",
    "stochastic_sampling": False,
}
_REQUIRED = {
    "use_dynamic_shifting": False,
    "invert_sigmas": False,
    "shift_terminal": None,
    "use_karras_sigmas": False,
    "use_exponential_sigmas": False,
    "use_beta_sigmas": False,
    "time_shift_type": "exponential",
    "stochastic_sampling": False,
}


def z_image_sigmas(steps: int) -> list[float]:
    """Return the float32 sigma schedule supplied by the Z-Image pipeline."""
    return torch.linspace(1.0, 1 / steps, steps).tolist()


class FlowMatchEulerDiscreteScheduler:
    """Stateful deterministic flow-match Euler scheduler."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in FLOW_MATCH_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported FlowMatch Euler setting {unknown[0]}")
        cfg = FLOW_MATCH_DEFAULTS | {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        for key, expected in _REQUIRED.items():
            if cfg[key] != expected:
                raise ValueError(f"Unsupported FlowMatch Euler setting {key}={cfg[key]!r}")

        self.config = cfg
        self.shift = float(cfg["shift"])
        count = int(cfg["num_train_timesteps"])
        sigmas = torch.arange(count, 0, -1, dtype=torch.float32) / count
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = sigmas * count
        self.sigmas = sigmas
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self.num_inference_steps = 0
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def set_timesteps(
        self,
        steps: int,
        *,
        sigmas: Sequence[float] | None = None,
        mu: float | None = None,
    ) -> None:
        del mu
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive int")
        if sigmas is not None and len(sigmas) != steps:
            raise ValueError("sigmas must have the same length as steps")

        self.num_inference_steps = steps
        if sigmas is None:
            start = self.sigma_max * self.config["num_train_timesteps"]
            stop = self.sigma_min * self.config["num_train_timesteps"]
            if steps == 1:
                values = [start / self.config["num_train_timesteps"]]
            else:
                delta = (stop - start) / (steps - 1)
                timesteps = [start + index * delta for index in range(steps)]
                timesteps[-1] = stop
                values = [timestep / self.config["num_train_timesteps"] for timestep in timesteps]
            shifted = [self.shift * sigma / (1 + (self.shift - 1) * sigma) for sigma in values]
            schedule = torch.tensor(shifted, dtype=torch.float32)
        else:
            schedule = torch.tensor(sigmas, dtype=torch.float32)
            schedule = self.shift * schedule / (1 + (self.shift - 1) * schedule)

        self.timesteps = schedule * self.config["num_train_timesteps"]
        self.sigmas = torch.cat([schedule, torch.zeros(1, dtype=torch.float32)])
        self._step_index = None
        self._begin_index = None

    def _init_step_index(self, timestep: Tensor | float) -> None:
        if self._begin_index is not None:
            self._step_index = self._begin_index
            return
        indices = (self.timesteps == torch.as_tensor(timestep)).nonzero()
        if not len(indices):
            raise ValueError(f"timestep {timestep} is not in the current schedule")
        position = 1 if len(indices) > 1 else 0
        self._step_index = int(indices[position].item())

    def step(self, model_output: Tensor, timestep: Tensor | float, sample: Tensor) -> Tensor:
        if self.num_inference_steps == 0:
            raise RuntimeError("Call set_timesteps before step")
        if isinstance(timestep, int) or (
            isinstance(timestep, Tensor) and timestep.dtype in (torch.int32, torch.int64)
        ):
            raise ValueError("timestep must be one of the scheduler's floating-point timesteps")
        if self._step_index is None:
            self._init_step_index(timestep)
        assert self._step_index is not None
        sample = sample.to(torch.float32)
        delta = self.sigmas[self._step_index + 1] - self.sigmas[self._step_index]
        previous = sample + delta * model_output
        self._step_index += 1
        return previous.to(model_output.dtype)
