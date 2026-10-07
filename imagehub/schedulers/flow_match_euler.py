"""Flow-match Euler scheduler for Z-Image-Turbo."""

from __future__ import annotations

import math
import struct
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
    "invert_sigmas": False,
    "use_karras_sigmas": False,
    "use_exponential_sigmas": False,
    "use_beta_sigmas": False,
    "stochastic_sampling": False,
}
_TIME_SHIFT_TYPES = ("exponential", "linear")


def z_image_sigmas(steps: int) -> list[float]:
    """Return the float32 sigma schedule supplied by the Z-Image pipeline."""
    return torch.linspace(1.0, 1 / steps, steps).tolist()


def qwen_image_mu(height: int, width: int, config: Mapping[str, Any]) -> float:
    """Return the Qwen-Image resolution-dependent scheduler shift."""
    image_seq_len = (height // 16) * (width // 16)
    base_seq_len = int(config.get("base_image_seq_len", 256))
    max_seq_len = int(config.get("max_image_seq_len", 4096))
    base_shift = float(config.get("base_shift", 0.5))
    max_shift = float(config.get("max_shift", 1.15))
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


def _float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _time_shift(
    values: list[float], mu: float, *, float32: bool, kind: str = "exponential"
) -> list[float]:
    """The reference's ``_time_shift_exponential`` / ``_time_shift_linear`` with sigma 1.0."""
    exp_mu = math.exp(mu) if kind == "exponential" else mu
    if not float32:
        return [exp_mu / (exp_mu + (1 / value - 1)) for value in values]
    exp_mu = _float32(exp_mu)
    shifted = []
    for value in values:
        value = _float32(value)
        reciprocal = _float32(1 / value)
        denominator = _float32(exp_mu + _float32(reciprocal - 1))
        shifted.append(_float32(exp_mu / denominator))
    return shifted


def _stretch_to_terminal(values: list[float], terminal: float, *, float32: bool) -> list[float]:
    if not float32:
        one_minus = [1 - value for value in values]
        scale_factor = one_minus[-1] / (1 - terminal)
        if scale_factor == 0:
            return [math.nan for _ in values]
        return [1 - value / scale_factor for value in one_minus]
    one_minus = [_float32(1 - value) for value in values]
    scale_factor = _float32(one_minus[-1] / _float32(1 - terminal))
    if scale_factor == 0:
        return [math.nan for _ in values]
    return [_float32(1 - _float32(value / scale_factor)) for value in one_minus]


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
        if cfg["time_shift_type"] not in _TIME_SHIFT_TYPES:
            raise ValueError(
                f"Unsupported FlowMatch Euler setting time_shift_type={cfg['time_shift_type']!r}"
            )

        self.config = cfg
        self.shift = float(cfg["shift"])
        count = int(cfg["num_train_timesteps"])
        sigmas = torch.arange(count, 0, -1, dtype=torch.float32) / count
        if not cfg["use_dynamic_shifting"]:
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
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive int")
        if sigmas is not None and len(sigmas) != steps:
            raise ValueError("sigmas must have the same length as steps")
        if self.config["use_dynamic_shifting"] and mu is None:
            raise ValueError("mu must be passed when use_dynamic_shifting is True")

        self.num_inference_steps = steps
        values: list[float]
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
            if self.config["use_dynamic_shifting"]:
                assert mu is not None
                values = _time_shift(values, mu, float32=False, kind=self.config["time_shift_type"])
            else:
                shifted = [self.shift * sigma / (1 + (self.shift - 1) * sigma) for sigma in values]
                schedule = torch.tensor(shifted, dtype=torch.float32)
        else:
            if self.config["use_dynamic_shifting"]:
                assert mu is not None
                values = _time_shift(
                    [float(value) for value in sigmas],
                    mu,
                    float32=True,
                    kind=self.config["time_shift_type"],
                )
            else:
                schedule = torch.tensor(sigmas, dtype=torch.float32)

        if self.config["use_dynamic_shifting"]:
            shift_terminal = self.config["shift_terminal"]
            if shift_terminal:
                values = _stretch_to_terminal(
                    values,
                    float(shift_terminal),
                    float32=sigmas is not None,
                )
            schedule = torch.tensor(values, dtype=torch.float32)
        elif sigmas is not None:
            schedule = self.shift * schedule / (1 + (self.shift - 1) * schedule)

        shift_terminal = self.config["shift_terminal"]
        if shift_terminal and not self.config["use_dynamic_shifting"]:
            one_minus_schedule = 1 - schedule
            scale_factor = one_minus_schedule[-1] / (1 - shift_terminal)
            schedule = 1 - one_minus_schedule / scale_factor

        schedule = schedule.to(torch.float32)

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
