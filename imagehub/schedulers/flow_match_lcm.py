"""Flow-matching LCM scheduler with the stochastic re-noising step used by HiDream-I1-Fast."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

FLOW_MATCH_LCM_DEFAULTS: dict[str, Any] = {
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
    "scale_factors": None,
    "upscale_mode": "bicubic",
}
# Options the native scheduler does not implement; only these values are accepted.
_REQUIRED = {
    "use_dynamic_shifting": False,
    "invert_sigmas": False,
    "shift_terminal": None,
    "use_karras_sigmas": False,
    "use_exponential_sigmas": False,
    "use_beta_sigmas": False,
    "scale_factors": None,
    "upscale_mode": "bicubic",
}
_TIME_SHIFT_TYPES = ("exponential", "linear")


def _randn(
    shape: tuple[int, ...],
    generator: torch.Generator | Sequence[torch.Generator] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Gaussian noise drawn the way the reference ``randn_tensor`` draws it."""
    rand_device = device
    if generator is not None:
        first = generator[0] if isinstance(generator, Sequence) else generator
        if first.device.type == "cpu" and device.type != "cpu":
            rand_device = torch.device("cpu")
    if isinstance(generator, Sequence) and len(generator) == 1:
        generator = generator[0]
    if isinstance(generator, Sequence):
        if len(generator) != shape[0]:
            raise ValueError("The generator list must have one generator per batch element")
        rows = [
            torch.randn((1, *shape[1:]), generator=item, device=rand_device, dtype=dtype)
            for item in generator
        ]
        return torch.cat(rows, dim=0).to(device)
    return torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)


class FlowMatchLCMScheduler:
    """Stateful flow-match LCM scheduler: predict x0, then re-noise to the next sigma."""

    order = 1

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in FLOW_MATCH_LCM_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported FlowMatch LCM setting {unknown[0]}")
        cfg = FLOW_MATCH_LCM_DEFAULTS | {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        for key, expected in _REQUIRED.items():
            if cfg[key] != expected:
                raise ValueError(f"Unsupported FlowMatch LCM setting {key}={cfg[key]!r}")
        if cfg["time_shift_type"] not in _TIME_SHIFT_TYPES:
            raise ValueError(
                f"Unsupported FlowMatch LCM setting time_shift_type={cfg['time_shift_type']!r}"
            )

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

    def __len__(self) -> int:
        return int(self.config["num_train_timesteps"])

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
        """``mu`` is accepted for pipeline compatibility; the static shift ignores it."""
        del mu
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive int")
        if sigmas is not None and len(sigmas) != steps:
            raise ValueError("sigmas must have the same length as steps")

        count = self.config["num_train_timesteps"]
        if sigmas is None:
            # numpy.linspace in float64: start + index * step, with the stop value pinned.
            start = self.sigma_max * count
            stop = self.sigma_min * count
            if steps == 1:
                timesteps = [start]
            else:
                delta = (stop - start) / (steps - 1)
                timesteps = [index * delta + start for index in range(steps)]
                timesteps[-1] = stop
            values = [timestep / count for timestep in timesteps]
            shifted = [self.shift * sigma / (1 + (self.shift - 1) * sigma) for sigma in values]
            schedule = torch.tensor(shifted, dtype=torch.float32)
        else:
            schedule = torch.tensor(list(sigmas), dtype=torch.float32)
            schedule = self.shift * schedule / (1 + (self.shift - 1) * schedule)

        self.num_inference_steps = steps
        self.timesteps = schedule * count
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

    def step(
        self,
        model_output: Tensor,
        timestep: Tensor | float,
        sample: Tensor,
        generator: torch.Generator | Sequence[torch.Generator] | None = None,
    ) -> Tensor:
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
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        x0_pred = sample - sigma * model_output
        noise = _randn(tuple(x0_pred.shape), generator, x0_pred.device, x0_pred.dtype)
        previous = (1 - sigma_next) * x0_pred + sigma_next * noise
        self._step_index += 1
        return previous.to(model_output.dtype)
