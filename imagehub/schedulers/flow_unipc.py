"""Flow-matching UniPC scheduler used by HiDream-O1."""

from __future__ import annotations

import math

import torch
from torch import Tensor


class FlowUniPCMultistepScheduler:
    """Second-order UniPC predictor-corrector for flow predictions."""

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        shift: float = 1.0,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: tuple[int, ...] = (),
    ) -> None:
        if solver_order < 1:
            raise ValueError("solver_order must be positive")
        if solver_type not in ("bh1", "bh2"):
            raise ValueError("solver_type must be 'bh1' or 'bh2'")
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.shift = shift
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.disable_corrector = disable_corrector
        base = torch.linspace(1.0 - 1.0 / num_train_timesteps, 0.0, num_train_timesteps)
        self.sigmas = shift * base / (1 + (shift - 1) * base)
        self.timesteps = self.sigmas * num_train_timesteps
        self.num_inference_steps: int | None = None
        self.model_outputs: list[Tensor | None] = [None] * solver_order
        self.timestep_list: list[Tensor | None] = [None] * solver_order
        self.lower_order_nums = 0
        self.last_sample: Tensor | None = None
        self.step_index: int | None = None
        self.this_order = 1

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        device: str | torch.device | None = None,
        timesteps: list[int] | None = None,
    ) -> None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if timesteps is None:
            start, stop = float(self.sigmas[0]), float(self.sigmas[-1])
            raw = torch.tensor(
                [
                    start + (stop - start) * index / num_inference_steps
                    for index in range(num_inference_steps)
                ],
                dtype=torch.float64,
            )
            sigmas = (self.shift * raw / (1 + (self.shift - 1) * raw)).to(torch.float32)
            selected = (sigmas * self.num_train_timesteps).to(torch.int64)
        else:
            if not timesteps:
                raise ValueError("timesteps must not be empty")
            selected = torch.tensor(timesteps, dtype=torch.int64)
            sigmas = selected.to(torch.float32) / self.num_train_timesteps
            num_inference_steps = len(timesteps)
        self.timesteps = selected.to(device=device)
        self.sigmas = torch.cat((sigmas.to(torch.float32), torch.zeros(1)))
        self.num_inference_steps = num_inference_steps
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.step_index = None

    @staticmethod
    def _alpha_sigma(sigma: Tensor) -> tuple[Tensor, Tensor]:
        return 1 - sigma, sigma

    def _predictor(self, sample: Tensor, order: int) -> Tensor:
        assert self.step_index is not None
        m0 = self.model_outputs[-1]
        assert m0 is not None
        sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._alpha_sigma(sigma_t)
        _, sigma_s0 = self._alpha_sigma(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(1 - sigma_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        hh = -h
        h_phi_1 = torch.expm1(hh)
        b_h = hh if self.solver_type == "bh1" else torch.expm1(hh)
        result = sigma_t / sigma_s0 * sample - alpha_t * h_phi_1 * m0
        if order > 1:
            previous = self.model_outputs[-2]
            assert previous is not None
            sigma_si = self.sigmas[self.step_index - 1]
            lambda_si = torch.log(1 - sigma_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            d1 = (previous - m0) / rk
            result = result - alpha_t * b_h * 0.5 * d1
        return result.to(sample.dtype)

    def _corrector(self, model_output: Tensor, last_sample: Tensor, sample: Tensor) -> Tensor:
        assert self.step_index is not None
        m0 = self.model_outputs[-1]
        assert m0 is not None
        sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
        alpha_t, sigma_t = self._alpha_sigma(sigma_t)
        _, sigma_s0 = self._alpha_sigma(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(1 - sigma_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        hh = -h
        h_phi_1 = torch.expm1(hh)
        b_h = hh if self.solver_type == "bh1" else torch.expm1(hh)
        converted = sample - sigma_t * model_output
        base = sigma_t / sigma_s0 * last_sample - alpha_t * h_phi_1 * m0
        if self.this_order == 1:
            correction = 0.5 * (converted - m0)
        else:
            previous = self.model_outputs[-2]
            assert previous is not None
            sigma_si = self.sigmas[self.step_index - 2]
            lambda_si = torch.log(1 - sigma_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            d1 = (previous - m0) / rk
            rks = torch.stack((rk, torch.ones_like(rk)))
            matrix = torch.stack((torch.ones_like(rks), rks))
            phi2 = h_phi_1 / hh - 1
            phi3 = phi2 / hh - 0.5
            rhs = torch.stack((phi2 / b_h, 2 * phi3 / b_h))
            rho = torch.linalg.solve(matrix, rhs).to(sample.dtype)
            correction = rho[0] * d1 + rho[1] * (converted - m0)
        return (base - alpha_t * b_h * correction).to(sample.dtype)

    def step(self, model_output: Tensor, timestep: Tensor, sample: Tensor) -> Tensor:
        if self.num_inference_steps is None:
            raise ValueError("set_timesteps must be called before step")
        if self.step_index is None:
            matches = (self.timesteps == timestep.to(self.timesteps.device)).nonzero()
            self.step_index = int(matches[0].item())
        index = self.step_index
        sigma = self.sigmas[index].to(sample.device)
        converted = sample - sigma * model_output
        if index > 0 and index - 1 not in self.disable_corrector and self.last_sample is not None:
            sample = self._corrector(model_output, self.last_sample, sample)
        self.model_outputs = self.model_outputs[1:] + [converted]
        self.timestep_list = self.timestep_list[1:] + [timestep]
        remaining = len(self.timesteps) - index
        order = min(self.solver_order, remaining) if self.lower_order_final else self.solver_order
        self.this_order = min(order, self.lower_order_nums + 1)
        self.last_sample = sample
        result = self._predictor(sample, self.this_order)
        self.lower_order_nums = min(self.lower_order_nums + 1, self.solver_order)
        self.step_index += 1
        return result

    @staticmethod
    def time_shift(mu: float, sigma: float, timestep: Tensor) -> Tensor:
        factor = math.exp(mu)
        return factor / (factor + (1 / timestep - 1) ** sigma)
