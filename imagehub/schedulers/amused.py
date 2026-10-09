"""Masked-token scheduler used by aMUSEd: confidence-based re-masking on a cosine schedule."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
from torch import Tensor

AMUSED_DEFAULTS: dict[str, Any] = {"masking_schedule": "cosine"}


class AmusedSchedulerOutput(NamedTuple):
    prev_sample: Tensor
    pred_original_sample: Tensor


def gumbel_noise(t: Tensor, generator: torch.Generator | None = None) -> Tensor:
    device = generator.device if generator is not None else t.device
    noise = torch.zeros_like(t, device=device).uniform_(0, 1, generator=generator).to(t.device)
    return -torch.log((-torch.log(noise.clamp(1e-20))).clamp(1e-20))


def mask_by_random_topk(
    mask_len: Tensor,
    probs: Tensor,
    temperature: Tensor | float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """True for the ``mask_len`` tokens with the lowest temperature-perturbed confidence."""
    confidence = torch.log(probs.clamp(1e-20)) + temperature * gumbel_noise(
        probs, generator=generator
    )
    sorted_confidence = torch.sort(confidence, dim=-1).values
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    return confidence < cut_off


class AmusedScheduler:
    order = 1

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key
            for key in config
            if not key.startswith("_") and key not in AMUSED_DEFAULTS | {"mask_token_id": 0}
        )
        if unknown:
            raise ValueError(f"Unsupported aMUSEd scheduler setting {unknown[0]}")
        if "mask_token_id" not in config:
            raise ValueError("The aMUSEd scheduler config needs mask_token_id")
        self.config = AMUSED_DEFAULTS | {k: v for k, v in config.items() if not k.startswith("_")}
        if self.config["masking_schedule"] not in ("cosine", "linear"):
            raise ValueError(
                f"Unsupported aMUSEd scheduler setting "
                f"masking_schedule={self.config['masking_schedule']!r}"
            )
        self.mask_token_id: int = self.config["mask_token_id"]
        self.timesteps: Tensor | None = None
        self.temperatures: Tensor | None = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        temperature: float | tuple[float, float] | list[float] = (2, 0),
        device: str | torch.device | None = None,
    ) -> None:
        self.timesteps = torch.arange(num_inference_steps, device=device).flip(0)
        if isinstance(temperature, tuple | list):
            self.temperatures = torch.linspace(
                temperature[0], temperature[1], num_inference_steps, device=device
            )
        else:
            self.temperatures = torch.linspace(
                temperature, 0.01, num_inference_steps, device=device
            )

    def step(
        self,
        model_output: Tensor,
        timestep: Tensor,
        sample: Tensor,
        starting_mask_ratio: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> AmusedSchedulerOutput:
        assert self.timesteps is not None and self.temperatures is not None
        two_dim_input = sample.ndim == 3 and model_output.ndim == 4
        if two_dim_input:
            batch_size, codebook_size, height, width = model_output.shape
            sample = sample.reshape(batch_size, height * width)
            model_output = model_output.reshape(batch_size, codebook_size, height * width).permute(
                0, 2, 1
            )

        unknown_map = sample == self.mask_token_id
        probs = model_output.softmax(dim=-1)

        device = probs.device
        probs_ = probs.to(generator.device) if generator is not None else probs
        if probs_.device.type == "cpu" and probs_.dtype != torch.float32:
            probs_ = probs_.float()
        probs_ = probs_.reshape(-1, probs.size(-1))
        pred_original_sample = torch.multinomial(probs_, 1, generator=generator).to(device=device)
        pred_original_sample = pred_original_sample[:, 0].view(*probs.shape[:-1])
        pred_original_sample = torch.where(unknown_map, pred_original_sample, sample)

        if timestep == 0:
            prev_sample = pred_original_sample
        else:
            seq_len = sample.shape[1]
            step_idx = (self.timesteps == timestep).nonzero()
            ratio = (step_idx + 1) / len(self.timesteps)
            if self.config["masking_schedule"] == "cosine":
                mask_ratio = torch.cos(ratio * math.pi / 2)
            else:
                mask_ratio = 1 - ratio
            mask_ratio = starting_mask_ratio * mask_ratio

            mask_len = (seq_len * mask_ratio).floor()
            # Never mask more tokens than are still masked, and always keep one masked.
            mask_len = torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            mask_len = torch.max(torch.tensor([1], device=model_output.device), mask_len)

            selected_probs = torch.gather(probs, -1, pred_original_sample[:, :, None])[:, :, 0]
            # Tokens given in the input keep the highest confidence.
            selected_probs = torch.where(
                unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max
            )
            masking = mask_by_random_topk(
                mask_len, selected_probs, self.temperatures[step_idx], generator
            )
            prev_sample = torch.where(masking, self.mask_token_id, pred_original_sample)

        if two_dim_input:
            prev_sample = prev_sample.reshape(batch_size, height, width)
            pred_original_sample = pred_original_sample.reshape(batch_size, height, width)
        return AmusedSchedulerOutput(prev_sample, pred_original_sample)
