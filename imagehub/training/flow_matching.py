"""The rectified-flow training step shared by diffusers' DiT LoRA scripts (SD3 and later).

Timestep sampling, the training sigma table and the loss weighting follow
``diffusers.training_utils`` and ``FlowMatchEulerDiscreteScheduler`` operation for operation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from imagehub.training.family import Objective

WEIGHTING_SCHEMES = ("logit_normal", "mode", "sigma_sqrt", "cosmap", "none")


def training_sigmas(config: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    """``(timesteps, sigmas)`` of a freshly built flow-matching scheduler (shift applied)."""
    count = int(config.get("num_train_timesteps", 1000))
    shift = float(config.get("shift", 1.0))
    timesteps = torch.linspace(1, count, count, dtype=torch.float64).flip(0).to(torch.float32)
    sigmas = timesteps / count
    if not config.get("use_dynamic_shifting", False):
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return sigmas * count, sigmas


def sample_density(
    scheme: str,
    batch_size: int,
    *,
    logit_mean: float,
    logit_std: float,
    mode_scale: float,
    generator: torch.Generator,
) -> Tensor:
    """``compute_density_for_timestep_sampling``: ``u`` in ``[0, 1]`` per sample."""
    if scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), generator=generator)
        return torch.nn.functional.sigmoid(u)
    u = torch.rand(size=(batch_size,), generator=generator)
    if scheme == "mode":
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    return u


def loss_weighting(scheme: str, sigmas: Tensor) -> Tensor:
    """``compute_loss_weighting_for_sd3``."""
    if scheme == "sigma_sqrt":
        return (sigmas**-2.0).float()
    if scheme == "cosmap":
        bottom = 1 - 2 * sigmas + 2 * sigmas**2
        return 2 / (math.pi * bottom)
    return torch.ones_like(sigmas)


class FlowMatchingObjective(Objective):
    """``noisy = (1 - sigma) * latents + sigma * noise``; target ``noise - latents`` (or the latents
    with preconditioned outputs); per-sample weighted MSE, then the batch mean.

    Families provide :meth:`latent_shape`, :meth:`encode_images`, :meth:`encode_condition` and
    :meth:`predict`.
    """

    def __init__(
        self,
        scheduler_config: Mapping[str, Any],
        *,
        weighting_scheme: str,
        logit_mean: float,
        logit_std: float,
        mode_scale: float,
        precondition_outputs: bool,
    ) -> None:
        if weighting_scheme not in WEIGHTING_SCHEMES:
            raise ValueError(f"Unknown weighting scheme {weighting_scheme!r}")
        self.timesteps, self.sigmas = training_sigmas(scheduler_config)
        self.num_train_timesteps = len(self.timesteps)
        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mode_scale = mode_scale
        self.precondition_outputs = precondition_outputs

    def latent_shape(self, pixels: Tensor) -> tuple[int, ...]:
        raise NotImplementedError

    def encode_images(self, pixels: Tensor, posterior_noise: Tensor) -> Tensor:
        raise NotImplementedError

    def encode_condition(self, batch: Mapping[str, Tensor]) -> Any:
        raise NotImplementedError

    def predict(self, noisy: Tensor, timesteps: Tensor, condition: Any) -> Tensor:
        raise NotImplementedError

    def draw(self, batch: Mapping[str, Tensor], generator: torch.Generator) -> dict[str, Tensor]:
        """``posterior``, ``noise``, ``timesteps`` and their ``sigmas``, drawn on the CPU."""
        shape = self.latent_shape(batch["pixel_values"])
        posterior = torch.randn(shape, generator=generator)
        noise = torch.randn(shape, generator=generator)
        u = sample_density(
            self.weighting_scheme,
            shape[0],
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
            generator=generator,
        )
        indices = (u * self.num_train_timesteps).long()
        return {
            "posterior": posterior,
            "noise": noise,
            "timesteps": self.timesteps[indices],
            "sigmas": self.sigmas[indices],
        }

    def loss(self, latents: Tensor, condition: Any, draws: Mapping[str, Tensor]) -> Tensor:
        noise = draws["noise"].to(device=latents.device, dtype=latents.dtype)
        timesteps = draws["timesteps"].to(latents.device)
        sigmas = draws["sigmas"].to(device=latents.device, dtype=latents.dtype).flatten()
        while sigmas.ndim < latents.ndim:
            sigmas = sigmas.unsqueeze(-1)
        noisy = (1.0 - sigmas) * latents + sigmas * noise
        prediction = self.predict(noisy, timesteps, condition)
        if self.precondition_outputs:
            prediction = prediction * (-sigmas) + noisy
            target = latents
        else:
            target = noise - latents
        weighting = loss_weighting(self.weighting_scheme, sigmas)
        squared = (weighting.float() * (prediction.float() - target.float()) ** 2).reshape(
            target.shape[0], -1
        )
        return torch.mean(squared, 1).mean()

    def loss_from(self, batch: Mapping[str, Tensor], draws: Mapping[str, Tensor]) -> Tensor:
        latents = self.encode_images(batch["pixel_values"], draws["posterior"])
        return self.loss(latents, self.encode_condition(batch), draws)
