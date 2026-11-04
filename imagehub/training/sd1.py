"""The Stable Diffusion 1.x LoRA objective of diffusers' ``train_text_to_image_lora.py``."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from imagehub.backends.stable_diffusion import StableDiffusionComponents
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.schedulers.ddpm import DDPMScheduler

_SCHEDULE_KEYS = (
    "num_train_timesteps",
    "beta_start",
    "beta_end",
    "beta_schedule",
    "trained_betas",
    "prediction_type",
)


class StableDiffusionLoRAObjective:
    """Noise-prediction loss with frozen VAE and text encoder.

    The random tensors of a step come from :meth:`draw`, in the reference script's order: the
    posterior sample, the noise, the offset noise and the timesteps.
    """

    def __init__(
        self,
        components: StableDiffusionComponents,
        *,
        prediction_type: str | None = None,
        snr_gamma: float | None = None,
        noise_offset: float = 0.0,
    ) -> None:
        if not isinstance(components.vae, AutoencoderKL):
            raise ValueError("Training needs a KL autoencoder to sample latents from")
        self.components = components
        # Training needs only the noise schedule and prediction type of the snapshot's scheduler;
        # its sampling settings do not apply.
        self.scheduler = DDPMScheduler(
            {k: v for k, v in components.scheduler_config.items() if k in _SCHEDULE_KEYS}
        )
        self.prediction_type = prediction_type or self.scheduler.config["prediction_type"]
        if self.prediction_type not in ("epsilon", "v_prediction"):
            raise ValueError(f"Unknown prediction type {self.prediction_type!r}")
        self.snr_gamma = snr_gamma
        self.noise_offset = noise_offset
        vae = components.vae
        self.latent_channels = int(vae.config["latent_channels"])
        self.downsample = 2 ** (len(vae.config["block_out_channels"]) - 1)
        self.num_train_timesteps = int(self.scheduler.config["num_train_timesteps"])

    def latent_shape(self, pixels: Tensor) -> tuple[int, int, int, int]:
        batch, _, height, width = pixels.shape
        return (
            batch,
            self.latent_channels,
            height // self.downsample,
            width // self.downsample,
        )

    def draw(self, pixels: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
        """``(posterior noise, noise, timesteps)`` drawn on the CPU from ``generator``."""
        shape = self.latent_shape(pixels)
        posterior = torch.randn(shape, generator=generator)
        noise = torch.randn(shape, generator=generator)
        if self.noise_offset:
            noise += self.noise_offset * torch.randn(shape[:2] + (1, 1), generator=generator)
        timesteps = torch.randint(0, self.num_train_timesteps, (shape[0],), generator=generator)
        return posterior, noise, timesteps.long()

    @torch.no_grad()
    def encode_images(self, pixels: Tensor, posterior_noise: Tensor) -> Tensor:
        vae = self.components.vae
        assert isinstance(vae, AutoencoderKL)
        mean, logvar = vae.encode_moments(pixels)
        std = torch.exp(0.5 * logvar)
        latents = mean + std * posterior_noise.to(device=mean.device, dtype=mean.dtype)
        return latents * vae.scaling_factor

    @torch.no_grad()
    def encode_text(self, input_ids: Tensor) -> Tensor:
        return self.components.text_encoder(input_ids)

    def _alphas(self, like: Tensor, timesteps: Tensor) -> tuple[Tensor, Tensor]:
        alphas = self.scheduler.alphas_cumprod.to(device=like.device, dtype=like.dtype)
        timesteps = timesteps.to(like.device)
        sqrt_alpha = (alphas[timesteps] ** 0.5).flatten()
        sqrt_one_minus = ((1 - alphas[timesteps]) ** 0.5).flatten()
        while sqrt_alpha.ndim < like.ndim:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha, sqrt_one_minus

    def snr(self, timesteps: Tensor) -> Tensor:
        alphas = self.scheduler.alphas_cumprod
        alpha = (alphas**0.5).to(device=timesteps.device)[timesteps].float()
        sigma = ((1.0 - alphas) ** 0.5).to(device=timesteps.device)[timesteps].float()
        return (alpha / sigma) ** 2

    def loss(
        self,
        latents: Tensor,
        encoder_hidden_states: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        noise = noise.to(device=latents.device, dtype=latents.dtype)
        timesteps = timesteps.to(latents.device)
        sqrt_alpha, sqrt_one_minus = self._alphas(latents, timesteps)
        noisy = sqrt_alpha * latents + sqrt_one_minus * noise
        if self.prediction_type == "epsilon":
            target = noise
        else:
            target = sqrt_alpha * noise - sqrt_one_minus * latents
        prediction = self.components.unet(noisy, timesteps, encoder_hidden_states)
        if self.snr_gamma is None:
            return F.mse_loss(prediction.float(), target.float(), reduction="mean")
        snr = self.snr(timesteps)
        weights = torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[
            0
        ]
        weights = weights / snr if self.prediction_type == "epsilon" else weights / (snr + 1)
        loss = F.mse_loss(prediction.float(), target.float(), reduction="none")
        return (loss.mean(dim=list(range(1, loss.ndim))) * weights).mean()

    def __call__(self, pixels: Tensor, input_ids: Tensor, generator: torch.Generator) -> Tensor:
        posterior, noise, timesteps = self.draw(pixels, generator)
        latents = self.encode_images(pixels, posterior)
        return self.loss(latents, self.encode_text(input_ids), noise, timesteps)
