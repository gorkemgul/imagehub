"""Stable Diffusion XL training after diffusers' ``train_text_to_image_lora_sdxl.py``."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from imagehub.backends.stable_diffusion import decode_latents
from imagehub.backends.stable_diffusion_xl import StableDiffusionXLComponents, denoise
from imagehub.catalog import MODEL_CATALOG
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.training.data import StableDiffusionXLCollator
from imagehub.training.family import Collator, TrainingFamily
from imagehub.training.sd1 import StableDiffusionFamily, StableDiffusionLoRAObjective

if TYPE_CHECKING:
    from imagehub.training.arguments import TrainingArguments


class StableDiffusionXLLoRAObjective(StableDiffusionLoRAObjective):
    """The SDXL step: float32 autoencoder, both CLIP encoders and ``time_ids`` conditioning.

    Batches hold ``pixel_values``, ``input_ids``, ``input_ids_2`` and ``time_ids``.
    """

    components: StableDiffusionXLComponents  # type: ignore[assignment]

    @torch.no_grad()
    def encode_images(self, pixels: Tensor, posterior_noise: Tensor) -> Tensor:
        vae = self.components.vae
        assert isinstance(vae, AutoencoderKL)
        # The script keeps the autoencoder in float32 and casts the latents to the training dtype.
        mean, logvar = vae.encode_moments(pixels.to(next(vae.parameters()).dtype))
        std = torch.exp(0.5 * logvar)
        latents = mean + std * posterior_noise.to(device=mean.device, dtype=mean.dtype)
        return (latents * vae.scaling_factor).to(self.components.unet.conv_in.weight.dtype)

    @torch.no_grad()
    def encode_condition(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        components = self.components
        assert components.text_encoder is not None
        _, hidden_1 = components.text_encoder(batch["input_ids"], output_hidden_states=True)
        pooled, _, hidden_2 = components.text_encoder_2(
            batch["input_ids_2"], output_hidden_states=True
        )
        return {
            "prompt_embeds": torch.cat([hidden_1[-2], hidden_2[-2]], dim=-1),
            "text_embeds": pooled,
            "time_ids": batch["time_ids"],
        }

    def predict(self, noisy: Tensor, timesteps: Tensor, condition: Any) -> Tensor:
        return self.components.unet(
            noisy,
            timesteps,
            condition["prompt_embeds"],
            text_embeds=condition["text_embeds"],
            time_ids=condition["time_ids"],
        )


class StableDiffusionXLFamily(TrainingFamily):
    """SDXL base and fine-tunes: LoRA on the U-Net attention projections."""

    prefix = "unet"
    batch_keys = ("pixel_values", "input_ids", "input_ids_2", "time_ids")
    checkpoint_blocks = StableDiffusionFamily.checkpoint_blocks

    def load(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str,
    ) -> StableDiffusionXLComponents:
        overrides = dict(MODEL_CATALOG[model_id].config.get("scheduler", {}))
        components = StableDiffusionXLComponents.load(
            checkpoint,
            device=device,
            dtype=dtype,
            model_id=model_id,
            scheduler_overrides=overrides,
        )
        # The SDXL autoencoder overflows in float16; the script keeps it in float32.
        components.vae.to(torch.float32)
        return components

    def trainable(self, components: Any) -> nn.Module:
        return components.unet

    def frozen(self, components: Any) -> list[nn.Module]:
        return [components.text_encoder, components.text_encoder_2, components.vae]

    def collator(
        self,
        components: Any,
        args: TrainingArguments,
        *,
        generator: torch.Generator,
        train: bool,
    ) -> Collator:
        return StableDiffusionXLCollator(
            components.tokenizer,
            components.tokenizer_2,
            resolution=args.resolution,
            center_crop=args.center_crop if train else True,
            random_flip=args.random_flip if train else False,
            generator=generator,
        )

    def objective(self, components: Any, args: TrainingArguments) -> StableDiffusionXLLoRAObjective:
        return StableDiffusionXLLoRAObjective(
            components,
            prediction_type=args.prediction_type,
            snr_gamma=args.snr_gamma,
            noise_offset=args.noise_offset,
        )

    def sample(
        self, components: Any, prompt: str, args: TrainingArguments, device: torch.device
    ) -> Tensor:
        # As StableDiffusionXLBackend renders a seeded request of resolution x resolution.
        generator = torch.Generator("cpu").manual_seed(args.seed)
        side = args.resolution // 8
        latents = torch.randn((1, 4, side, side), generator=generator, dtype=torch.float32)
        size = (args.resolution, args.resolution)
        final = denoise(
            components,
            prompt=prompt,
            negative_prompt=None,
            steps=args.validation_num_inference_steps,
            guidance=args.validation_guidance_scale,
            latents=latents.to(device),
            original_size=size,
            crops_coords_top_left=(0, 0),
            target_size=size,
            generator=generator,
        )
        return decode_latents(components, final.to(next(components.vae.parameters()).dtype))[0]
