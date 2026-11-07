"""Stable Diffusion 3 / 3.5 training after diffusers' ``train_dreambooth_lora_sd3.py``."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.backends.stable_diffusion_3 import (
    OPTION_DEFAULTS,
    StableDiffusion3Components,
    decode_latents,
    denoise,
)
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.sd3_transformer import JointTransformerBlock
from imagehub.training.data import preprocess_image_xl
from imagehub.training.family import DDPM_OPTIONS, Collator, TrainingFamily, reject_options
from imagehub.training.flow_matching import FlowMatchingObjective

if TYPE_CHECKING:
    from imagehub.training.arguments import TrainingArguments

#: The script's defaults.
WEIGHTING_SCHEME = "logit_normal"
LOGIT_MEAN = 0.0
LOGIT_STD = 1.0
MODE_SCALE = 1.29
PRECONDITION_OUTPUTS = True
MAX_SEQUENCE_LENGTH = 77


class StableDiffusion3Collator:
    """Pixels (resize, flip, crop) and ids for both CLIP tokenizers and the T5 tokenizer."""

    def __init__(
        self,
        components: StableDiffusion3Components,
        *,
        resolution: int,
        center_crop: bool,
        random_flip: bool,
        max_sequence_length: int,
        generator: torch.Generator | None = None,
    ) -> None:
        self.components = components
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.max_sequence_length = max_sequence_length
        self.generator = generator

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
        components = self.components
        pixels, ids, ids_2, ids_3 = [], [], [], []
        for feature in features:
            image, _, _ = preprocess_image_xl(
                feature["image"],
                resolution=self.resolution,
                center_crop=self.center_crop,
                random_flip=self.random_flip,
                generator=self.generator,
            )
            pixels.append(image)
            ids.append(components.tokenizer(feature["text"]))
            ids_2.append(components.tokenizer_2(feature["text"]))
            t5_ids, _ = components.tokenizer_3([feature["text"]], self.max_sequence_length)
            ids_3.append(t5_ids[0])
        return {
            "pixel_values": torch.stack(pixels),
            "input_ids": torch.tensor(ids),
            "input_ids_2": torch.tensor(ids_2),
            "input_ids_3": torch.stack(ids_3),
        }


class StableDiffusion3LoRAObjective(FlowMatchingObjective):
    """The SD3 step: float32 autoencoder with shift and scale, CLIP-L / CLIP-G / T5 prompts."""

    def __init__(self, components: StableDiffusion3Components, **options: Any) -> None:
        super().__init__(components.scheduler_config, **options)
        vae = components.vae
        if not isinstance(vae, AutoencoderKL):
            raise ValueError("Training needs a KL autoencoder to sample latents from")
        self.components = components
        self.latent_channels = int(vae.config["latent_channels"])
        self.downsample = 2 ** (len(vae.config["block_out_channels"]) - 1)

    def latent_shape(self, pixels: Tensor) -> tuple[int, ...]:
        batch, _, height, width = pixels.shape
        return (batch, self.latent_channels, height // self.downsample, width // self.downsample)

    @torch.no_grad()
    def encode_images(self, pixels: Tensor, posterior_noise: Tensor) -> Tensor:
        vae = self.components.vae
        mean, logvar = vae.encode_moments(pixels.to(next(vae.parameters()).dtype))
        std = torch.exp(0.5 * logvar)
        latents = mean + std * posterior_noise.to(device=mean.device, dtype=mean.dtype)
        latents = (latents - (vae.shift_factor or 0.0)) * vae.scaling_factor
        return latents.to(next(self.components.transformer.parameters()).dtype)

    @torch.no_grad()
    def encode_condition(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        components = self.components
        clip_embeds, pooled = [], []
        for encoder, key in (
            (components.text_encoder, "input_ids"),
            (components.text_encoder_2, "input_ids_2"),
        ):
            text_embeds, _, hidden_states = encoder(batch[key], output_hidden_states=True)
            clip_embeds.append(hidden_states[-2])
            pooled.append(text_embeds)
        clip = torch.cat(clip_embeds, dim=-1)
        t5 = components.text_encoder_3(batch["input_ids_3"])[0]
        clip = F.pad(clip, (0, t5.shape[-1] - clip.shape[-1]))
        return {
            "prompt_embeds": torch.cat([clip, t5], dim=-2),
            "pooled_projections": torch.cat(pooled, dim=-1),
        }

    def predict(self, noisy: Tensor, timesteps: Tensor, condition: Any) -> Tensor:
        return self.components.transformer(
            noisy, condition["prompt_embeds"], condition["pooled_projections"], timesteps
        )


class StableDiffusion3Family(TrainingFamily):
    """SD 3 / 3.5: LoRA on the joint attention projections of the transformer."""

    name = "Stable Diffusion 3"
    prefix = "transformer"
    batch_keys = ("pixel_values", "input_ids", "input_ids_2", "input_ids_3")
    checkpoint_blocks = (JointTransformerBlock,)

    def validate(self, args: TrainingArguments) -> None:
        reject_options(
            args, DDPM_OPTIONS + ("complex_human_instruction", "guidance_scale"), self.name
        )

    def load(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str,
    ) -> StableDiffusion3Components:
        components = StableDiffusion3Components.load(
            checkpoint, device=device, dtype=dtype, model_id=model_id
        )
        # The script keeps the autoencoder in float32.
        components.vae.to(torch.float32)
        return components

    def trainable(self, components: Any) -> nn.Module:
        return components.transformer

    def frozen(self, components: Any) -> list[nn.Module]:
        return [
            components.text_encoder,
            components.text_encoder_2,
            components.text_encoder_3,
            components.vae,
        ]

    def collator(
        self,
        components: Any,
        args: TrainingArguments,
        *,
        generator: torch.Generator,
        train: bool,
    ) -> Collator:
        return StableDiffusion3Collator(
            components,
            resolution=args.resolution,
            center_crop=args.center_crop if train else True,
            random_flip=args.random_flip if train else False,
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
            generator=generator,
        )

    def objective(self, components: Any, args: TrainingArguments) -> StableDiffusion3LoRAObjective:
        def pick(value: Any, default: Any) -> Any:
            return default if value is None else value

        return StableDiffusion3LoRAObjective(
            components,
            weighting_scheme=pick(args.weighting_scheme, WEIGHTING_SCHEME),
            logit_mean=pick(args.logit_mean, LOGIT_MEAN),
            logit_std=pick(args.logit_std, LOGIT_STD),
            mode_scale=pick(args.mode_scale, MODE_SCALE),
            precondition_outputs=pick(args.precondition_outputs, PRECONDITION_OUTPUTS),
        )

    def sample(
        self, components: Any, prompt: str, args: TrainingArguments, device: torch.device
    ) -> Tensor:
        # As StableDiffusion3Backend renders a seeded request of resolution x resolution.
        vae = components.vae
        downsample = 2 ** (len(vae.config["block_out_channels"]) - 1)
        side = args.resolution // downsample
        shape = (1, int(vae.config["latent_channels"]), side, side)
        latents = torch.randn(
            shape, generator=torch.Generator("cpu").manual_seed(args.seed), dtype=torch.float32
        )
        final = denoise(
            components,
            prompt=prompt,
            negative_prompt=None,
            steps=args.validation_num_inference_steps,
            guidance=args.validation_guidance_scale,
            latents=latents.to(device),
            max_sequence_length=int(OPTION_DEFAULTS["max_sequence_length"]),
        )
        return decode_latents(vae, final)[0]
