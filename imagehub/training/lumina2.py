"""Lumina-Image 2.0 LoRA training after diffusers' DreamBooth script."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from imagehub.backends.lumina2 import (
    DEFAULT_SYSTEM_PROMPT,
    Lumina2Components,
    decode_latents,
    denoise,
)
from imagehub.models.lumina2_transformer import Lumina2TransformerBlock
from imagehub.training.data import preprocess_image_xl
from imagehub.training.family import DDPM_OPTIONS, Collator, TrainingFamily, reject_options
from imagehub.training.flow_matching import FlowMatchingObjective, loss_weighting

if TYPE_CHECKING:
    from imagehub.training.arguments import TrainingArguments


#: ``train_dreambooth_lora_lumina2.py`` argparse defaults.
WEIGHTING_SCHEME = "none"
LOGIT_MEAN = 0.0
LOGIT_STD = 1.0
MODE_SCALE = 1.29
MAX_SEQUENCE_LENGTH = 256
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


class Lumina2Collator:
    """The script's image transform and Gemma prompt encoding."""

    def __init__(
        self,
        components: Lumina2Components,
        *,
        resolution: int,
        center_crop: bool,
        random_flip: bool,
        max_sequence_length: int,
        system_prompt: str,
        generator: torch.Generator | None,
    ) -> None:
        self.components, self.resolution = components, resolution
        self.center_crop, self.random_flip = center_crop, random_flip
        self.max_sequence_length, self.system_prompt = max_sequence_length, system_prompt
        self.generator = generator

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
        pixels, prompts = [], []
        for feature in features:
            image, _, _ = preprocess_image_xl(
                feature["image"],
                resolution=self.resolution,
                center_crop=self.center_crop,
                random_flip=self.random_flip,
                generator=self.generator,
            )
            pixels.append(image)
            prompts.append(self.system_prompt + " <Prompt Start> " + feature["text"])
        input_ids, attention_mask = self.components.tokenizer(
            prompts, max_length=self.max_sequence_length, padding_side="right"
        )
        return {
            "pixel_values": torch.stack(pixels).contiguous(),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class Lumina2LoRAObjective(FlowMatchingObjective):
    """The script's posterior, reversed flow interpolation and flow target."""

    def __init__(self, components: Lumina2Components, **options: Any) -> None:
        super().__init__(components.scheduler_config, precondition_outputs=False, **options)
        self.components = components
        self.latent_channels = int(components.vae.config["latent_channels"])
        self.downsample = 2 ** (len(components.vae.config["block_out_channels"]) - 1)

    def latent_shape(self, pixels: Tensor) -> tuple[int, ...]:
        batch, _, height, width = pixels.shape
        return batch, self.latent_channels, height // self.downsample, width // self.downsample

    @torch.no_grad()
    def encode_images(self, pixels: Tensor, posterior_noise: Tensor) -> Tensor:
        vae = self.components.vae
        mean, logvar = vae.encode_moments(pixels.to(next(vae.parameters()).dtype))
        latents = mean + torch.exp(0.5 * logvar) * posterior_noise.to(mean)
        latents = (latents - (vae.shift_factor or 0.0)) * vae.scaling_factor
        return latents.to(next(self.components.transformer.parameters()).dtype)

    @torch.no_grad()
    def encode_condition(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        components = self.components
        embeds = components.text_encoder(
            batch["input_ids"], batch["attention_mask"].long(), hidden_state_index=-2
        )
        return {
            "prompt_embeds": embeds.to(next(components.transformer.parameters()).dtype),
            "attention_mask": batch["attention_mask"],
        }

    def predict(self, noisy: Tensor, timesteps: Tensor, condition: Any) -> Tensor:
        return self.components.transformer(
            noisy,
            timesteps / self.num_train_timesteps,
            condition["prompt_embeds"],
            condition["attention_mask"],
        )

    def loss(self, latents: Tensor, condition: Any, draws: Mapping[str, Tensor]) -> Tensor:
        noise = draws["noise"].to(device=latents.device, dtype=latents.dtype)
        timesteps = draws["timesteps"].to(latents.device)
        sigmas = draws["sigmas"].to(device=latents.device, dtype=latents.dtype).flatten()
        while sigmas.ndim < latents.ndim:
            sigmas = sigmas.unsqueeze(-1)
        prediction = self.predict((1.0 - sigmas) * noise + sigmas * latents, timesteps, condition)
        target = latents - noise
        squared = (
            loss_weighting(self.weighting_scheme, sigmas).float()
            * (prediction.float() - target.float()) ** 2
        ).reshape(target.shape[0], -1)
        return torch.mean(squared, 1).mean()


class Lumina2Family(TrainingFamily):
    """Lumina-Image 2.0 transformer-only LoRA training."""

    name = "Lumina-Image 2.0"
    prefix = "transformer"
    batch_keys = ("pixel_values", "input_ids", "attention_mask")
    checkpoint_blocks = (Lumina2TransformerBlock,)

    def validate(self, args: TrainingArguments) -> None:
        reject_options(
            args,
            DDPM_OPTIONS + ("precondition_outputs", "guidance_scale", "complex_human_instruction"),
            self.name,
        )

    def load(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str,
    ) -> Lumina2Components:
        components = Lumina2Components.load(
            checkpoint, device=device, dtype=dtype, model_id=model_id
        )
        components.vae.to(torch.float32)
        components.text_encoder.to(torch.bfloat16)
        return components

    def trainable(self, components: Any) -> nn.Module:
        return components.transformer

    def frozen(self, components: Any) -> list[nn.Module]:
        return [components.text_encoder, components.vae]

    def collator(
        self, components: Any, args: TrainingArguments, *, generator: torch.Generator, train: bool
    ) -> Collator:
        return Lumina2Collator(
            components,
            resolution=args.resolution,
            center_crop=args.center_crop if train else True,
            random_flip=args.random_flip if train else False,
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
            system_prompt=SYSTEM_PROMPT if args.system_prompt is None else args.system_prompt,
            generator=generator,
        )

    def objective(self, components: Any, args: TrainingArguments) -> Lumina2LoRAObjective:
        def pick(value: Any, default: Any) -> Any:
            return default if value is None else value

        return Lumina2LoRAObjective(
            components,
            weighting_scheme=pick(args.weighting_scheme, WEIGHTING_SCHEME),
            logit_mean=pick(args.logit_mean, LOGIT_MEAN),
            logit_std=pick(args.logit_std, LOGIT_STD),
            mode_scale=pick(args.mode_scale, MODE_SCALE),
        )

    def sample(
        self, components: Any, prompt: str, args: TrainingArguments, device: torch.device
    ) -> Tensor:
        vae = components.vae
        downsample = 2 ** (len(vae.config["block_out_channels"]) - 1)
        latents = torch.randn(
            (
                1,
                int(vae.config["latent_channels"]),
                args.resolution // downsample,
                args.resolution // downsample,
            ),
            generator=torch.Generator("cpu").manual_seed(args.seed),
        )
        final = denoise(
            components,
            prompt=prompt,
            negative_prompt="",
            system_prompt=SYSTEM_PROMPT if args.system_prompt is None else args.system_prompt,
            steps=args.validation_num_inference_steps,
            guidance=args.validation_guidance_scale,
            cfg_trunc_ratio=1.0,
            cfg_normalization=True,
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
            latents=latents.to(device),
        )
        return decode_latents(vae, final)[0]
