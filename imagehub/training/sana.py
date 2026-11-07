"""Sana LoRA training after diffusers' ``train_dreambooth_lora_sana.py``."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from imagehub.backends.sana import (
    SanaComponents,
    classify_height_width_bin,
    decode_latents,
    denoise,
)
from imagehub.models.sana_transformer import SanaTransformerBlock
from imagehub.training.data import preprocess_image_xl
from imagehub.training.family import DDPM_OPTIONS, Collator, TrainingFamily, reject_options
from imagehub.training.flow_matching import FlowMatchingObjective, sample_density

if TYPE_CHECKING:
    from imagehub.training.arguments import TrainingArguments


#: The script's argparse defaults.
WEIGHTING_SCHEME = "none"
LOGIT_MEAN = 0.0
LOGIT_STD = 1.0
MODE_SCALE = 1.29
MAX_SEQUENCE_LENGTH = 300


def _instruction(args: TrainingArguments) -> tuple[str, ...]:
    # The training script passes its argparse default (None) directly to encode_prompt.
    return () if args.complex_human_instruction is None else args.complex_human_instruction


class SanaCollator:
    """The Sana script's image transform and Gemma prompt preparation."""

    def __init__(
        self,
        components: SanaComponents,
        *,
        resolution: int,
        center_crop: bool,
        random_flip: bool,
        max_sequence_length: int,
        instruction: Sequence[str],
        generator: torch.Generator | None = None,
    ) -> None:
        self.components = components
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.max_sequence_length = max_sequence_length
        self.instruction = instruction
        self.generator = generator

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
        pixels, prompts = [], []
        instruction = self.instruction
        max_length = self.max_sequence_length
        if instruction:
            header = "\n".join(instruction)
            max_length = len(self.components.tokenizer.tokenize(header)) + max_length - 2
        else:
            header = ""
        for feature in features:
            image, _, _ = preprocess_image_xl(
                feature["image"],
                resolution=self.resolution,
                center_crop=self.center_crop,
                random_flip=self.random_flip,
                generator=self.generator,
            )
            pixels.append(image)
            # SanaPipeline._text_preprocessing(clean_caption=False).
            prompts.append(header + feature["text"].lower().strip())
        input_ids, attention_mask = self.components.tokenizer(
            prompts, max_length=max_length, padding_side="right"
        )
        return {
            "pixel_values": torch.stack(pixels),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class SanaLoRAObjective(FlowMatchingObjective):
    """The deterministic DC-AE Sana flow-matching step.

    Unlike KL autoencoders, DC-AE returns a latent directly.  Keeping ``draw`` here avoids a
    posterior-noise draw before noise and changes neither the shared SD 3 nor FLUX streams.
    """

    def __init__(
        self, components: SanaComponents, *, max_sequence_length: int, **options: Any
    ) -> None:
        super().__init__(components.scheduler_config, precondition_outputs=False, **options)
        self.components = components
        self.latent_channels = int(components.vae.config["latent_channels"])
        self.downsample = int(components.vae.spatial_compression_ratio)
        self.max_sequence_length = max_sequence_length

    def latent_shape(self, pixels: Tensor) -> tuple[int, ...]:
        batch, _, height, width = pixels.shape
        return (batch, self.latent_channels, height // self.downsample, width // self.downsample)

    def draw(self, batch: Mapping[str, Tensor], generator: torch.Generator) -> dict[str, Tensor]:
        shape = self.latent_shape(batch["pixel_values"])
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
            "noise": noise,
            "timesteps": self.timesteps[indices],
            "sigmas": self.sigmas[indices],
        }

    @torch.no_grad()
    def encode_images(self, pixels: Tensor, posterior_noise: Tensor | None = None) -> Tensor:
        vae = self.components.vae
        # The script keeps DC-AE in fp32 even during mixed-precision training.
        latents = vae.encode(pixels.to(next(vae.parameters()).dtype)) * vae.scaling_factor
        return latents.to(next(self.components.transformer.parameters()).dtype)

    @torch.no_grad()
    def encode_condition(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        components = self.components
        embeds = components.text_encoder(batch["input_ids"], batch["attention_mask"])
        # The tokenized sequence may include the instruction; retain one BOS and the final prompt
        # span exactly as SanaPipeline.encode_prompt does.
        select = [0, *range(-self.max_sequence_length + 1, 0)]
        return {
            "prompt_embeds": embeds[:, select].to(next(components.transformer.parameters()).dtype),
            "attention_mask": batch["attention_mask"][:, select],
        }

    def predict(self, noisy: Tensor, timesteps: Tensor, condition: Any) -> Tensor:
        return self.components.transformer(
            noisy,
            condition["prompt_embeds"],
            timesteps,
            encoder_attention_mask=condition["attention_mask"],
        )

    def loss_from(self, batch: Mapping[str, Tensor], draws: Mapping[str, Tensor]) -> Tensor:
        latents = self.encode_images(batch["pixel_values"])
        return self.loss(latents, self.encode_condition(batch), draws)


class SanaFamily(TrainingFamily):
    """Sana's transformer-only, deterministic-DC-AE LoRA path."""

    name = "Sana"
    prefix = "transformer"
    batch_keys = ("pixel_values", "input_ids", "attention_mask")
    checkpoint_blocks = (SanaTransformerBlock,)

    def validate(self, args: TrainingArguments) -> None:
        reject_options(
            args,
            DDPM_OPTIONS + ("precondition_outputs", "guidance_scale", "system_prompt"),
            self.name,
        )

    def load(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str,
    ) -> SanaComponents:
        components = SanaComponents.load(checkpoint, device=device, dtype=dtype, model_id=model_id)
        components.vae.to(torch.float32)
        return components

    def trainable(self, components: Any) -> nn.Module:
        return components.transformer

    def frozen(self, components: Any) -> list[nn.Module]:
        return [components.text_encoder, components.vae]

    def collator(
        self,
        components: Any,
        args: TrainingArguments,
        *,
        generator: torch.Generator,
        train: bool,
    ) -> Collator:
        return SanaCollator(
            components,
            resolution=args.resolution,
            center_crop=args.center_crop if train else True,
            random_flip=args.random_flip if train else False,
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
            instruction=_instruction(args),
            generator=generator,
        )

    def objective(self, components: Any, args: TrainingArguments) -> SanaLoRAObjective:
        def pick(value: Any, default: Any) -> Any:
            return default if value is None else value

        objective = SanaLoRAObjective(
            components,
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
            weighting_scheme=pick(args.weighting_scheme, WEIGHTING_SCHEME),
            logit_mean=pick(args.logit_mean, LOGIT_MEAN),
            logit_std=pick(args.logit_std, LOGIT_STD),
            mode_scale=pick(args.mode_scale, MODE_SCALE),
        )
        return objective

    def sample(
        self, components: Any, prompt: str, args: TrainingArguments, device: torch.device
    ) -> Tensor:
        vae = components.vae
        size = int(components.transformer.config.get("sample_size", 32))
        height, width = classify_height_width_bin(args.resolution, args.resolution, size)
        latents = torch.randn(
            (
                1,
                int(vae.config["latent_channels"]),
                height // vae.spatial_compression_ratio,
                width // vae.spatial_compression_ratio,
            ),
            generator=torch.Generator("cpu").manual_seed(args.seed),
            dtype=torch.float32,
        )
        final = denoise(
            components,
            prompt=prompt,
            negative_prompt="",
            steps=args.validation_num_inference_steps,
            guidance=args.validation_guidance_scale,
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
            instruction=_instruction(args),
            latents=latents.to(device),
        )
        return decode_latents(
            vae,
            final,
            output_width=args.resolution,
            output_height=args.resolution,
            resize=True,
        )[0]
