"""Z-Image LoRA training after diffusers' ``train_dreambooth_lora_z_image.py``."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from imagehub.backends.z_image import ZImageComponents, decode_latents, denoise
from imagehub.models.z_image_transformer import ZImageTransformerBlock
from imagehub.training.data import preprocess_image_xl
from imagehub.training.family import DDPM_OPTIONS, Collator, TrainingFamily, reject_options
from imagehub.training.flow_matching import FlowMatchingObjective, sample_density

if TYPE_CHECKING:
    from imagehub.training.arguments import TrainingArguments


#: ``train_dreambooth_lora_z_image.py`` argparse defaults.
WEIGHTING_SCHEME = "none"
LOGIT_MEAN = 0.0
LOGIT_STD = 1.0
MODE_SCALE = 1.29
MAX_SEQUENCE_LENGTH = 512


class ZImageCollator:
    """The Z-Image script's image transform and chat-template prompt encoding."""

    def __init__(
        self,
        components: ZImageComponents,
        *,
        resolution: int,
        center_crop: bool,
        random_flip: bool,
        max_sequence_length: int,
        generator: torch.Generator | None,
    ) -> None:
        self.components = components
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.max_sequence_length = max_sequence_length
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
            prompts.append(self.components.tokenizer.chat_prompt(feature["text"]))
        input_ids, attention_mask = self.components.tokenizer(prompts, self.max_sequence_length)
        return {
            "pixel_values": torch.stack(pixels),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class ZImageLoRAObjective(FlowMatchingObjective):
    """The Z-Image script's deterministic VAE and rectified-flow step."""

    def __init__(self, components: ZImageComponents, **options: Any) -> None:
        super().__init__(components.scheduler_config, precondition_outputs=False, **options)
        self.components = components
        self.downsample = 2 ** (len(components.vae.config["block_out_channels"]) - 1)
        self.latent_channels = int(components.vae.config["latent_channels"])

    def latent_shape(self, pixels: Tensor) -> tuple[int, ...]:
        batch, _, height, width = pixels.shape
        return batch, self.latent_channels, height // self.downsample, width // self.downsample

    def draw(self, batch: Mapping[str, Tensor], generator: torch.Generator) -> dict[str, Tensor]:
        """The script has no posterior draw: noise is followed immediately by density."""
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
        latents = vae.encode(pixels.to(next(vae.parameters()).dtype))
        latents = (latents - (vae.shift_factor or 0.0)) * vae.scaling_factor
        return latents.to(next(self.components.transformer.parameters()).dtype)

    @torch.no_grad()
    def encode_condition(self, batch: Mapping[str, Tensor]) -> list[Tensor]:
        components = self.components
        _, hidden_states = components.text_encoder(
            batch["input_ids"], batch["attention_mask"], output_hidden_states=True
        )
        embeds = hidden_states[-2]
        return [
            embeds[index][mask].to(next(components.transformer.parameters()).dtype)
            for index, mask in enumerate(batch["attention_mask"])
        ]

    def predict(self, noisy: Tensor, timesteps: Tensor, condition: Any) -> Tensor:
        output = self.components.transformer(
            list(noisy.unsqueeze(2).unbind(dim=0)), (1000 - timesteps) / 1000, condition
        )
        return -torch.stack(output).squeeze(2)

    def loss_from(self, batch: Mapping[str, Tensor], draws: Mapping[str, Tensor]) -> Tensor:
        latents = self.encode_images(batch["pixel_values"])
        return self.loss(latents, self.encode_condition(batch), draws)


class ZImageFamily(TrainingFamily):
    """Z-Image's transformer-only LoRA training path."""

    name = "Z-Image"
    prefix = "transformer"
    batch_keys = ("pixel_values", "input_ids", "attention_mask")
    checkpoint_blocks = (ZImageTransformerBlock,)

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
    ) -> ZImageComponents:
        return ZImageComponents.load(checkpoint, device=device, dtype=dtype, model_id=model_id)

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
        return ZImageCollator(
            components,
            resolution=args.resolution,
            center_crop=args.center_crop if train else True,
            random_flip=args.random_flip if train else False,
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
            generator=generator,
        )

    def objective(self, components: Any, args: TrainingArguments) -> ZImageLoRAObjective:
        def pick(value: Any, default: Any) -> Any:
            return default if value is None else value

        return ZImageLoRAObjective(
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
            dtype=torch.float32,
        )
        final = denoise(
            components,
            prompt=prompt,
            negative_prompt=None,
            steps=args.validation_num_inference_steps,
            guidance=args.validation_guidance_scale,
            cfg_normalization=False,
            cfg_truncation=1.0,
            latents=latents.to(device),
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
        )
        return decode_latents(vae, final)[0]
