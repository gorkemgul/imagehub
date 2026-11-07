"""Qwen-Image and Qwen-Image 2.1 LoRA training reference implementations."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

from imagehub.backends.qwen_image import (
    _ORIGINAL_DROP_INDEX,
    _ORIGINAL_PROMPT_TEMPLATE,
    _PROMPT_TEMPLATE,
    _SYSTEM_TEMPLATE,
    OriginalQwenImageComponents,
    QwenImageComponents,
    decode_latents,
    decode_original_latents,
    denoise,
    denoise_original,
)
from imagehub.models.qwen_image21_transformer import QwenImage21TransformerBlock
from imagehub.models.qwen_image_transformer import QwenImageTransformerBlock
from imagehub.training.data import preprocess_image_xl
from imagehub.training.family import DDPM_OPTIONS, Collator, TrainingFamily, reject_options
from imagehub.training.flow_matching import FlowMatchingObjective

if TYPE_CHECKING:
    from imagehub.training.arguments import TrainingArguments


# argparse defaults in train_dreambooth_lora_qwen_image.py and its 2.1 counterpart.
WEIGHTING_SCHEME = "none"
LOGIT_MEAN = 0.0
LOGIT_STD = 1.0
MODE_SCALE = 1.29
ORIGINAL_MAX_SEQUENCE_LENGTH = 512


def _pack_original(latents: Tensor) -> Tensor:
    """``QwenImagePipeline._pack_latents`` for the original single-frame VAE."""
    batch, channels, frames, height, width = latents.shape
    latents = latents.permute(0, 2, 1, 3, 4)
    latents = latents.view(batch, frames, channels, height // 2, 2, width // 2, 2)
    return latents.permute(0, 3, 5, 2, 1, 4, 6).reshape(
        batch, (height // 2) * (width // 2), channels * frames * 4
    )


def _unpack_original(
    latents: Tensor, *, channels: int, frames: int, height: int, width: int
) -> Tensor:
    batch = latents.shape[0]
    latents = latents.view(batch, height // 2, width // 2, channels, frames, 2, 2)
    return latents.permute(0, 3, 4, 1, 5, 2, 6).reshape(batch, channels, frames, height, width)


class QwenImageCollator:
    """The scripts' resize/crop/flip transform plus each release's chat template."""

    def __init__(
        self,
        components: OriginalQwenImageComponents | QwenImageComponents,
        *,
        original: bool,
        resolution: int,
        center_crop: bool,
        random_flip: bool,
        generator: torch.Generator | None,
    ) -> None:
        self.components = components
        self.original = original
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.generator = generator

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
        pixels, token_rows = [], []
        for feature in features:
            image, _, _ = preprocess_image_xl(
                feature["image"],
                resolution=self.resolution,
                center_crop=self.center_crop,
                random_flip=self.random_flip,
                generator=self.generator,
            )
            pixels.append(image)
            template = _ORIGINAL_PROMPT_TEMPLATE if self.original else _PROMPT_TEMPLATE
            ids = self.components.tokenizer.tokenize(template.format(feature["text"]))
            if self.original:
                ids = ids[: _ORIGINAL_DROP_INDEX + ORIGINAL_MAX_SEQUENCE_LENGTH]
            token_rows.append(ids)
        width = max(map(len, token_rows))
        pad_id = self.components.tokenizer.pad_id
        input_ids = torch.tensor(
            [
                ids + [pad_id] * (width - len(ids))
                if self.original
                else [pad_id] * (width - len(ids)) + ids
                for ids in token_rows
            ]
        )
        attention_mask = input_ids != pad_id
        return {
            "pixel_values": torch.stack(pixels),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class QwenImageLoRAObjective(FlowMatchingObjective):
    """The Qwen scripts' VAE normalization, packing and flow-matching loss."""

    def __init__(
        self,
        components: OriginalQwenImageComponents | QwenImageComponents,
        *,
        original: bool,
        **options: Any,
    ) -> None:
        super().__init__(components.scheduler_config, precondition_outputs=False, **options)
        self.components = components
        self.original = original
        self.vae_scale_factor = 8 if original else 16

    def latent_shape(self, pixels: Tensor) -> tuple[int, ...]:
        batch, _, height, width = pixels.shape
        channels = int(self.components.vae.config["z_dim"])
        if self.original:
            return (
                batch,
                channels,
                1,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
        return batch, channels, 1, height // self.vae_scale_factor, width // self.vae_scale_factor

    @torch.no_grad()
    def encode_images(self, pixels: Tensor, posterior_noise: Tensor) -> Tensor:
        vae = self.components.vae
        dtype = next(vae.parameters()).dtype
        if self.original:
            pixels = pixels.unsqueeze(2)
        elif pixels.shape[1] < int(vae.config.get("in_channels", pixels.shape[1])):
            missing = int(vae.config["in_channels"]) - pixels.shape[1]
            pixels = torch.cat(
                [pixels, pixels.new_zeros(pixels.shape[0], missing, *pixels.shape[2:])], dim=1
            )
        moments = vae.encode_moments(pixels.to(dtype))
        mean, logvar = moments.chunk(2, dim=1)
        latents = mean + torch.exp(0.5 * logvar.clamp(-30.0, 20.0)) * posterior_noise.to(mean)
        channels = mean.shape[1]
        means = vae.latents_mean or [0.0] * channels
        stds = vae.latents_std or [1.0] * channels
        mean_value = torch.tensor(means, dtype=latents.dtype, device=latents.device).view(
            1, channels, 1, 1, 1
        )
        std_value = torch.tensor(stds, dtype=latents.dtype, device=latents.device).view(
            1, channels, 1, 1, 1
        )
        latents = (latents - mean_value) * std_value.reciprocal()
        return latents.to(next(self.components.transformer.parameters()).dtype)

    @torch.no_grad()
    def encode_condition(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor | None]:
        hidden = self.components.text_encoder(batch["input_ids"], batch["attention_mask"])[-1]
        mask = batch["attention_mask"]
        if self.original:
            hidden = hidden[
                :, _ORIGINAL_DROP_INDEX : _ORIGINAL_DROP_INDEX + ORIGINAL_MAX_SEQUENCE_LENGTH
            ]
            mask = mask[
                :, _ORIGINAL_DROP_INDEX : _ORIGINAL_DROP_INDEX + ORIGINAL_MAX_SEQUENCE_LENGTH
            ]
        else:
            drop = len(self.components.tokenizer.tokenize(_SYSTEM_TEMPLATE))
            rows = [row[row_mask][drop:] for row, row_mask in zip(hidden, mask, strict=True)]
            width = max(row.shape[0] for row in rows)
            hidden = torch.stack(
                [
                    torch.cat([row, row.new_zeros(width - row.shape[0], row.shape[1])])
                    for row in rows
                ]
            )
            mask = torch.stack(
                [
                    torch.cat(
                        [
                            row.new_ones(row.shape[0]),
                            row.new_zeros(width - row.shape[0]),
                        ]
                    )
                    for row in rows
                ]
            ).bool()
        return {
            "prompt_embeds": hidden.to(next(self.components.transformer.parameters()).dtype),
            "prompt_mask": None if mask.all() else mask,
        }

    def predict(self, noisy: Tensor, timesteps: Tensor, condition: Any) -> Tensor:
        prompt = cast(Tensor, condition["prompt_embeds"])
        prompt_mask = cast(Tensor | None, condition["prompt_mask"])
        batch = noisy.shape[0]
        if self.original:
            _, channels, frames, height, width = noisy.shape
            packed = _pack_original(noisy)
            output = self.components.transformer(
                packed,
                prompt,
                timesteps / 1000,
                [[(1, height // 2, width // 2)]] * batch,
                encoder_hidden_states_mask=prompt_mask,
            )
            return _unpack_original(
                output, channels=channels, frames=frames, height=height, width=width
            )
        _, channels, frames, height, width = noisy.shape
        if frames != 1:
            raise ValueError("Qwen-Image 2.1 training accepts one-frame latents only")
        packed = noisy[:, :, 0].view(batch, channels, height * width).transpose(1, 2)
        target_slots = (height * width) // 4
        image_mask = torch.cat(
            [
                torch.zeros(batch, prompt.shape[1], dtype=torch.bool, device=noisy.device),
                torch.ones(batch, target_slots, dtype=torch.bool, device=noisy.device),
            ],
            dim=1,
        )
        output = self.components.transformer(
            packed,
            prompt,
            timesteps / 1000,
            [[(1, height, width)]] * batch,
            image_mask,
            encoder_hidden_states_mask=prompt_mask,
        )
        output = output[:, -packed.shape[1] :]
        return output.transpose(1, 2).reshape(batch, channels, 1, height, width)


class QwenImageFamily(TrainingFamily):
    name = "Qwen-Image"
    prefix = "transformer"
    batch_keys = ("pixel_values", "input_ids", "attention_mask")
    checkpoint_blocks = (QwenImageTransformerBlock, QwenImage21TransformerBlock)

    @staticmethod
    def _original(components: Any) -> bool:
        return isinstance(components, OriginalQwenImageComponents)

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
    ) -> Any:
        components_type = (
            OriginalQwenImageComponents if model_id == "qwen-image" else QwenImageComponents
        )
        return components_type.load(checkpoint, device=device, dtype=dtype, model_id=model_id)

    def trainable(self, components: Any) -> nn.Module:
        return components.transformer

    def frozen(self, components: Any) -> list[nn.Module]:
        return [components.text_encoder, components.vae]

    def collator(
        self, components: Any, args: TrainingArguments, *, generator: torch.Generator, train: bool
    ) -> Collator:
        return QwenImageCollator(
            components,
            original=self._original(components),
            resolution=args.resolution,
            center_crop=args.center_crop if train else True,
            random_flip=args.random_flip if train else False,
            generator=generator,
        )

    def objective(self, components: Any, args: TrainingArguments) -> QwenImageLoRAObjective:
        def pick(value: Any, default: Any) -> Any:
            return default if value is None else value

        if not self._original(components) and args.max_sequence_length is not None:
            raise ValueError("max_sequence_length does not apply to Qwen-Image 2.1 training")
        return QwenImageLoRAObjective(
            components,
            original=self._original(components),
            weighting_scheme=pick(args.weighting_scheme, WEIGHTING_SCHEME),
            logit_mean=pick(args.logit_mean, LOGIT_MEAN),
            logit_std=pick(args.logit_std, LOGIT_STD),
            mode_scale=pick(args.mode_scale, MODE_SCALE),
        )

    def sample(
        self, components: Any, prompt: str, args: TrainingArguments, device: torch.device
    ) -> Tensor:
        original = self._original(components)
        generator = torch.Generator("cpu").manual_seed(args.seed)
        if original:
            channels = int(components.vae.config["z_dim"])
            raw = torch.randn(
                (1, 1, channels, args.resolution // 8, args.resolution // 8), generator=generator
            )
            latents = _pack_original(raw).to(device)
            final = denoise_original(
                components,
                prompt=prompt,
                negative_prompt=None,
                steps=args.validation_num_inference_steps,
                true_cfg=1.0,
                width=args.resolution,
                height=args.resolution,
                latents=latents,
            )
            return decode_original_latents(components.vae, final, args.resolution, args.resolution)[
                0
            ]
        channels = int(components.vae.config["z_dim"])
        latents = (
            torch.randn((1, channels, (args.resolution // 16) ** 2), generator=generator)
            .transpose(1, 2)
            .to(device)
        )
        final = denoise(
            components,
            prompt=prompt,
            negative_prompt=None,
            steps=args.validation_num_inference_steps,
            true_cfg=1.0,
            width=args.resolution,
            height=args.resolution,
            latents=latents,
        )
        return decode_latents(components.vae, final, args.resolution, args.resolution)[0]
