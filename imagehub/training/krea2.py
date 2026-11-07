"""Krea 2 LoRA training after diffusers' DreamBooth script."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from imagehub.backends.krea2 import (
    _PROMPT_PREFIX,
    _PROMPT_START,
    _PROMPT_SUFFIX,
    _PROMPT_SUFFIX_TOKENS,
    Krea2Components,
    decode_latents,
    denoise,
    prepare_position_ids,
)
from imagehub.models.krea2_transformer import Krea2TransformerBlock
from imagehub.training.data import preprocess_image
from imagehub.training.family import DDPM_OPTIONS, Collator, TrainingFamily, reject_options
from imagehub.training.flow_matching import FlowMatchingObjective, sample_density

if TYPE_CHECKING:
    from imagehub.training.arguments import TrainingArguments
    from imagehub.training.specs import TrainingSpec


#: ``train_dreambooth_lora_krea2.py`` argparse defaults.
WEIGHTING_SCHEME = "none"
LOGIT_MEAN = 0.0
LOGIT_STD = 1.0
MODE_SCALE = 1.29
MAX_SEQUENCE_LENGTH = 512
VALIDATION_STEPS = 8
VALIDATION_GUIDANCE = 0.0


def _pack_latents(latents: Tensor) -> Tensor:
    """The script's inlined ``Krea2Pipeline._pack_latents`` for its 5-D VAE output."""
    batch, channels, frames, height, width = latents.shape
    latents = latents.permute(0, 2, 1, 3, 4)
    latents = latents.view(batch, frames, channels, height // 2, 2, width // 2, 2)
    return latents.permute(0, 3, 5, 2, 1, 4, 6).reshape(
        batch, (height // 2) * (width // 2), channels * frames * 4
    )


def _unpack_latents(latents: Tensor, height: int, width: int) -> Tensor:
    """The script's inlined ``Krea2Pipeline._unpack_latents`` for one latent frame."""
    batch, _, channels = latents.shape
    latents = latents.view(batch, height // 2, width // 2, channels // 4, 2, 2)
    return latents.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels // 4, 1, height, width)


class Krea2Collator:
    """The DreamBooth resize/crop/flip transform and Krea prompt template."""

    def __init__(
        self,
        components: Krea2Components,
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
            pixels.append(
                preprocess_image(
                    feature["image"],
                    resolution=self.resolution,
                    center_crop=self.center_crop,
                    random_flip=self.random_flip,
                    generator=self.generator,
                )
            )
            prompts.append(_PROMPT_PREFIX + feature["text"])
        input_ids, attention_mask = self.components.tokenizer(
            prompts, self.max_sequence_length + _PROMPT_START - _PROMPT_SUFFIX_TOKENS
        )
        suffix = torch.tensor(
            [self.components.tokenizer.tokenize(_PROMPT_SUFFIX)], dtype=torch.long
        )
        suffix = suffix.expand(input_ids.shape[0], -1)
        return {
            "pixel_values": torch.stack(pixels),
            "input_ids": torch.cat([input_ids, suffix], dim=1),
            "attention_mask": torch.cat(
                [attention_mask, torch.ones_like(suffix, dtype=torch.bool)], dim=1
            ),
        }


class Krea2LoRAObjective(FlowMatchingObjective):
    """Krea 2's posterior sample, normalization, patch layout and flow target."""

    def __init__(self, components: Krea2Components, **options: Any) -> None:
        super().__init__(components.scheduler_config, precondition_outputs=False, **options)
        self.components = components
        self.channels = int(components.vae.config["z_dim"])

    def latent_shape(self, pixels: Tensor) -> tuple[int, ...]:
        batch, _, height, width = pixels.shape
        return (
            batch,
            self.channels,
            1,
            height // self.components.vae_scale_factor,
            width // self.components.vae_scale_factor,
        )

    @torch.no_grad()
    def encode_images(self, pixels: Tensor, posterior_noise: Tensor) -> Tensor:
        vae = self.components.vae
        moments = vae.encode_moments(pixels.unsqueeze(2).to(next(vae.parameters()).dtype))
        mean, logvar = moments.chunk(2, dim=1)
        latents = mean + torch.exp(0.5 * logvar.clamp(-30.0, 20.0)) * posterior_noise.to(mean)
        mean_value = torch.tensor(
            vae.latents_mean, device=latents.device, dtype=latents.dtype
        ).view(1, -1, 1, 1, 1)
        inverse_std = (
            torch.tensor(vae.latents_std, device=latents.device, dtype=latents.dtype)
            .reciprocal()
            .view(1, -1, 1, 1, 1)
        )
        return ((latents - mean_value) * inverse_std).to(
            next(self.components.transformer.parameters()).dtype
        )

    @torch.no_grad()
    def encode_condition(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        mask = batch["attention_mask"]
        positions = (mask.long().cumsum(dim=-1) - 1).clamp(min=0).unsqueeze(0).expand(3, -1, -1)
        hidden = self.components.text_encoder(
            batch["input_ids"], mask, position_ids=positions, output_hidden_states=True
        )
        selected = torch.stack(
            [hidden[index] for index in self.components.text_encoder_select_layers], dim=2
        )[:, _PROMPT_START:]
        return {
            "prompt_embeds": selected.to(next(self.components.transformer.parameters()).dtype),
            "prompt_mask": mask[:, _PROMPT_START:],
        }

    def draw(self, batch: Mapping[str, Tensor], generator: torch.Generator) -> dict[str, Tensor]:
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

    def predict(self, noisy: Tensor, timesteps: Tensor, condition: Any) -> Tensor:
        _, _, _, height, width = noisy.shape
        packed = _pack_latents(noisy)
        position_ids = prepare_position_ids(
            condition["prompt_embeds"].shape[1], height // 2, width // 2, noisy.device
        )
        prediction = self.components.transformer(
            packed,
            condition["prompt_embeds"],
            timesteps / 1000,
            position_ids,
            condition["prompt_mask"],
        )
        return _unpack_latents(prediction, height, width)


class Krea2Family(TrainingFamily):
    """Krea 2 transformer-only LoRA training."""

    name = "Krea 2"
    prefix = "transformer"
    batch_keys = ("pixel_values", "input_ids", "attention_mask")
    checkpoint_blocks = (Krea2TransformerBlock,)

    def check_checkpoint(self, checkpoint: os.PathLike[str], spec: TrainingSpec) -> None:
        """Reject Turbo before opening any component weights.

        Krea2Components.load requires this index, so missing metadata must be surfaced here rather
        than deferred until after the training setup has begun.
        """
        index = os.fspath(checkpoint)
        try:
            with open(os.path.join(index, "model_index.json"), encoding="utf-8") as source:
                model_index = json.load(source)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"{error.filename} is missing from the Krea 2 checkpoint"
            ) from error
        if bool(model_index.get("is_distilled", False)):
            assert spec.training_repo is not None and spec.training_revision is not None
            raise ValueError(
                "Krea 2 Turbo is distilled and cannot be trained. Download the training checkpoint "
                f"{spec.training_repo} at {spec.training_revision} instead."
            )

    def validate(self, args: TrainingArguments) -> None:
        reject_options(
            args,
            DDPM_OPTIONS
            + (
                "precondition_outputs",
                "complex_human_instruction",
                "system_prompt",
                "guidance_scale",
            ),
            self.name,
        )

    def load(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str,
    ) -> Krea2Components:
        return Krea2Components.load(checkpoint, device=device, dtype=dtype, model_id=model_id)

    def trainable(self, components: Any) -> nn.Module:
        return components.transformer

    def frozen(self, components: Any) -> list[nn.Module]:
        return [components.text_encoder, components.vae]

    def collator(
        self, components: Any, args: TrainingArguments, *, generator: torch.Generator, train: bool
    ) -> Collator:
        return Krea2Collator(
            components,
            resolution=args.resolution,
            center_crop=args.center_crop if train else True,
            random_flip=args.random_flip if train else False,
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
            generator=generator,
        )

    def objective(self, components: Any, args: TrainingArguments) -> Krea2LoRAObjective:
        def pick(value: Any, default: Any) -> Any:
            return default if value is None else value

        return Krea2LoRAObjective(
            components,
            weighting_scheme=pick(args.weighting_scheme, WEIGHTING_SCHEME),
            logit_mean=pick(args.logit_mean, LOGIT_MEAN),
            logit_std=pick(args.logit_std, LOGIT_STD),
            mode_scale=pick(args.mode_scale, MODE_SCALE),
        )

    def sample(
        self, components: Any, prompt: str, args: TrainingArguments, device: torch.device
    ) -> Tensor:
        channels = int(components.transformer.config["in_channels"])
        side = args.resolution // (components.vae_scale_factor * components.patch_size)
        latents = torch.randn(
            (1, side * side, channels), generator=torch.Generator("cpu").manual_seed(args.seed)
        )
        final = denoise(
            components,
            prompt=prompt,
            negative_prompt=None,
            steps=VALIDATION_STEPS
            if args.validation_num_inference_steps == 25
            else args.validation_num_inference_steps,
            guidance=VALIDATION_GUIDANCE
            if args.validation_guidance_scale == 7.5
            else args.validation_guidance_scale,
            width=args.resolution,
            height=args.resolution,
            latents=latents.to(device),
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
        )
        return decode_latents(components, final, args.resolution, args.resolution)[0]
