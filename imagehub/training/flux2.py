"""FLUX.2 klein LoRA training after diffusers' DreamBooth script."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from imagehub.backends.flux2 import Flux2Components, decode_latents, denoise
from imagehub.models.autoencoder_kl_flux2 import (
    normalize_latents,
    patchify_latents,
)
from imagehub.models.flux2_transformer import Flux2SingleTransformerBlock, Flux2TransformerBlock
from imagehub.training.data import preprocess_image
from imagehub.training.family import DDPM_OPTIONS, Collator, TrainingFamily, reject_options
from imagehub.training.flow_matching import FlowMatchingObjective, sample_density

if TYPE_CHECKING:
    from imagehub.training.arguments import TrainingArguments


#: ``train_dreambooth_lora_flux2_klein.py`` argparse defaults.
WEIGHTING_SCHEME = "none"
LOGIT_MEAN = 0.0
LOGIT_STD = 1.0
MODE_SCALE = 1.29
GUIDANCE_SCALE = 3.5
MAX_SEQUENCE_LENGTH = 512
TEXT_ENCODER_OUT_LAYERS = (10, 20, 30)


def _unpack_latents_with_ids(tokens: Tensor, ids: Tensor, height: int, width: int) -> Tensor:
    """``Flux2KleinPipeline._unpack_latents_with_ids`` with known spatial dimensions."""
    unpacked = []
    for data, positions in zip(tokens, ids, strict=True):
        channels = data.shape[-1]
        flat_ids = positions[:, 1].long() * width + positions[:, 2].long()
        image = torch.zeros((height * width, channels), device=data.device, dtype=data.dtype)
        image.scatter_(0, flat_ids.unsqueeze(1).expand(-1, channels), data)
        unpacked.append(image.view(height, width, channels).permute(2, 0, 1))
    return torch.stack(unpacked)


class Flux2Collator:
    """The script's fixed-size resize/crop/flip transform and Qwen chat prompt."""

    def __init__(
        self,
        components: Flux2Components,
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
            prompts.append(
                self.components.tokenizer.chat_prompt(feature["text"], enable_thinking=False)
            )
        input_ids, attention_mask = self.components.tokenizer(prompts, self.max_sequence_length)
        return {
            "pixel_values": torch.stack(pixels),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class Flux2LoRAObjective(FlowMatchingObjective):
    """The FLUX.2 klein deterministic posterior, packed latents and flow target."""

    def __init__(
        self, components: Flux2Components, *, guidance_scale: float, **options: Any
    ) -> None:
        super().__init__(components.scheduler_config, precondition_outputs=False, **options)
        self.components = components
        self.guidance_scale = guidance_scale
        self.latent_channels = int(components.vae.config["latent_channels"])
        self.downsample = 2 ** (len(components.vae.config["block_out_channels"]) - 1)

    def latent_shape(self, pixels: Tensor) -> tuple[int, ...]:
        batch, _, height, width = pixels.shape
        return (
            batch,
            self.latent_channels * 4,
            height // (self.downsample * 2),
            width // (self.downsample * 2),
        )

    def draw(self, batch: Mapping[str, Tensor], generator: torch.Generator) -> dict[str, Tensor]:
        """The script does not draw posterior noise before drawing flow noise."""
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
        latents = patchify_latents(latents)
        assert vae.bn.running_mean is not None and vae.bn.running_var is not None
        latents = normalize_latents(
            latents, vae.bn.running_mean, vae.bn.running_var, vae.batch_norm_eps
        )
        return latents.to(next(self.components.transformer.parameters()).dtype)

    @torch.no_grad()
    def encode_condition(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        components = self.components
        _, hidden_states = components.text_encoder(
            batch["input_ids"], batch["attention_mask"], output_hidden_states=True
        )
        selected = torch.stack([hidden_states[index] for index in TEXT_ENCODER_OUT_LAYERS], dim=1)
        batch_size, channels, sequence_length, hidden_size = selected.shape
        prompt_embeds = selected.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, channels * hidden_size
        )
        text_ids = (
            torch.cartesian_prod(
                torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(sequence_length)
            )
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        return {
            "prompt_embeds": prompt_embeds.to(next(components.transformer.parameters()).dtype),
            "text_ids": text_ids.to(batch["input_ids"].device),
        }

    def predict(self, noisy: Tensor, timesteps: Tensor, condition: Any) -> Tensor:
        batch, _, height, width = noisy.shape
        packed = noisy.reshape(batch, noisy.shape[1], height * width).permute(0, 2, 1)
        image_ids = (
            torch.cartesian_prod(
                torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(1)
            )
            .unsqueeze(0)
            .expand(batch, -1, -1)
            .to(noisy.device)
        )
        guidance = None
        if self.components.transformer.config["guidance_embeds"]:
            guidance = torch.full((batch,), self.guidance_scale, device=noisy.device)
        prediction = self.components.transformer(
            packed,
            condition["prompt_embeds"],
            timesteps / 1000,
            image_ids,
            condition["text_ids"],
            guidance,
        )
        prediction = prediction[:, : packed.shape[1] :]
        return _unpack_latents_with_ids(prediction, image_ids, height, width)

    def loss_from(self, batch: Mapping[str, Tensor], draws: Mapping[str, Tensor]) -> Tensor:
        return self.loss(
            self.encode_images(batch["pixel_values"]), self.encode_condition(batch), draws
        )


class Flux2Family(TrainingFamily):
    """FLUX.2 klein transformer-only LoRA training."""

    name = "FLUX.2 klein"
    prefix = "transformer"
    batch_keys = ("pixel_values", "input_ids", "attention_mask")
    checkpoint_blocks = (Flux2TransformerBlock, Flux2SingleTransformerBlock)

    def validate(self, args: TrainingArguments) -> None:
        reject_options(
            args,
            DDPM_OPTIONS + ("precondition_outputs", "complex_human_instruction", "system_prompt"),
            self.name,
        )

    def load(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str,
    ) -> Flux2Components:
        return Flux2Components.load(checkpoint, device=device, dtype=dtype, model_id=model_id)

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
        return Flux2Collator(
            components,
            resolution=args.resolution,
            center_crop=args.center_crop if train else True,
            random_flip=args.random_flip if train else False,
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
            generator=generator,
        )

    def objective(self, components: Any, args: TrainingArguments) -> Flux2LoRAObjective:
        if (
            not components.transformer.config["guidance_embeds"]
            and args.guidance_scale != GUIDANCE_SCALE
        ):
            raise ValueError(
                "guidance_scale does not apply to FLUX.2 klein checkpoints without guidance"
            )

        def pick(value: Any, default: Any) -> Any:
            return default if value is None else value

        return Flux2LoRAObjective(
            components,
            weighting_scheme=pick(args.weighting_scheme, WEIGHTING_SCHEME),
            logit_mean=pick(args.logit_mean, LOGIT_MEAN),
            logit_std=pick(args.logit_std, LOGIT_STD),
            mode_scale=pick(args.mode_scale, MODE_SCALE),
            guidance_scale=args.guidance_scale,
        )

    def sample(
        self, components: Any, prompt: str, args: TrainingArguments, device: torch.device
    ) -> Tensor:
        vae = components.vae
        downsample = 2 ** (len(vae.config["block_out_channels"]) - 1)
        latents = torch.randn(
            (
                1,
                int(components.transformer.config["in_channels"]),
                args.resolution // (downsample * 2),
                args.resolution // (downsample * 2),
            ),
            generator=torch.Generator("cpu").manual_seed(args.seed),
            dtype=torch.float32,
        )
        final = denoise(
            components,
            prompt=prompt,
            steps=args.validation_num_inference_steps,
            guidance=args.validation_guidance_scale,
            width=args.resolution,
            height=args.resolution,
            latents=latents.to(device),
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
        )
        return decode_latents(vae, final)[0]
