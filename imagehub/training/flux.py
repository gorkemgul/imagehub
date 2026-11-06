"""FLUX.1 LoRA training after diffusers' ``train_dreambooth_lora_flux.py``."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

from imagehub.backends.flux import (
    FluxComponents,
    _pack_latents,
    _prepare_latent_image_ids,
    _unpack_latents,
    decode_latents,
    denoise,
)
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.flux_transformer import FluxSingleTransformerBlock, FluxTransformerBlock
from imagehub.training.data import preprocess_image_xl
from imagehub.training.family import DDPM_OPTIONS, Collator, TrainingFamily, reject_options
from imagehub.training.flow_matching import FlowMatchingObjective

if TYPE_CHECKING:
    from imagehub.training.arguments import TrainingArguments


#: The script's defaults.
WEIGHTING_SCHEME = "none"
LOGIT_MEAN = 0.0
LOGIT_STD = 1.0
MODE_SCALE = 1.29
GUIDANCE_SCALE = 3.5
MAX_SEQUENCE_LENGTH = 512


class Flux1Collator:
    """The FLUX script's image transform, CLIP ids and T5 ids."""

    def __init__(
        self,
        components: FluxComponents,
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
        pixels, clip_ids, t5_ids = [], [], []
        for feature in features:
            image, _, _ = preprocess_image_xl(
                feature["image"],
                resolution=self.resolution,
                center_crop=self.center_crop,
                random_flip=self.random_flip,
                generator=self.generator,
            )
            pixels.append(image)
            clip_ids.append(self.components.tokenizer(feature["text"]))
            ids, _ = self.components.tokenizer_2([feature["text"]], self.max_sequence_length)
            t5_ids.append(ids[0])
        return {
            "pixel_values": torch.stack(pixels),
            "input_ids": torch.tensor(clip_ids),
            "input_ids_2": torch.stack(t5_ids),
        }


class Flux1LoRAObjective(FlowMatchingObjective):
    """Packed FLUX latents, FLUX position ids and the un-preconditioned flow target."""

    def __init__(
        self, components: FluxComponents, *, guidance_scale: float, **options: Any
    ) -> None:
        super().__init__(components.scheduler_config, precondition_outputs=False, **options)
        vae = components.vae
        if not isinstance(vae, AutoencoderKL):
            raise ValueError("Training needs a KL autoencoder to sample latents from")
        self.components = components
        self.guidance_scale = guidance_scale
        self.latent_channels = int(vae.config["latent_channels"])
        self.downsample = 2 ** (len(vae.config["block_out_channels"]) - 1)

    def latent_shape(self, pixels: Tensor) -> tuple[int, ...]:
        batch, _, height, width = pixels.shape
        return (batch, self.latent_channels, height // self.downsample, width // self.downsample)

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
        input_ids = batch["input_ids"]
        hidden_states = cast(Tensor, components.text_encoder(input_ids))
        eos_token_id = int(components.text_encoder.config["eos_token_id"])
        if eos_token_id == 2:
            positions = input_ids.to(dtype=torch.int).argmax(dim=-1)
        else:
            positions = (input_ids.to(dtype=torch.int) == eos_token_id).int().argmax(dim=-1)
        pooled = hidden_states[
            torch.arange(hidden_states.shape[0], device=input_ids.device), positions
        ]
        prompt_embeds = components.text_encoder_2(batch["input_ids_2"])[0]
        prompt_embeds = prompt_embeds.to(components.text_encoder_2.shared.weight.dtype)
        text_ids = torch.zeros(
            prompt_embeds.shape[1], 3, device=prompt_embeds.device, dtype=prompt_embeds.dtype
        )
        return {"prompt_embeds": prompt_embeds, "pooled_projections": pooled, "text_ids": text_ids}

    def predict(self, noisy: Tensor, timesteps: Tensor, condition: Any) -> Tensor:
        batch, channels, height, width = noisy.shape
        packed = _pack_latents(noisy, batch, channels, height, width)
        image_ids = _prepare_latent_image_ids(
            batch, height // 2, width // 2, noisy.device, noisy.dtype
        )
        guidance = None
        if self.components.transformer.config["guidance_embeds"]:
            guidance = torch.full((batch,), self.guidance_scale, device=noisy.device)
        prediction = self.components.transformer(
            packed,
            condition["prompt_embeds"],
            condition["pooled_projections"],
            timesteps / 1000,
            image_ids,
            condition["text_ids"],
            guidance,
        )
        return _unpack_latents(
            prediction, height * self.downsample, width * self.downsample, self.downsample
        )


class Flux1Family(TrainingFamily):
    """FLUX.1 and text-only Kontext LoRA training."""

    name = "FLUX.1"
    prefix = "transformer"
    batch_keys = ("pixel_values", "input_ids", "input_ids_2")
    checkpoint_blocks = (FluxTransformerBlock, FluxSingleTransformerBlock)

    def validate(self, args: TrainingArguments) -> None:
        reject_options(args, DDPM_OPTIONS + ("precondition_outputs",), self.name)

    def load(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str,
    ) -> FluxComponents:
        components = FluxComponents.load(checkpoint, device=device, dtype=dtype, model_id=model_id)
        components.vae.to(torch.float32)
        return components

    def trainable(self, components: Any) -> nn.Module:
        return components.transformer

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
        return Flux1Collator(
            components,
            resolution=args.resolution,
            center_crop=args.center_crop if train else True,
            random_flip=args.random_flip if train else False,
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
            generator=generator,
        )

    def objective(self, components: Any, args: TrainingArguments) -> Flux1LoRAObjective:
        def pick(value: Any, default: Any) -> Any:
            return default if value is None else value

        return Flux1LoRAObjective(
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
        shape = (
            1,
            int(vae.config["latent_channels"]),
            args.resolution // downsample,
            args.resolution // downsample,
        )
        latents = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(args.seed))
        final = denoise(
            components,
            prompt=prompt,
            negative_prompt=None,
            true_cfg_scale=1.0,
            steps=args.validation_num_inference_steps,
            guidance=args.validation_guidance_scale,
            width=args.resolution,
            height=args.resolution,
            latents=latents.to(device),
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
        )
        return decode_latents(vae, final, args.resolution, args.resolution)[0]
