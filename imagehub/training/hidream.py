"""HiDream-I1 LoRA training after diffusers' DreamBooth script."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from imagehub.backends.hidream import HiDreamComponents, decode_latents, denoise, effective_size
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.hidream_transformer import BlockWrapper, HiDreamImageTransformer2DModel
from imagehub.models.llama_text import LlamaTextModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.training.data import preprocess_image_xl
from imagehub.training.family import DDPM_OPTIONS, Collator, TrainingFamily, reject_options
from imagehub.training.flow_matching import FlowMatchingObjective

if TYPE_CHECKING:
    from imagehub.training.arguments import TrainingArguments


#: ``train_dreambooth_lora_hidream.py`` argparse defaults.
WEIGHTING_SCHEME = "none"
LOGIT_MEAN = 0.0
LOGIT_STD = 1.0
MODE_SCALE = 1.29
MAX_SEQUENCE_LENGTH = 128


@dataclass
class HiDreamTrainingComponents:
    """All four frozen text encoders, VAE and trainable transformer kept resident for a step."""

    tokenizer: Any
    tokenizer_2: Any
    tokenizer_3: Any
    tokenizer_4: Any
    text_encoder: CLIPTextModelWithProjection
    text_encoder_2: CLIPTextModelWithProjection
    text_encoder_3: T5EncoderModel
    text_encoder_4: LlamaTextModel
    transformer: HiDreamImageTransformer2DModel
    vae: AutoencoderKL
    scheduler_config: dict[str, Any]
    inference_components: HiDreamComponents

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> HiDreamTrainingComponents:
        components = HiDreamComponents.load(
            directory, device=device, dtype=dtype, model_id=model_id
        )
        transformer = components._from_pretrained(HiDreamImageTransformer2DModel, "transformer")
        # The script trains with force_inference_output=True; imagehub implements no other path.
        return cls(
            tokenizer=components.tokenizer,
            tokenizer_2=components.tokenizer_2,
            tokenizer_3=components.tokenizer_3,
            tokenizer_4=components.tokenizer_4,
            text_encoder=components._from_pretrained(CLIPTextModelWithProjection, "text_encoder"),
            text_encoder_2=components._from_pretrained(
                CLIPTextModelWithProjection, "text_encoder_2"
            ),
            text_encoder_3=components._from_pretrained(T5EncoderModel, "text_encoder_3"),
            text_encoder_4=components._from_pretrained(LlamaTextModel, "text_encoder_4"),
            transformer=transformer,
            vae=components.vae,
            scheduler_config=components.scheduler_config,
            inference_components=components,
        )


class HiDreamCollator:
    """The script's resize/flip/crop transform and four tokenizer calls."""

    def __init__(
        self,
        components: HiDreamTrainingComponents,
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
            prompts.append(feature["text"])
        components = self.components
        clip_length = min(self.max_sequence_length, 218)
        components.tokenizer.max_length = clip_length
        components.tokenizer_2.max_length = clip_length
        ids = torch.tensor([components.tokenizer(prompt) for prompt in prompts])
        ids_2 = torch.tensor([components.tokenizer_2(prompt) for prompt in prompts])
        ids_3, mask_3 = components.tokenizer_3(prompts, self.max_sequence_length)
        ids_4, mask_4 = components.tokenizer_4(prompts, self.max_sequence_length)
        return {
            "pixel_values": torch.stack(pixels).contiguous(),
            "input_ids": ids,
            "input_ids_2": ids_2,
            "input_ids_3": ids_3,
            "attention_mask_3": mask_3,
            "input_ids_4": ids_4,
            "attention_mask_4": mask_4,
        }


class HiDreamLoRAObjective(FlowMatchingObjective):
    """HiDream's sampled VAE posterior and un-preconditioned flow loss."""

    def __init__(self, components: HiDreamTrainingComponents, **options: Any) -> None:
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
        dtype = next(components.transformer.parameters()).dtype
        pooled_1 = components.text_encoder(batch["input_ids"], output_hidden_states=True)[0]
        pooled_2 = components.text_encoder_2(batch["input_ids_2"], output_hidden_states=True)[0]
        t5 = components.text_encoder_3(batch["input_ids_3"], batch["attention_mask_3"])[0]
        llama = components.text_encoder_4(batch["input_ids_4"], batch["attention_mask_4"])
        # Llama's frozen encoder intentionally returns inference tensors. Transformer training
        # needs normal tensors for its linear layers to save activations for LoRA backward.
        with torch.inference_mode(False):
            llama = llama.clone()
        return {
            "pooled_embeds": torch.cat([pooled_1, pooled_2], dim=-1).to(dtype),
            "t5": t5.to(dtype),
            "llama": llama.to(dtype),
        }

    def predict(self, noisy: Tensor, timesteps: Tensor, condition: Any) -> Tensor:
        return -self.components.transformer(
            noisy,
            timesteps,
            condition["t5"],
            condition["llama"],
            condition["pooled_embeds"],
        )


class HiDreamFamily(TrainingFamily):
    """HiDream-I1 transformer-only LoRA training."""

    name = "HiDream-I1"
    prefix = "transformer"
    batch_keys = (
        "pixel_values",
        "input_ids",
        "input_ids_2",
        "input_ids_3",
        "attention_mask_3",
        "input_ids_4",
        "attention_mask_4",
    )
    checkpoint_blocks = (BlockWrapper,)

    def validate(self, args: TrainingArguments) -> None:
        reject_options(
            args,
            DDPM_OPTIONS
            + (
                "precondition_outputs",
                "guidance_scale",
                "complex_human_instruction",
                "system_prompt",
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
    ) -> HiDreamTrainingComponents:
        return HiDreamTrainingComponents.load(
            checkpoint, device=device, dtype=dtype, model_id=model_id
        )

    def trainable(self, components: Any) -> nn.Module:
        return components.transformer

    def frozen(self, components: Any) -> list[nn.Module]:
        return [
            components.text_encoder,
            components.text_encoder_2,
            components.text_encoder_3,
            components.text_encoder_4,
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
        return HiDreamCollator(
            components,
            resolution=args.resolution,
            center_crop=args.center_crop if train else True,
            random_flip=args.random_flip if train else False,
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
            generator=generator,
        )

    def objective(self, components: Any, args: TrainingArguments) -> HiDreamLoRAObjective:
        def pick(value: Any, default: Any) -> Any:
            return default if value is None else value

        return HiDreamLoRAObjective(
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
        height, width = effective_size(args.resolution, args.resolution, downsample)
        latents = torch.randn(
            (
                1,
                int(vae.config["latent_channels"]),
                height // downsample,
                width // downsample,
            ),
            generator=torch.Generator("cpu").manual_seed(args.seed),
        )
        final = denoise(
            components.inference_components,
            prompt=prompt,
            negative_prompt=None,
            steps=args.validation_num_inference_steps,
            guidance=args.validation_guidance_scale,
            latents=latents.to(device),
            max_sequence_length=args.max_sequence_length or MAX_SEQUENCE_LENGTH,
            noise_generator=torch.Generator("cpu").manual_seed(args.seed),
            transformer=components.transformer,
        )
        return decode_latents(vae, final)[0]
