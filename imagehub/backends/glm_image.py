"""GLM-Image text-to-image backend over a local diffusers-layout snapshot.

Generation has two stages. The autoregressive model samples discrete image tokens from the
prompt (a low-resolution preview first, then the full-size grid); the diffusion transformer then
denoises VAE latents conditioned on those tokens and on ByT5 embeddings of the text that the
prompt quotes. The stages are loaded one after the other and released in between, because the
autoregressive model and the transformer together exceed the memory of most machines.
"""

from __future__ import annotations

import gc
import math
import os
import re
import secrets
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.cogview4 import calculate_shift
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.glm_image_ar import GlmImageARModel
from imagehub.models.glm_image_transformer import GlmImageTransformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_byt5 import ByT5Tokenizer
from imagehub.tokenization_glm_image import GlmImageTokenizer, build_prior_prompt
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = (
    "processor",
    "tokenizer",
    "text_encoder",
    "vision_language_encoder",
    "transformer",
    "vae",
    "scheduler",
)
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 50,
    "guidance": 1.5,
    "width": 1024,
    "height": 1024,
    "seed": None,
    "max_sequence_length": 2048,
    "latents": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_MULTIPLE = 32


def glm_image_sigmas(steps: int, num_train_timesteps: int) -> list[float]:
    """Sigmas the pipeline derives: truncated linspace timesteps in float32, over the train steps."""
    step = (1.0 - num_train_timesteps) / steps
    timesteps = [int(index * step + num_train_timesteps) for index in range(steps)]
    return (torch.tensor(timesteps, dtype=torch.float32) / num_train_timesteps).tolist()


def glyph_texts(prompt: str) -> list[str]:
    """Quoted passages of the prompt, which the glyph encoder renders into the image."""
    return (
        re.findall(r"'([^']*)'", prompt)
        + re.findall(r"“([^“”]*)”", prompt)
        + re.findall(r'"([^"]*)"', prompt)
        + re.findall(r"「([^「」]*)」", prompt)
    )


@torch.inference_mode()
def glyph_embeds(
    tokenizer: ByT5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: str,
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    texts = glyph_texts(prompt) or [""]
    input_ids = [tokenizer.tokenize(text, max_sequence_length) for text in texts]
    # The reference pads every row on the left when the number of passages is even.
    pad = tokenizer.pad_token_id
    input_ids = [[pad] * ((len(input_ids) + 1) % 2) + ids for ids in input_ids]
    length = max(len(ids) for ids in input_ids)
    attention_mask = torch.tensor(
        [[1] * len(ids) + [0] * (length - len(ids)) for ids in input_ids], device=device
    )
    ids = torch.tensor([ids + [pad] * (length - len(ids)) for ids in input_ids], device=device)
    hidden_states = text_encoder(ids, attention_mask=attention_mask)[0]
    return hidden_states[attention_mask.bool()].unsqueeze(0).to(device=device, dtype=dtype)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "glm-image"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


@dataclass
class GenerationSettings:
    temperature: float | None
    top_k: int | None
    top_p: float | None

    @classmethod
    def read(cls, directory: Path) -> GenerationSettings:
        config = read_json(directory / "generation_config.json")
        return cls(
            temperature=config.get("temperature"),
            top_k=config.get("top_k"),
            top_p=config.get("top_p"),
        )


class GlmImageBackend(TextToImageBackend):
    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        model_id: str | None = None,
    ) -> None:
        super().__init__()
        self.source = Path(model)
        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = dtype or torch.float32
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be torch.float32, torch.float16 or torch.bfloat16")
        self.glm_tokenizer: GlmImageTokenizer | None = None
        self.byt5_tokenizer: ByT5Tokenizer | None = None
        self.scheduler_config: dict[str, Any] | None = None
        self.settings: GenerationSettings | None = None
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults

    def _latent_shape(self, options: dict[str, Any]) -> tuple[int, int, int, int]:
        channels = 16
        vae_scale = 8
        transformer_config = self.source / "transformer" / "config.json"
        vae_config = self.source / "vae" / "config.json"
        if transformer_config.is_file():
            channels = int(read_json(transformer_config).get("in_channels", channels))
        if vae_config.is_file():
            blocks = read_json(vae_config).get("block_out_channels", [1, 1, 1, 1])
            vae_scale = 2 ** (len(blocks) - 1)
        return (1, channels, options["height"] // vae_scale, options["width"] // vae_scale)

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown GLM-Image options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        for name in ("steps", "width", "height", "max_sequence_length"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        for name in ("width", "height"):
            if o[name] % _MULTIPLE:
                raise ValueError(f"{name} must be a multiple of {_MULTIPLE}")
        guidance = o["guidance"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, int | float)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError("guidance must be a finite number >= 0")
        seed, latents = o["seed"], o["latents"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        shape = self._latent_shape(o)
        if latents is not None and (
            not isinstance(latents, Tensor)
            or not latents.is_floating_point()
            or tuple(latents.shape) != shape
        ):
            raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        o["latent_shape"] = shape
        return o

    def _require_snapshot(self) -> None:
        missing = [name for name in COMPONENTS if not (self.source / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{self.source} is not a GLM-Image snapshot (missing {', '.join(missing)}). "
                + _hint(self.source, self.model_id)
            )

    def _load(self) -> None:
        self._require_snapshot()
        root = self.source
        try:
            self.glm_tokenizer = GlmImageTokenizer(root / "processor")
            self.byt5_tokenizer = ByT5Tokenizer(root / "tokenizer")
            self.scheduler_config = read_json(root / "scheduler" / "scheduler_config.json")
            self.settings = GenerationSettings.read(root / "vision_language_encoder")
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, self.model_id)
            ) from error
        self.snapshot = read_snapshot(root)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the GLM-Image architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.glm_tokenizer = None
        self.byt5_tokenizer = None
        self.scheduler_config = None
        self.settings = None
        self.snapshot = None

    def _load_component(self, loader: Any, name: str) -> Any:
        try:
            return loader.from_pretrained(self.source / name, device=self.device, dtype=self.dtype)
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(self.source, self.model_id)
            ) from error

    @torch.inference_mode()
    def _prior_tokens(self, prompt: str, height: int, width: int, seed: int) -> Tensor:
        assert self.glm_tokenizer is not None and self.settings is not None
        input_ids, grids = build_prior_prompt(self.glm_tokenizer, prompt, height, width)
        (token_h, token_w), (preview_h, preview_w) = grids
        large, preview = token_h * token_w, preview_h * preview_w
        model = cast(GlmImageARModel, self._load_component(GlmImageARModel, "vision_language_encoder"))
        try:
            sequence = model.generate(
                input_ids.to(self.device),
                grids,
                max_new_tokens=large + preview + 1,
                temperature=self.settings.temperature,
                top_k=self.settings.top_k,
                top_p=self.settings.top_p,
                generator=torch.Generator("cpu").manual_seed(seed),
            )
        finally:
            del model
            gc.collect()
        generated = sequence[0, input_ids.shape[1] :]
        tokens = generated[preview : preview + large]
        if tokens.numel() != large:
            raise RuntimeError("The autoregressive model ended before the image was complete")
        tokens = tokens.view(1, 1, token_h, token_w)
        tokens = torch.nn.functional.interpolate(tokens.float(), scale_factor=2, mode="nearest")
        return tokens.to(dtype=torch.long).view(1, -1)

    @torch.inference_mode()
    def _denoise(
        self,
        *,
        prompt: str,
        prior_token_ids: Tensor,
        steps: int,
        guidance: float,
        width: int,
        height: int,
        latents: Tensor,
        max_sequence_length: int,
    ) -> Tensor:
        assert self.byt5_tokenizer is not None and self.scheduler_config is not None
        use_cfg = guidance > 1
        text_encoder = cast(T5EncoderModel, self._load_component(T5EncoderModel, "text_encoder"))
        try:
            dtype = cast(torch.dtype, text_encoder.shared.weight.dtype)
            prompt_embeds = glyph_embeds(
                self.byt5_tokenizer,
                text_encoder,
                prompt,
                max_sequence_length,
                self.device,
                dtype,
            )
            negative_embeds = (
                glyph_embeds(
                    self.byt5_tokenizer,
                    text_encoder,
                    "",
                    max_sequence_length,
                    self.device,
                    dtype,
                )
                if use_cfg
                else None
            )
        finally:
            del text_encoder
            gc.collect()
        transformer = cast(
            GlmImageTransformer2DModel,
            self._load_component(GlmImageTransformer2DModel, "transformer"),
        )
        try:
            return self._sample(
                transformer,
                prompt_embeds,
                negative_embeds,
                prior_token_ids.to(self.device),
                steps=steps,
                guidance=guidance,
                width=width,
                height=height,
                latents=latents,
            )
        finally:
            del transformer
            gc.collect()

    def _sample(
        self,
        transformer: GlmImageTransformer2DModel,
        prompt_embeds: Tensor,
        negative_embeds: Tensor | None,
        prior_token_ids: Tensor,
        *,
        steps: int,
        guidance: float,
        width: int,
        height: int,
        latents: Tensor,
    ) -> Tensor:
        assert self.scheduler_config is not None
        config = self.scheduler_config
        device = latents.device
        target_size = torch.tensor([[height, width]], dtype=prompt_embeds.dtype, device=device)
        crop_coords = torch.zeros((1, 2), dtype=prompt_embeds.dtype, device=device)
        vae_scale = 8
        vae_config = self.source / "vae" / "config.json"
        if vae_config.is_file():
            vae_scale = 2 ** (len(read_json(vae_config)["block_out_channels"]) - 1)
        image_seq_len = ((height // vae_scale) * (width // vae_scale)) // (transformer.patch_size**2)
        scheduler = FlowMatchEulerDiscreteScheduler(config)
        num_train_timesteps = int(scheduler.config["num_train_timesteps"])
        scheduler.set_timesteps(
            steps,
            sigmas=glm_image_sigmas(steps, num_train_timesteps),
            mu=calculate_shift(image_seq_len, config),
        )
        drop_cond = torch.zeros_like(prior_token_ids, dtype=torch.bool)
        drop_uncond = torch.ones_like(prior_token_ids, dtype=torch.bool)
        transformer_dtype = cast(torch.dtype, transformer.image_projector.proj.weight.dtype)
        for timestep in scheduler.timesteps:
            latent_model_input = latents.to(transformer_dtype)
            current_timestep = timestep.expand(latents.shape[0]) - 1
            noise_pred_cond = transformer(
                latent_model_input,
                prompt_embeds,
                prior_token_ids,
                drop_cond,
                current_timestep,
                target_size,
                crop_coords,
            ).float()
            if negative_embeds is not None:
                noise_pred_uncond = transformer(
                    latent_model_input,
                    negative_embeds,
                    prior_token_ids,
                    drop_uncond,
                    current_timestep,
                    target_size,
                    crop_coords,
                ).float()
                noise_pred = noise_pred_uncond + guidance * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
            latents = scheduler.step(noise_pred, timestep, latents)
        return latents

    @torch.inference_mode()
    def _decode(self, latents: Tensor) -> Tensor:
        vae = cast(AutoencoderKL, self._load_component(AutoencoderKL, "vae"))
        try:
            dtype = cast(torch.dtype, vae.decoder.conv_in.weight.dtype)
            latents = latents.to(dtype)
            channels = vae.config["latent_channels"]
            mean = torch.tensor(vae.latents_mean).view(1, channels, 1, 1)
            std = torch.tensor(vae.latents_std).view(1, channels, 1, 1)
            mean = mean.to(latents.device, latents.dtype)
            std = std.to(latents.device, latents.dtype)
            image = vae.decode(latents * std + mean)
            image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
            return (image * 255).round().to(torch.uint8)
        finally:
            del vae
            gc.collect()

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        seed = o["seed"]
        if seed is None:
            seed = secrets.randbits(63)
        latents = o["latents"]
        if latents is None:
            latents = torch.randn(
                o["latent_shape"],
                generator=torch.Generator("cpu").manual_seed(seed),
                dtype=torch.float32,
            )
        prior_token_ids = self._prior_tokens(inputs.prompt, o["height"], o["width"], seed)
        final = self._denoise(
            prompt=inputs.prompt,
            prior_token_ids=prior_token_ids,
            steps=o["steps"],
            guidance=float(o["guidance"]),
            width=o["width"],
            height=o["height"],
            latents=latents.to(torch.float32).to(self.device),
            max_sequence_length=o["max_sequence_length"],
        )
        pixels = self._decode(final)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": o["steps"],
            "guidance": float(o["guidance"]),
            "width": o["width"],
            "height": o["height"],
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
