"""Sana text-to-image backend over a local diffusers snapshot."""

from __future__ import annotations

import math
import os
import secrets
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.pixart import (
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
    resize_and_crop_tensor,
)
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.lora_formats import LoRAOption, apply_lora, parse_lora_option
from imagehub.models.autoencoder_dc import AutoencoderDC
from imagehub.models.gemma2 import Gemma2Model
from imagehub.models.sana_transformer import SanaTransformer2DModel
from imagehub.schedulers.dpmsolver_multistep import DPMSolverMultistepScheduler
from imagehub.schedulers.scm import SCMScheduler
from imagehub.tokenization_gemma import GemmaTokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
COMPLEX_HUMAN_INSTRUCTION: tuple[str, ...] = (
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual "
    "descriptions suitable for image generation. Evaluate the level of detail in the user "
    "prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, "
    "textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly "
    "without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round "
    "shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red "
    "flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, "
    "featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a "
    "double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any "
    "additional commentary or evaluations:",
    "User Prompt: ",
)
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 20,
    "guidance": 4.5,
    "width": 1024,
    "height": 1024,
    "seed": None,
    "negative_prompt": "",
    "use_resolution_binning": True,
    "max_sequence_length": 300,
    # ``None`` selects the published instruction; an empty list disables it.
    "complex_human_instruction": None,
}
# Aspect-ratio bin tables by the transformer's sample_size (1024 and 512 px images).
ASPECT_RATIO_BINS = {32: ASPECT_RATIO_1024_BIN, 16: ASPECT_RATIO_512_BIN}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def classify_height_width_bin(height: int, width: int, sample_size: int) -> tuple[int, int]:
    try:
        ratios = ASPECT_RATIO_BINS[sample_size]
    except KeyError as error:
        raise ValueError("Invalid sample size") from error
    aspect_ratio = float(height / width)
    closest = min(ratios, key=lambda ratio: abs(float(ratio) - aspect_ratio))
    selected = ratios[closest]
    return int(selected[0]), int(selected[1])


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "sana-1600m-1024"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class SanaComponents:
    tokenizer: GemmaTokenizer
    text_encoder: Gemma2Model
    transformer: SanaTransformer2DModel
    vae: AutoencoderDC
    scheduler_config: dict[str, Any]

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> SanaComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Sana snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            class_name = read_json(root / "model_index.json").get("_class_name")
            if class_name not in ("SanaPipeline", "SanaSprintPipeline"):
                raise CheckpointError(f"Unsupported Sana pipeline class {class_name!r}")
            return cls(
                tokenizer=GemmaTokenizer(root / "tokenizer"),
                text_encoder=Gemma2Model.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                transformer=SanaTransformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vae=AutoencoderDC.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _prompt_embeds(
    components: SanaComponents,
    prompt: str,
    max_sequence_length: int,
    instruction: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """``SanaPipeline._get_gemma_prompt_embeds`` for one prompt (right padding)."""
    prompt = prompt.lower().strip()
    if not instruction:
        max_length = max_sequence_length
    else:
        header = "\n".join(instruction)
        prompt = header + prompt
        max_length = len(components.tokenizer.tokenize(header)) + max_sequence_length - 2
    input_ids, attention_mask = components.tokenizer(
        [prompt], max_length=max_length, padding_side="right"
    )
    attention_mask = attention_mask.to(device)
    embeds = components.text_encoder(input_ids.to(device), attention_mask)
    return embeds.to(dtype=dtype, device=device), attention_mask


@torch.inference_mode()
def encode_prompt(
    components: SanaComponents,
    prompt: str,
    negative_prompt: str,
    max_sequence_length: int,
    instruction: Sequence[str],
    device: torch.device,
    do_classifier_free_guidance: bool,
) -> tuple[Tensor, Tensor]:
    dtype = components.text_encoder.embed_tokens.weight.dtype
    # Section 3.1 of the paper: the first token and the last ``max_sequence_length - 1``.
    select_index = [0, *range(-max_sequence_length + 1, 0)]
    prompt_embeds, prompt_mask = _prompt_embeds(
        components, prompt, max_sequence_length, instruction, device, dtype
    )
    prompt_embeds = prompt_embeds[:, select_index]
    prompt_mask = prompt_mask[:, select_index]
    if not do_classifier_free_guidance:
        return prompt_embeds, prompt_mask
    negative_embeds, negative_mask = _prompt_embeds(
        components, negative_prompt, max_sequence_length, (), device, dtype
    )
    return torch.cat([negative_embeds, prompt_embeds]), torch.cat([negative_mask, prompt_mask])


def _is_sprint_scheduler(config: dict[str, Any]) -> bool:
    return config.get("_class_name") == "SCMScheduler"


@torch.inference_mode()
def _denoise_sprint(
    components: SanaComponents,
    *,
    prompt: str,
    steps: int,
    guidance: float,
    max_sequence_length: int,
    instruction: Sequence[str],
    latents: Tensor,
    generator: torch.Generator,
) -> Tensor:
    transformer = components.transformer
    prompt_embeds, prompt_mask = encode_prompt(
        components, prompt, "", max_sequence_length, instruction, latents.device, False
    )
    scheduler = SCMScheduler(components.scheduler_config)
    intermediate = 1.3 if steps == 2 else None
    scheduler.set_timesteps(steps, max_timesteps=1.57080, intermediate_timesteps=intermediate)
    dtype = transformer.patch_embed.proj.weight.dtype
    sigma_data = float(scheduler.config["sigma_data"])
    guidance_embeds_scale = float(transformer.config.get("guidance_embeds_scale", 0.1))
    # Match diffusers: create in float32, expand, cast to model dtype, then scale.
    guidance_tensor = torch.full([1], guidance, device=latents.device, dtype=torch.float32)
    guidance_tensor = guidance_tensor.expand(latents.shape[0]).to(dtype)
    guidance_tensor = guidance_tensor * guidance_embeds_scale
    latents = latents * sigma_data
    for t in scheduler.timesteps[:-1]:
        timestep = t.expand(latents.shape[0])
        latents_model_input = latents / sigma_data
        scm = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
        scm_expanded = scm.view(-1, 1, 1, 1)
        latent_model_input = latents_model_input * torch.sqrt(
            scm_expanded**2 + (1 - scm_expanded) ** 2
        )
        transformer_out = transformer(
            latent_model_input.to(dtype=dtype),
            prompt_embeds.to(dtype=dtype),
            scm,
            prompt_mask,
            guidance=guidance_tensor,
        ).float()
        noise_pred = (
            (1 - 2 * scm_expanded) * latent_model_input
            + (1 - 2 * scm_expanded + 2 * scm_expanded**2) * transformer_out
        ) / torch.sqrt(scm_expanded**2 + (1 - scm_expanded) ** 2)
        noise_pred = noise_pred * sigma_data
        latents, _ = scheduler.step(noise_pred, t, latents, generator=generator)
    return latents / sigma_data


@torch.inference_mode()
def _denoise_standard(
    components: SanaComponents,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    max_sequence_length: int,
    instruction: Sequence[str],
    latents: Tensor,
) -> Tensor:
    use_cfg = guidance > 1.0
    transformer = components.transformer
    prompt_embeds, prompt_mask = encode_prompt(
        components,
        prompt,
        negative_prompt,
        max_sequence_length,
        instruction,
        latents.device,
        use_cfg,
    )
    scheduler = DPMSolverMultistepScheduler(components.scheduler_config)
    scheduler.set_timesteps(steps)
    dtype = transformer.patch_embed.proj.weight.dtype
    timestep_scale = float(transformer.config.get("timestep_scale", 1.0))
    in_channels = int(transformer.config.get("in_channels", 32))
    out_channels = int(transformer.config.get("out_channels") or in_channels)
    for t in scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2) if use_cfg else latents
        timestep = t.expand(latent_model_input.shape[0]) * timestep_scale
        noise_pred = transformer(
            latent_model_input.to(dtype=dtype),
            prompt_embeds.to(dtype=dtype),
            timestep,
            prompt_mask,
        ).float()
        if use_cfg:
            noise_unconditional, noise_text = noise_pred.chunk(2)
            noise_pred = noise_unconditional + guidance * (noise_text - noise_unconditional)
        if out_channels // 2 == in_channels:
            noise_pred = noise_pred.chunk(2, dim=1)[0]
        latents = scheduler.step(noise_pred, t, latents)
    return latents


@torch.inference_mode()
def denoise(
    components: SanaComponents,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    max_sequence_length: int,
    instruction: Sequence[str],
    latents: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    if _is_sprint_scheduler(components.scheduler_config):
        return _denoise_sprint(
            components,
            prompt=prompt,
            steps=steps,
            guidance=guidance,
            max_sequence_length=max_sequence_length,
            instruction=instruction,
            latents=latents,
            generator=generator or torch.Generator("cpu"),
        )
    return _denoise_standard(
        components,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance=guidance,
        max_sequence_length=max_sequence_length,
        instruction=instruction,
        latents=latents,
    )


@torch.inference_mode()
def decode_latents(
    vae: AutoencoderDC,
    latents: Tensor,
    *,
    output_width: int,
    output_height: int,
    resize: bool,
) -> Tensor:
    latents = latents.to(next(vae.parameters()).dtype)
    image = vae.decode(latents / vae.scaling_factor)
    if resize:
        image = resize_and_crop_tensor(image, output_width, output_height)
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class SanaBackend(TextToImageBackend):
    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        model_id: str | None = None,
        lora: LoRAOption = None,
        lora_scale: float = 1.0,
        lora_fuse: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(lora_fuse, bool):
            raise TypeError("lora_fuse must be a bool")
        self.lora = parse_lora_option(lora, lora_scale)
        self.lora_fuse = lora_fuse
        self.source = Path(model)
        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = dtype or torch.float32
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be torch.float32, torch.float16 or torch.bfloat16")
        self.components: SanaComponents | None = None
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown Sana options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if o["negative_prompt"] is None:
            o["negative_prompt"] = ""
        if not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        if type(o["use_resolution_binning"]) is not bool:
            raise TypeError("use_resolution_binning must be a bool")
        instruction = o["complex_human_instruction"]
        if instruction is None:
            instruction = COMPLEX_HUMAN_INSTRUCTION
        if (
            not isinstance(instruction, list | tuple)
            or isinstance(instruction, str)
            or not all(isinstance(line, str) for line in instruction)
        ):
            raise TypeError("complex_human_instruction must be a list of strings or None")
        o["complex_human_instruction"] = tuple(instruction)
        for name in ("steps", "width", "height", "max_sequence_length"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        if (
            isinstance(o["guidance"], bool)
            or not isinstance(o["guidance"], int | float)
            or not math.isfinite(o["guidance"])
            or o["guidance"] < 0
        ):
            raise ValueError("guidance must be a finite number >= 0")
        seed = o["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        o["output_width"], o["output_height"] = o["width"], o["height"]
        try:
            transformer_config = read_json(self.source / "transformer" / "config.json")
        except FileNotFoundError:
            transformer_config = {"in_channels": 32, "sample_size": 32}
        if o["use_resolution_binning"]:
            o["height"], o["width"] = classify_height_width_bin(
                o["height"], o["width"], int(transformer_config.get("sample_size", 32))
            )
        for name in ("width", "height"):
            if o[name] % 32:
                raise ValueError(f"{name} must be a multiple of 32")
        try:
            vae_config = read_json(self.source / "vae" / "config.json")
        except FileNotFoundError:
            vae_config = {"encoder_block_out_channels": [1] * 6}
        vae_scale = 2 ** (len(vae_config.get("encoder_block_out_channels", [1] * 6)) - 1)
        o["latent_shape"] = (
            1,
            int(transformer_config.get("in_channels", 32)),
            o["height"] // vae_scale,
            o["width"] // vae_scale,
        )
        return o

    def _load(self) -> None:
        self.components = SanaComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        for module in (
            self.components.text_encoder,
            self.components.transformer,
            self.components.vae,
        ):
            module.eval()
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the Sana architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )
        if self.lora:
            # SanaPipeline.load_lora_weights loads transformer adapters only.
            apply_lora({"transformer": self.components.transformer}, self.lora, fuse=self.lora_fuse)

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        seed = o["seed"]
        if seed is None:
            seed = secrets.randbits(63)
        generator = torch.Generator("cpu").manual_seed(seed)
        latents = torch.randn(o["latent_shape"], generator=generator, dtype=torch.float32)
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
            steps=o["steps"],
            guidance=float(o["guidance"]),
            max_sequence_length=o["max_sequence_length"],
            instruction=o["complex_human_instruction"],
            latents=latents.to(self.device),
            generator=generator,
        )
        pixels = decode_latents(
            self.components.vae,
            final,
            output_width=o["output_width"],
            output_height=o["output_height"],
            resize=o["use_resolution_binning"],
        )
        snapshot = self.snapshot or {}
        scheduler_name = (
            "scm" if _is_sprint_scheduler(self.components.scheduler_config) else "dpm-solver++"
        )
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": o["steps"],
            "guidance": float(o["guidance"]),
            "width": o["output_width"],
            "height": o["output_height"],
            "negative_prompt": o["negative_prompt"],
            "scheduler": scheduler_name,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
