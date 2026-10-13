"""DreamLite text-to-image backend over a local diffusers-layout snapshot."""

from __future__ import annotations

import math
import os
import secrets
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import load_weights, read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_tiny import AutoencoderTiny
from imagehub.models.dreamlite import DreamLiteUNetModel
from imagehub.models.qwen3_vl_text import Qwen3VLTextModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler, qwen_image_mu
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "unet", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 30,
    "guidance": 3.5,
    "image_guidance": 1.5,
    "width": 1024,
    "height": 1024,
    "num_images": 1,
    "seed": None,
    "latents": None,
    "max_sequence_length": 200,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_VAE_SCALE = 8
_PREFIX_TOKENS = 34
_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "dreamlite-base"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


def _load_text_encoder(
    directory: Path, *, device: torch.device, dtype: torch.dtype
) -> Qwen3VLTextModel:
    config = read_json(directory / "config.json")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise CheckpointError(f"Expected text_config in {directory / 'config.json'}")
    state = load_weights(directory)
    prefix = "model.language_model."
    language_state = {
        name.removeprefix(prefix): value for name, value in state.items() if name.startswith(prefix)
    }
    unexpected = sorted(
        name
        for name in state
        if not name.startswith((prefix, "model.visual.")) and name != "lm_head.weight"
    )
    if unexpected or not language_state:
        detail = unexpected[:20] if unexpected else ["missing language-model tensors"]
        raise CheckpointError("Unexpected DreamLite text encoder: " + ", ".join(detail))
    with torch.device("meta"):
        model = Qwen3VLTextModel(text_config)
    model.load_checkpoint(language_state)
    model.init_non_persistent_buffers()
    model.to(device=device, dtype=dtype).eval()
    return model


@dataclass
class DreamLiteComponents:
    tokenizer: Qwen2Tokenizer
    text_encoder: Qwen3VLTextModel
    unet: DreamLiteUNetModel
    vae: AutoencoderTiny
    scheduler_config: dict[str, Any]

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> DreamLiteComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a DreamLite snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            return cls(
                tokenizer=Qwen2Tokenizer(root / "tokenizer"),
                text_encoder=_load_text_encoder(root / "text_encoder", device=device, dtype=dtype),
                unet=DreamLiteUNetModel.from_pretrained(root / "unet", device=device, dtype=dtype),
                vae=AutoencoderTiny.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


@torch.inference_mode()
def encode_prompt(
    components: DreamLiteComponents,
    prompts: list[str],
    device: torch.device,
    max_sequence_length: int,
) -> tuple[Tensor, Tensor]:
    token_rows = [
        components.tokenizer.tokenize(_TEMPLATE.format(prompt))[
            : max_sequence_length + _PREFIX_TOKENS
        ]
        for prompt in prompts
    ]
    length = max(len(row) for row in token_rows)
    input_ids = torch.tensor(
        [row + [components.tokenizer.pad_id] * (length - len(row)) for row in token_rows],
        dtype=torch.long,
        device=device,
    )
    mask = torch.tensor(
        [[True] * len(row) + [False] * (length - len(row)) for row in token_rows],
        dtype=torch.bool,
        device=device,
    )
    hidden = components.text_encoder(input_ids, mask)[-1]
    selected = [hidden[index, mask[index]][_PREFIX_TOKENS:] for index in range(len(prompts))]
    output_length = max(row.shape[0] for row in selected)
    output = hidden.new_zeros(len(prompts), output_length, hidden.shape[-1])
    output_mask = torch.zeros(len(prompts), output_length, dtype=torch.bool, device=device)
    for index, row in enumerate(selected):
        output[index, : row.shape[0]] = row
        output_mask[index, : row.shape[0]] = True
    return output, output_mask


@torch.inference_mode()
def denoise(
    components: DreamLiteComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float | None,
    width: int,
    height: int,
    latents: Tensor,
    mobile: bool,
    max_sequence_length: int,
) -> Tensor:
    dtype = components.unet.conv_in.weight.dtype
    batch = latents.shape[0]
    use_cfg = not mobile
    prompts = (
        [negative_prompt or "", f"[Generate]: {prompt}"] if use_cfg else [f"[Generate]: {prompt}"]
    )
    embeds, embed_mask = encode_prompt(components, prompts, latents.device, max_sequence_length)
    embeds = embeds.to(dtype)
    embeds = embeds.repeat_interleave(batch, dim=0)
    embed_mask = embed_mask.repeat_interleave(batch, dim=0)
    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    sigmas = torch.linspace(1.0, 1 / steps, steps, dtype=torch.float64).tolist()
    scheduler.set_timesteps(steps, sigmas=sigmas, mu=qwen_image_mu(height, width, scheduler.config))
    scheduler.set_begin_index(0)
    time_ids = torch.tensor([[width, height]], device=latents.device, dtype=dtype)
    for timestep in scheduler.timesteps:
        count = 2 if use_cfg else 1
        latent_input = latents.to(dtype).repeat(count, 1, 1, 1)
        model_input = torch.cat([latent_input, torch.zeros_like(latent_input)], dim=3)
        prediction = components.unet(
            model_input,
            timestep.expand(model_input.shape[0]).to(dtype),
            embeds,
            time_ids=time_ids.repeat(model_input.shape[0], 1),
            context_mask=embed_mask,
        )[..., : latents.shape[-1]]
        if use_cfg:
            assert guidance is not None
            uncond, cond = prediction.chunk(2)
            prediction = uncond + float(guidance) * (cond - uncond)
        latents = scheduler.step(prediction, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderTiny, latents: Tensor) -> Tensor:
    latents = latents / vae.scaling_factor + vae.shift_factor
    image = vae.decode(latents.to(next(vae.parameters()).dtype))
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


class DreamLiteBackend(TextToImageBackend):
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
        self.device = resolve_device(device)
        self.dtype = dtype or torch.bfloat16
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be torch.float32, torch.float16 or torch.bfloat16")
        self.model_id = model_id
        self.mobile = model_id == "dreamlite-mobile"
        self.components: DreamLiteComponents | None = None
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
            raise TypeError(f"Unknown DreamLite options: {', '.join(unknown)}")
        selected = self.option_defaults | merged
        if selected["negative_prompt"] is not None and not isinstance(
            selected["negative_prompt"], str
        ):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "width", "height", "num_images", "max_sequence_length"):
            if type(selected[name]) is not int or selected[name] <= 0:
                raise ValueError(f"{name} must be a positive int")
        if selected["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        for name in ("width", "height"):
            if selected[name] % 16:
                raise ValueError(f"{name} must be a multiple of 16")
        for name in ("guidance", "image_guidance"):
            value = selected[name]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number or None")
        seed, latents = selected["seed"], selected["latents"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        shape = (
            selected["num_images"],
            4,
            selected["height"] // _VAE_SCALE,
            selected["width"] // _VAE_SCALE,
        )
        if latents is not None:
            if seed is not None:
                raise ValueError("Pass either seed or latents, not both")
            if (
                not isinstance(latents, Tensor)
                or not latents.is_floating_point()
                or tuple(latents.shape) != shape
            ):
                raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        return selected

    def _load(self) -> None:
        self.components = DreamLiteComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the DreamLite architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed, latents = options["seed"], options["latents"]
        if latents is None:
            if seed is None:
                seed = secrets.randbits(63)
            latents = torch.randn(
                (
                    options["num_images"],
                    4,
                    options["height"] // _VAE_SCALE,
                    options["width"] // _VAE_SCALE,
                ),
                generator=torch.Generator("cpu").manual_seed(seed),
                dtype=self.dtype,
            )
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=options["negative_prompt"],
            steps=options["steps"],
            guidance=options["guidance"],
            width=options["width"],
            height=options["height"],
            latents=latents.to(self.dtype).to(self.device),
            mobile=self.mobile,
            max_sequence_length=options["max_sequence_length"],
        )
        pixels = decode_latents(self.components.vae, final)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": options["steps"],
            "guidance": options["guidance"],
            "width": options["width"],
            "height": options["height"],
            "negative_prompt": options["negative_prompt"],
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
