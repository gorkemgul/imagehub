"""aMUSEd text-to-image backend over a local diffusers snapshot (masked-token generation)."""

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
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.uvit_2d import UVit2DModel
from imagehub.models.vq_model import VQModel
from imagehub.schedulers.amused import AmusedScheduler
from imagehub.tokenization import CLIPTokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vqvae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "steps": 12,
    "guidance": 10.0,
    "width": None,
    "height": None,
    "seed": None,
    "negative_prompt": "",
    "aesthetic_score": 6,
    "crop_coord": (0, 0),
    "temperature": (2, 0),
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "amused-512"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class AmusedComponents:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModelWithProjection
    transformer: UVit2DModel
    vqvae: VQModel
    scheduler_config: dict[str, Any]

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> AmusedComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not an aMUSEd snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            class_name = read_json(root / "model_index.json").get("_class_name")
            if class_name != "AmusedPipeline":
                raise CheckpointError(f"Unsupported aMUSEd pipeline class {class_name!r}")
            return cls(
                tokenizer=CLIPTokenizer(root / "tokenizer"),
                text_encoder=CLIPTextModelWithProjection.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                transformer=UVit2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                vqvae=VQModel.from_pretrained(root / "vqvae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def vae_scale_factor(vqvae: VQModel) -> int:
    return 2 ** (len(vqvae.config["block_out_channels"]) - 1)


def _embed(components: AmusedComponents, text: str, device: torch.device) -> tuple[Tensor, Tensor]:
    """Pooled projected embedding and the penultimate hidden state for one text."""
    ids = torch.tensor([components.tokenizer(text)], device=device)
    text_embeds, _, hidden_states = components.text_encoder(ids, output_hidden_states=True)
    return text_embeds, hidden_states[-2]


@torch.inference_mode()
def encode_prompt(
    components: AmusedComponents,
    prompt: str,
    negative_prompt: str,
    device: torch.device,
    do_classifier_free_guidance: bool,
) -> tuple[Tensor, Tensor]:
    """Pooled embeddings and hidden states, with the negative prompt first when guiding."""
    pooled, hidden = _embed(components, prompt, device)
    if not do_classifier_free_guidance:
        return pooled, hidden
    negative_pooled, negative_hidden = _embed(components, negative_prompt, device)
    return torch.cat([negative_pooled, pooled]), torch.cat([negative_hidden, hidden])


@torch.inference_mode()
def denoise(
    components: AmusedComponents,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    height: int,
    width: int,
    aesthetic_score: float = 6,
    crop_coord: tuple[int, int] = (0, 0),
    temperature: float | tuple[float, float] | list[float] = (2, 0),
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Run the masked-token loop from fully masked tokens and return ids (1, h/16, w/16)."""
    transformer = components.transformer
    device = device or transformer.embed.embeddings.weight.device
    use_cfg = guidance > 1.0
    pooled, hidden = encode_prompt(components, prompt, negative_prompt, device, use_cfg)
    scale = vae_scale_factor(components.vqvae)
    # The micro-conditioning lists the width before the height, as the reference does.
    micro_conds = torch.tensor(
        [width, height, crop_coord[0], crop_coord[1], aesthetic_score],
        device=device,
        dtype=hidden.dtype,
    )
    micro_conds = micro_conds.unsqueeze(0).expand(2 if use_cfg else 1, -1)
    scheduler = AmusedScheduler(components.scheduler_config)
    latents = torch.full(
        (1, height // scale, width // scale),
        scheduler.mask_token_id,
        dtype=torch.long,
        device=device,
    )
    scheduler.set_timesteps(steps, temperature, device)
    assert scheduler.timesteps is not None
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents] * 2) if use_cfg else latents
        logits = transformer(model_input, hidden, pooled, micro_conds)
        if use_cfg:
            uncond_logits, cond_logits = logits.chunk(2)
            logits = uncond_logits + guidance * (cond_logits - uncond_logits)
        latents = scheduler.step(logits, timestep, latents, generator=generator).prev_sample
    return latents


@torch.inference_mode()
def decode_tokens(vqvae: VQModel, tokens: Tensor) -> Tensor:
    """Decode token ids to uint8 images of shape (batch, height, width, 3) on CPU."""
    batch, height, width = tokens.shape
    upcast = vqvae.post_quant_conv.weight.dtype == torch.float16 and vqvae.config["force_upcast"]
    if upcast:
        vqvae.float()
    try:
        image = vqvae.decode(
            tokens,
            force_not_quantize=True,
            shape=(batch, height, width, vqvae.config["latent_channels"]),
        ).clip(0, 1)
    finally:
        if upcast:
            vqvae.half()
    image = image.permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        bound = "" if minimum is None else f" >= {minimum:g}"
        raise ValueError(f"{name} must be a finite number{bound}")


class AmusedBackend(TextToImageBackend):
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
        self.components: AmusedComponents | None = None
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
            raise TypeError(f"Unknown aMUSEd options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if o["negative_prompt"] is None:
            o["negative_prompt"] = ""
        if not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        _positive_int(o, "steps")
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        _finite_number(o["guidance"], "guidance", minimum=0)
        _finite_number(o["aesthetic_score"], "aesthetic_score")
        crop = o["crop_coord"]
        if (
            not isinstance(crop, tuple | list)
            or len(crop) != 2
            or any(type(value) is not int for value in crop)
        ):
            raise ValueError("crop_coord must be a pair of ints")
        o["crop_coord"] = (int(crop[0]), int(crop[1]))
        temperature = o["temperature"]
        if isinstance(temperature, tuple | list):
            if len(temperature) != 2:
                raise ValueError("temperature must be a number or a (start, end) pair")
            for value in temperature:
                _finite_number(value, "temperature", minimum=0)
            o["temperature"] = (temperature[0], temperature[1])
        else:
            _finite_number(temperature, "temperature", minimum=0)
        seed = o["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        try:
            transformer_config = read_json(self.source / "transformer" / "config.json")
        except FileNotFoundError:
            transformer_config = {}
        try:
            blocks = read_json(self.source / "vqvae" / "config.json").get(
                "block_out_channels", [1, 1, 1, 1, 1]
            )
        except FileNotFoundError:
            blocks = [1, 1, 1, 1, 1]
        scale = 2 ** (len(blocks) - 1)
        # The 512 model halves the token grid inside the network, so it needs even grids.
        multiple = scale * (2 if transformer_config.get("downsample") else 1)
        size = int(transformer_config.get("sample_size", 64)) * scale
        for name in ("width", "height"):
            if o[name] is None:
                o[name] = size
            _positive_int(o, name)
            if o[name] % multiple:
                raise ValueError(f"{name} must be a multiple of {multiple}")
        return o

    def _load(self) -> None:
        self.components = AmusedComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the aMUSEd architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        seed = o["seed"]
        if seed is None:
            seed = secrets.randbits(63)
        generator = torch.Generator("cpu").manual_seed(seed)
        tokens = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
            steps=o["steps"],
            guidance=float(o["guidance"]),
            height=o["height"],
            width=o["width"],
            aesthetic_score=o["aesthetic_score"],
            crop_coord=o["crop_coord"],
            temperature=o["temperature"],
            generator=generator,
            device=self.device,
        )
        pixels = decode_tokens(self.components.vqvae, tokens)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": o["steps"],
            "guidance": float(o["guidance"]),
            "width": o["width"],
            "height": o["height"],
            "negative_prompt": o["negative_prompt"],
            "aesthetic_score": o["aesthetic_score"],
            "crop_coord": list(o["crop_coord"]),
            "temperature": list(o["temperature"])
            if isinstance(o["temperature"], tuple)
            else o["temperature"],
            "scheduler": "amused",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
