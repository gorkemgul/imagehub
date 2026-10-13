"""PRX text-to-image backend over a local diffusers-layout snapshot."""

from __future__ import annotations

import html
import math
import os
import re
import secrets
import urllib.parse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.prx_transformer import PRXTransformer2DModel
from imagehub.models.t5gemma import T5GemmaEncoder
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_gemma import GemmaTokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": "",
    "steps": 28,
    "guidance": 4.0,
    "width": 512,
    "height": 512,
    "seed": None,
    "max_sequence_length": 256,
    "skip_text_cleaning": False,
    "use_resolution_binning": True,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_BAD_PUNCTUATION = re.compile(r"[#®•©™&@·º½¾¿¡§~)\]([}{|\\/*]+")
_RESOLUTION_BINS = {
    512: {
        "0.5": (352, 704),
        "0.57": (384, 672),
        "0.6": (384, 640),
        "0.68": (416, 608),
        "0.78": (448, 576),
        "0.88": (480, 544),
        "1.0": (512, 512),
        "1.13": (544, 480),
        "1.29": (576, 448),
        "1.46": (608, 416),
        "1.67": (640, 384),
        "1.75": (672, 384),
        "2.0": (704, 352),
    },
    1024: {
        "0.49": (704, 1440),
        "0.52": (736, 1408),
        "0.53": (736, 1376),
        "0.57": (768, 1344),
        "0.59": (768, 1312),
        "0.62": (800, 1280),
        "0.67": (832, 1248),
        "0.68": (832, 1216),
        "0.78": (896, 1152),
        "0.83": (928, 1120),
        "0.94": (992, 1056),
        "1.0": (1024, 1024),
        "1.06": (1056, 992),
        "1.13": (1088, 960),
        "1.21": (1120, 928),
        "1.29": (1152, 896),
        "1.37": (1184, 864),
        "1.46": (1216, 832),
        "1.5": (1248, 832),
        "1.71": (1312, 768),
        "1.75": (1344, 768),
        "1.87": (1376, 736),
        "1.91": (1408, 736),
        "2.05": (1440, 704),
    },
}


def clean_text(text: str) -> str:
    """The PRX prompt cleanup for ordinary Unicode text."""
    text = urllib.parse.unquote_plus(str(text)).strip().lower()
    text = re.sub("<person>", "person", text)
    text = re.sub(
        r"\b((?:https?|www):(?:/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b/?(?!@))",
        "",
        text,
    )
    text = re.sub(r"@[\w\d]+\b", "", text)
    text = re.sub(r"[\u31c0-\u31ef\u31f0-\u4dff\u4e00-\u9fff]+", "", text)
    text = re.sub(
        r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B"
        r"\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
        "-",
        text,
    )
    text = re.sub(r"[`´«»¨]", '"', text)
    text = re.sub(r"&quot;?", "", text)
    text = re.sub(r"&amp", "", text)
    text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", text)
    text = re.sub(r"\d:\d\d\s+$", "", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"#\d{1,3}\b|#\d{5,}\b|\b\d{6,}\b", "", text)
    text = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", text)
    text = re.sub(r"[\"']{2,}", '"', text)
    text = re.sub(r"[.]{2,}", " ", text)
    text = _BAD_PUNCTUATION.sub(" ", text)
    text = re.sub(r"\s+\.\s+", " ", text)
    if len(re.findall(r"[-_]", text)) > 3:
        text = re.sub(r"[-_]", " ", text)
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", text)
    text = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", text)
    text = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", text)
    text = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", text)
    text = re.sub(r"(free\s)?download(\sfree)?", "", text)
    text = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", text)
    text = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", text)
    text = re.sub(r"\bpage\s+\d+\b", "", text)
    text = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", " ", text)
    text = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", text)
    text = re.sub(r"\b\s+:\s+", ": ", text)
    text = re.sub(r"(\D[,./])\b", r"\1 ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"']([\w\W]+)[\"']$", r"\1", text)
    text = re.sub(r"^['_,\-:;]", "", text)
    text = re.sub(r"['_,:\-+]$", "", text)
    text = re.sub(r"^\.\S+$", "", text)
    return text.strip()


def basic_clean(text: str) -> str:
    return html.unescape(html.unescape(str(text))).strip()


@dataclass
class PRXComponents:
    tokenizer: GemmaTokenizer
    text_encoder: T5GemmaEncoder
    transformer: PRXTransformer2DModel
    vae: AutoencoderKL
    scheduler_config: dict[str, Any]

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> PRXComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(f"{root} is not a PRX snapshot (missing {', '.join(missing)})")
        try:
            if read_json(root / "model_index.json").get("_class_name") != "PRXPipeline":
                raise CheckpointError("Unsupported PRX pipeline class")
            return cls(
                GemmaTokenizer(root / "tokenizer"),
                T5GemmaEncoder.from_pretrained(root / "text_encoder", device=device, dtype=dtype),
                PRXTransformer2DModel.from_pretrained(
                    root / "transformer", device=device, dtype=dtype
                ),
                AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(f"{error.filename or error} is missing") from error


def encode_prompt(
    components: PRXComponents,
    prompt: str,
    negative_prompt: str,
    max_length: int,
    device: torch.device,
    guidance: bool,
    skip_cleaning: bool,
) -> tuple[Tensor, Tensor]:
    texts = ([negative_prompt] if guidance else []) + [prompt]
    cleaner = basic_clean if skip_cleaning else clean_text
    texts = [cleaner(value) for value in texts]
    ids, mask = components.tokenizer(texts, max_length=max_length, padding_side="right")
    mask = mask.to(device)
    return components.text_encoder(ids.to(device), mask), mask


@torch.inference_mode()
def denoise(
    components: PRXComponents,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    max_sequence_length: int,
    skip_text_cleaning: bool,
    latents: Tensor,
) -> Tensor:
    use_cfg = guidance > 1.0
    embeddings, mask = encode_prompt(
        components,
        prompt,
        negative_prompt,
        max_sequence_length,
        latents.device,
        use_cfg,
        skip_text_cleaning,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    scheduler.set_timesteps(steps)
    latents = latents.to(embeddings.dtype)
    for timestep in scheduler.timesteps:
        model_input = torch.cat((latents, latents)) if use_cfg else latents
        normalized = (timestep.float() / 1000).reshape(1).expand(model_input.shape[0])
        noise = components.transformer(
            model_input,
            normalized.to(device=latents.device, dtype=latents.dtype),
            embeddings,
            mask,
        )
        if use_cfg:
            unconditional, conditional = noise.chunk(2)
            noise = unconditional + guidance * (conditional - unconditional)
        latents = scheduler.step(noise, timestep, latents)
    return latents


@torch.inference_mode()
def decode_latents(
    vae: AutoencoderKL, latents: Tensor, output_size: tuple[int, int] | None = None
) -> Tensor:
    dtype = vae.decoder.conv_in.weight.dtype
    if dtype == torch.float16 and vae.config.get("force_upcast"):
        vae.to(dtype=torch.float32)
        dtype = torch.float32
    values = latents.to(dtype) / vae.scaling_factor + float(vae.config.get("shift_factor", 0.0))
    image = vae.decode(values)
    if output_size is not None and image.shape[-2:] != output_size:
        height, width = output_size
        ratio = max(height / image.shape[-2], width / image.shape[-1])
        resized = (int(image.shape[-2] * ratio), int(image.shape[-1] * ratio))
        image = F.interpolate(image, size=resized, mode="bilinear", align_corners=False)
        top = (resized[0] - height) // 2
        left = (resized[1] - width) // 2
        image = image[:, :, top : top + height, left : left + width]
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


class PRXBackend(TextToImageBackend):
    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        model_id: str | None = None,
    ) -> None:
        super().__init__()
        self.source, self.model_id = Path(model), model_id
        self.device, self.dtype = resolve_device(device), dtype or torch.float32
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be torch.float32, torch.float16 or torch.bfloat16")
        self.components: PRXComponents | None = None
        self.snapshot: dict[str, Any] | None = None
        defaults = dict(MODEL_CATALOG[model_id].config.get("defaults", {})) if model_id else {}
        self.option_defaults = OPTION_DEFAULTS | defaults

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown PRX options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        if not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string")
        for name in ("steps", "width", "height", "max_sequence_length"):
            if type(o[name]) is not int or o[name] <= 0:
                raise ValueError(f"{name} must be a positive int")
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        if type(o["use_resolution_binning"]) is not bool:
            raise TypeError("use_resolution_binning must be a bool")
        generation_height, generation_width = o["height"], o["width"]
        if o["use_resolution_binning"]:
            try:
                model_index = read_json(self.source / "model_index.json")
                default_size = int(
                    model_index.get("default_sample_size", self.option_defaults["width"])
                )
            except FileNotFoundError:
                default_size = int(self.option_defaults["width"])
            bins = _RESOLUTION_BINS.get(default_size)
            if bins is None:
                raise ValueError(f"resolution binning is not supported for size {default_size}")
            ratio = generation_height / generation_width
            generation_height, generation_width = bins[
                min(bins, key=lambda value: abs(float(value) - ratio))
            ]
        elif generation_width % 16 or generation_height % 16:
            raise ValueError("width and height must be multiples of 16")
        if (
            isinstance(o["guidance"], bool)
            or not isinstance(o["guidance"], int | float)
            or not math.isfinite(o["guidance"])
            or o["guidance"] < 1
        ):
            raise ValueError("guidance must be a finite number >= 1")
        if type(o["skip_text_cleaning"]) is not bool:
            raise TypeError("skip_text_cleaning must be a bool")
        if o["seed"] is not None and (type(o["seed"]) is not int or not 0 <= o["seed"] < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        try:
            channels = int(read_json(self.source / "transformer/config.json")["in_channels"])
        except (FileNotFoundError, KeyError):
            channels = 16
        o["latent_shape"] = (1, channels, generation_height // 8, generation_width // 8)
        return o

    def _load(self) -> None:
        self.components = PRXComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded and self.model_id and recorded != self.model_id:
            warnings.warn(
                f"{self.source} contains {recorded!r}, not {self.model_id!r}", stacklevel=2
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        seed = o["seed"] if o["seed"] is not None else secrets.randbits(63)
        latents = torch.randn(
            o["latent_shape"],
            generator=torch.Generator("cpu").manual_seed(seed),
            dtype=torch.float32,
        ).to(self.device)
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
            steps=o["steps"],
            guidance=float(o["guidance"]),
            max_sequence_length=o["max_sequence_length"],
            skip_text_cleaning=o["skip_text_cleaning"],
            latents=latents,
        )
        pixels = decode_latents(self.components.vae, final, (o["height"], o["width"]))
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
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
