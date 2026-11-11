"""HunyuanImage 2.1 text-to-image backend over a local diffusers-layout snapshot.

The checkpoint is large, so the stages run one after the other and each model is freed
before the next one is loaded: Qwen2.5-VL, ByT5 (only when the prompt quotes text), the
transformer and the VAE.
"""

from __future__ import annotations

import gc
import math
import os
import re
import secrets
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
from imagehub.models.autoencoder_kl_hunyuanimage import AutoencoderKLHunyuanImage
from imagehub.models.hunyuanimage_transformer import HunyuanImageTransformer2DModel
from imagehub.models.qwen2_5_vl_text import Qwen25VLTextModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_byt5 import ByT5Tokenizer
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = (
    "tokenizer",
    "tokenizer_2",
    "text_encoder",
    "text_encoder_2",
    "transformer",
    "vae",
    "scheduler",
)
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 50,
    "guidance": None,
    "width": 2048,
    "height": 2048,
    "seed": None,
    "latents": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_VAE_SCALE_FACTOR = 32
_LATENT_CHANNELS = 64
_TOKENIZER_MAX_LENGTH = 1000
_TOKENIZER_2_MAX_LENGTH = 128
_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>"
)
_TEMPLATE_START = 34
_SKIP_LAYER = 2
_QUOTES = (r"\'(.*?)\'", r"\"(.*?)\"", r"‘(.*?)’", r"“(.*?)”")


def extract_glyph_text(prompt: str) -> str | None:
    """Quoted passages of the prompt, formatted for the glyph (ByT5) encoder."""
    texts: list[str] = []
    for pattern in _QUOTES:
        texts.extend(re.findall(pattern, prompt))
    if not texts:
        return None
    return ". ".join(f'Text "{text}"' for text in texts) + ". "


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "hunyuanimage-2-1"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


def sigmas_for(steps: int) -> list[float]:
    """``numpy.linspace(1.0, 0.0, steps + 1)[:-1]`` in float64."""
    step = (0.0 - 1.0) / steps
    return [index * step + 1.0 for index in range(steps)]


class MomentumBuffer:
    def __init__(self, momentum: float) -> None:
        self.momentum = momentum
        self.running_average: Tensor | int = 0

    def update(self, value: Tensor) -> None:
        self.running_average = value + self.momentum * self.running_average


class AdaptiveProjectedMixGuidance:
    """Classifier-free guidance that switches to adaptive projected guidance after a step."""

    def __init__(self, config: dict[str, Any], guidance_scale: float | None = None) -> None:
        self.guidance_scale = float(
            config["guidance_scale"] if guidance_scale is None else guidance_scale
        )
        self.apg_scale = float(config["adaptive_projected_guidance_scale"])
        self.momentum = config["adaptive_projected_guidance_momentum"]
        self.rescale = float(config["adaptive_projected_guidance_rescale"])
        self.eta = float(config["eta"])
        self.apg_start_step = int(config["adaptive_projected_guidance_start_step"])
        self.original = bool(config["use_original_formulation"])
        for name in ("start", "stop", "guidance_rescale"):
            if config[name] != {"start": 0.0, "stop": 1.0, "guidance_rescale": 0.0}[name]:
                raise ValueError(f"Unsupported HunyuanImage guider setting {name}={config[name]}")
        if not config["enabled"]:
            raise ValueError("Unsupported HunyuanImage guider setting enabled=False")
        self.buffer: MomentumBuffer | None = None

    @property
    def cfg_enabled(self) -> bool:
        target = 0.0 if self.original else 1.0
        return not math.isclose(self.guidance_scale, target)

    def apg_enabled(self, step: int) -> bool:
        target = 0.0 if self.original else 1.0
        return (
            self.cfg_enabled
            and step > self.apg_start_step
            and not math.isclose(self.apg_scale, target)
        )

    def combine(self, step: int, cond: Tensor, uncond: Tensor | None) -> Tensor:
        if step == 0 and self.momentum is not None:
            self.buffer = MomentumBuffer(self.momentum)
        if uncond is None:
            return cond
        if not self.apg_enabled(step):
            if self.buffer is not None:
                self.buffer.update(cond - uncond)
            shift = cond - uncond
            base = cond if self.original else uncond
            return base + self.guidance_scale * shift
        if self.buffer is not None:
            self.buffer.update(cond - uncond)
            diff = self.buffer.running_average
        else:
            diff = cond - uncond
        assert isinstance(diff, Tensor)
        dim = [-i for i in range(1, len(diff.shape))]
        if self.rescale > 0:
            ones = torch.ones_like(diff)
            norm = diff.norm(p=2, dim=dim, keepdim=True)
            diff = diff * torch.minimum(ones, self.rescale / norm)
        v0, v1 = diff.double(), cond.double()
        v1 = torch.nn.functional.normalize(v1, dim=dim)  # type: ignore[arg-type]
        parallel = (v0 * v1).sum(dim=dim, keepdim=True) * v1
        orthogonal = v0 - parallel
        update = orthogonal.type_as(diff) + self.eta * parallel.type_as(diff)
        base = cond if self.original else uncond
        return base + self.apg_scale * update


def _release() -> None:
    gc.collect()


@torch.inference_mode()
def encode_qwen(
    root: Path, prompts: list[str], *, device: torch.device, dtype: torch.dtype
) -> list[tuple[Tensor, Tensor]]:
    """Qwen2.5-VL embeddings (third-from-last hidden state) and masks for each prompt."""
    tokenizer = Qwen2Tokenizer(root / "tokenizer")
    model = Qwen25VLTextModel.from_pretrained(root / "text_encoder", device=device, dtype=dtype)
    results = []
    for prompt in prompts:
        ids, mask = tokenizer(_TEMPLATE.format(prompt), _TOKENIZER_MAX_LENGTH + _TEMPLATE_START)
        hidden_states = model(ids.to(device), mask.to(device))
        embeds = hidden_states[-(_SKIP_LAYER + 1)][:, _TEMPLATE_START:]
        results.append((embeds.to(dtype), mask[:, _TEMPLATE_START:].to(device)))
    del model
    _release()
    return results


@torch.inference_mode()
def encode_glyph(
    root: Path, text: str, *, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    tokenizer = ByT5Tokenizer(root / "tokenizer_2")
    model = T5EncoderModel.from_pretrained(root / "text_encoder_2", device=device, dtype=dtype)
    ids, mask = tokenizer(text, _TOKENIZER_2_MAX_LENGTH)
    embeds = model(ids.to(device), mask.to(device).to(torch.float32))[0]
    del model
    _release()
    return embeds.to(dtype), mask.to(device)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class HunyuanImageBackend(TextToImageBackend):
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
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults

    def _layout(self) -> tuple[int, int]:
        """Latent channels and spatial compression, from the snapshot when it is present."""
        channels, scale = _LATENT_CHANNELS, _VAE_SCALE_FACTOR
        transformer_config = self.source / "transformer" / "config.json"
        vae_config = self.source / "vae" / "config.json"
        if transformer_config.is_file():
            channels = int(read_json(transformer_config).get("in_channels", channels))
        if vae_config.is_file():
            scale = int(read_json(vae_config).get("spatial_compression_ratio", scale))
        return channels, scale

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown HunyuanImage options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        for name in ("steps", "width", "height"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        channels, scale = self._layout()
        for name in ("width", "height"):
            if o[name] % scale:
                raise ValueError(f"{name} must be a multiple of {scale}")
        guidance = o["guidance"]
        if guidance is not None and (
            isinstance(guidance, bool)
            or not isinstance(guidance, int | float)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError("guidance must be None or a finite number >= 0")
        negative_prompt = o["negative_prompt"]
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise ValueError("negative_prompt must be a string or None")
        seed, latents = o["seed"], o["latents"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        shape = (1, channels, o["height"] // scale, o["width"] // scale)
        if latents is not None:
            if seed is not None:
                raise ValueError("Pass either seed or latents, not both")
            if (
                not isinstance(latents, Tensor)
                or not latents.is_floating_point()
                or tuple(latents.shape) != shape
            ):
                raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        return o

    def _load(self) -> None:
        missing = [name for name in COMPONENTS if not (self.source / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{self.source} is not a HunyuanImage snapshot (missing {', '.join(missing)}). "
                + _hint(self.source, self.model_id)
            )
        self.snapshot = read_snapshot(self.source)

    def _unload(self) -> None:
        self.snapshot = None

    def _guider_configs(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        guider = read_json(self.source / "guider" / "guider_config.json")
        ocr_path = self.source / "ocr_guider" / "guider_config.json"
        return guider, (read_json(ocr_path) if ocr_path.is_file() else None)

    @torch.inference_mode()
    def _denoise(
        self,
        *,
        pos: tuple[Tensor, Tensor],
        neg: tuple[Tensor, Tensor] | None,
        glyph_pos: tuple[Tensor, Tensor],
        glyph_neg: tuple[Tensor, Tensor],
        guider: AdaptiveProjectedMixGuidance,
        steps: int,
        latents: Tensor,
    ) -> Tensor:
        dtype = self.dtype
        transformer = HunyuanImageTransformer2DModel.from_pretrained(
            self.source / "transformer", device=self.device, dtype=dtype
        )
        scheduler = FlowMatchEulerDiscreteScheduler(
            read_json(self.source / "scheduler" / "scheduler_config.json")
        )
        scheduler.set_timesteps(steps, sigmas=sigmas_for(steps))
        scheduler.set_begin_index(0)
        latents = latents.to(dtype)
        for step, timestep in enumerate(scheduler.timesteps):
            model_timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
            cond = transformer(latents, model_timestep, pos[0], pos[1], glyph_pos[0], glyph_pos[1])
            uncond = None
            if guider.cfg_enabled:
                assert neg is not None
                uncond = transformer(
                    latents, model_timestep, neg[0], neg[1], glyph_neg[0], glyph_neg[1]
                )
            noise_pred = guider.combine(step, cond, uncond)
            latents = scheduler.step(noise_pred, timestep, latents)
        del transformer
        _release()
        return latents

    @torch.inference_mode()
    def _decode(self, latents: Tensor) -> Tensor:
        vae = AutoencoderKLHunyuanImage.from_pretrained(
            self.source / "vae", device=self.device, dtype=self.dtype
        )
        image = vae.decode(latents.to(self.dtype) / vae.scaling_factor)
        del vae
        _release()
        image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
        return (image * 255).round().to(torch.uint8)

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        seed, latents = o["seed"], o["latents"]
        if latents is None:
            if seed is None:
                seed = secrets.randbits(63)
            channels, scale = self._layout()
            latents = torch.randn(
                (1, channels, o["height"] // scale, o["width"] // scale),
                generator=torch.Generator("cpu").manual_seed(seed),
                dtype=torch.float32,
            )
        guider_config, ocr_config = self._guider_configs()
        glyph_text = extract_glyph_text(inputs.prompt)
        use_ocr = glyph_text is not None and ocr_config is not None
        guider = AdaptiveProjectedMixGuidance(
            ocr_config if use_ocr and ocr_config is not None else guider_config,
            o["guidance"],
        )
        negative = o["negative_prompt"] or ""
        prompts = [inputs.prompt] + ([negative] if guider.cfg_enabled else [])
        embeds = encode_qwen(self.source, prompts, device=self.device, dtype=self.dtype)
        d_model = int(read_json(self.source / "text_encoder_2" / "config.json")["d_model"])
        empty = (
            torch.zeros((1, _TOKENIZER_2_MAX_LENGTH, d_model), device=self.device).to(self.dtype),
            torch.zeros((1, _TOKENIZER_2_MAX_LENGTH), device=self.device, dtype=torch.int64),
        )
        glyph_pos = empty
        if glyph_text is not None:
            glyph_pos = encode_glyph(self.source, glyph_text, device=self.device, dtype=self.dtype)
            # An all-zero embedding would select the plain guider in the reference.
            if bool(torch.all(glyph_pos[0] == 0)):
                use_ocr = False
                guider = AdaptiveProjectedMixGuidance(guider_config, o["guidance"])
        final = self._denoise(
            pos=embeds[0],
            neg=embeds[1] if len(embeds) > 1 else None,
            glyph_pos=glyph_pos,
            glyph_neg=empty,
            guider=guider,
            steps=o["steps"],
            latents=latents.to(torch.float32).to(self.device),
        )
        pixels = self._decode(final)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": o["steps"],
            "guidance": guider.guidance_scale,
            "guider": "ocr" if use_ocr else "default",
            "width": o["width"],
            "height": o["height"],
            "scheduler": "flow_match_euler",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
