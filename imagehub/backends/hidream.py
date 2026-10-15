"""HiDream-I1 text-to-image backend over a local snapshot (diffusers layout, plus Llama).

The pipeline draws the initial latents from the passed generator but calls the scheduler without
one, so the per-step noise of the flow-match LCM scheduler comes from the global random state. This
backend never touches global state: it keeps two generators seeded with the same ``seed``, one for
the latents and one for the step noise. They produce the numbers the reference produces when the
caller seeds the global state with ``torch.manual_seed(seed)`` right before the pipeline call.
"""

from __future__ import annotations

import gc
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
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.hidream_transformer import HiDreamImageTransformer2DModel
from imagehub.models.llama_text import LlamaTextModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.flow_match_lcm import FlowMatchLCMScheduler
from imagehub.schedulers.unipc import UniPCMultistepScheduler
from imagehub.tokenization import CLIPTokenizer
from imagehub.tokenization_llama3 import Llama3Tokenizer
from imagehub.tokenization_t5 import T5Tokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = (
    "tokenizer",
    "tokenizer_2",
    "tokenizer_3",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "text_encoder_4",
    "transformer",
    "vae",
    "scheduler",
)
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 50,
    "guidance": 5.0,
    "width": 1024,
    "height": 1024,
    "num_images": 1,
    "seed": None,
    "latents": None,
    "max_sequence_length": 128,
}
DEFAULT_SAMPLE_SIZE = 128
CLIP_MAX_LENGTH = 218
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "hidream-i1-fast"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


def _release() -> None:
    gc.collect()


@dataclass
class HiDreamComponents:
    """Tokenizers, small configs and the VAE; the large networks are loaded one stage at a time."""

    root: Path
    device: torch.device
    dtype: torch.dtype
    tokenizer: CLIPTokenizer
    tokenizer_2: CLIPTokenizer
    tokenizer_3: T5Tokenizer
    tokenizer_4: Llama3Tokenizer
    vae: AutoencoderKL
    scheduler_config: dict[str, Any]
    transformer_config: dict[str, Any]

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> HiDreamComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a HiDream-I1 snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            return cls(
                root=root,
                device=device,
                dtype=dtype,
                tokenizer=CLIPTokenizer(root / "tokenizer", max_length=128),
                tokenizer_2=CLIPTokenizer(root / "tokenizer_2", max_length=128),
                tokenizer_3=T5Tokenizer(root / "tokenizer_3"),
                tokenizer_4=Llama3Tokenizer(root / "text_encoder_4"),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
                transformer_config=read_json(root / "transformer" / "config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error

    def _from_pretrained(self, cls: Any, name: str) -> Any:
        return cls.from_pretrained(self.root / name, device=self.device, dtype=self.dtype)


def _tokens(
    tokenizer: Any, prompt: str, max_length: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    input_ids, attention_mask = tokenizer([prompt], max_length)
    return input_ids.to(device), attention_mask.to(device)


@torch.inference_mode()
def encode_prompts(
    components: HiDreamComponents, prompts: list[str], max_sequence_length: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode each prompt with the four encoders, loading one encoder at a time.

    Returns the pooled CLIP embeddings ``(n, 2048)``, the T5 embeddings ``(n, length, 4096)`` and
    the stacked Llama layer outputs ``(layers, n, length, 4096)``. Every encoder sees the same
    prompt, as the pipeline does when ``prompt_2``, ``prompt_3`` and ``prompt_4`` are omitted.
    """
    device = components.device
    pooled = []
    for tokenizer, name in (
        (components.tokenizer, "text_encoder"),
        (components.tokenizer_2, "text_encoder_2"),
    ):
        encoder = components._from_pretrained(CLIPTextModelWithProjection, name)
        rows = []
        for prompt in prompts:
            tokenizer.max_length = min(max_sequence_length, CLIP_MAX_LENGTH)
            input_ids = torch.tensor([tokenizer(prompt)], device=device)
            rows.append(encoder(input_ids, output_hidden_states=True)[0])
        pooled.append(torch.cat(rows))
        del encoder
        _release()
    pooled_embeds = torch.cat(pooled, dim=-1)

    encoder = components._from_pretrained(T5EncoderModel, "text_encoder_3")
    rows = []
    for prompt in prompts:
        input_ids, mask = _tokens(components.tokenizer_3, prompt, max_sequence_length, device)
        rows.append(encoder(input_ids, attention_mask=mask)[0])
    t5_embeds = torch.cat(rows)
    del encoder
    _release()

    encoder = components._from_pretrained(LlamaTextModel, "text_encoder_4")
    rows = []
    for prompt in prompts:
        input_ids, mask = _tokens(components.tokenizer_4, prompt, max_sequence_length, device)
        rows.append(encoder(input_ids, attention_mask=mask))
    llama_embeds = torch.cat(rows, dim=1)
    del encoder
    _release()
    return pooled_embeds, t5_embeds, llama_embeds


def effective_size(width: int, height: int, vae_scale_factor: int) -> tuple[int, int]:
    """The pipeline rescales every request to about ``(128 * vae_scale_factor) ** 2`` pixels."""
    division = vae_scale_factor * 2
    maximum_area = (DEFAULT_SAMPLE_SIZE * vae_scale_factor) ** 2
    scale = math.sqrt(maximum_area / (width * height))
    return int(width * scale // division * division), int(height * scale // division * division)


@torch.inference_mode()
def denoise(
    components: HiDreamComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    latents: Tensor,
    max_sequence_length: int,
    noise_generator: torch.Generator | None = None,
) -> Tensor:
    use_cfg = guidance > 1.0
    prompts = [prompt, negative_prompt or ""] if use_cfg else [prompt]
    pooled, t5_embeds, llama_embeds = encode_prompts(components, prompts, max_sequence_length)
    count = latents.shape[0]
    pooled = pooled.repeat_interleave(count, dim=0)
    t5_embeds = t5_embeds.repeat_interleave(count, dim=0)
    llama_embeds = llama_embeds.repeat_interleave(count, dim=1)
    if use_cfg:
        # Rows are [prompt] * count then [negative] * count; the pipeline orders [negative, prompt].
        pooled = torch.cat([pooled[count:], pooled[:count]])
        t5_embeds = torch.cat([t5_embeds[count:], t5_embeds[:count]])
        llama_embeds = torch.cat([llama_embeds[:, count:], llama_embeds[:, :count]], dim=1)

    scheduler_class = components.scheduler_config.get("_class_name", "FlowMatchLCMScheduler")
    scheduler: FlowMatchLCMScheduler | UniPCMultistepScheduler
    if scheduler_class == "FlowMatchLCMScheduler":
        scheduler = FlowMatchLCMScheduler(components.scheduler_config)
        scheduler_name = "flow_match_lcm"
    elif scheduler_class == "UniPCMultistepScheduler":
        scheduler = UniPCMultistepScheduler(components.scheduler_config)
        scheduler_name = "unipc"
    else:
        raise ValueError(f"Unsupported HiDream scheduler {scheduler_class}")
    # The pipeline passes a resolution-dependent ``mu``; the configured static shift ignores it.
    scheduler.set_timesteps(steps)
    transformer = components._from_pretrained(HiDreamImageTransformer2DModel, "transformer")
    latents = latents.to(transformer.x_embedder.proj.weight.dtype)
    for timestep in scheduler.timesteps:
        model_input = torch.cat([latents] * 2) if use_cfg else latents
        noise = -transformer(
            model_input,
            timestep.expand(model_input.shape[0]),
            t5_embeds,
            llama_embeds,
            pooled,
        )
        if use_cfg:
            noise_unconditional, noise_text = noise.chunk(2)
            noise = noise_unconditional + guidance * (noise_text - noise_unconditional)
        if scheduler_name == "flow_match_lcm":
            latents = scheduler.step(noise, timestep, latents, generator=noise_generator)
        else:
            latents = scheduler.step(noise, timestep, latents)
    del transformer
    _release()
    return latents


@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: Tensor) -> Tensor:
    """Decode final latents to uint8 images of shape (batch, height, width, 3) on CPU."""
    latents = latents / vae.scaling_factor + (vae.shift_factor or 0.0)
    image = vae.decode(latents.to(vae.decoder.conv_in.weight.dtype))
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    if type(options[name]) is not int or options[name] <= 0:
        raise ValueError(f"{name} must be a positive int")


class HiDreamBackend(TextToImageBackend):
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
        self.components: HiDreamComponents | None = None
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
            raise TypeError(f"Unknown HiDream-I1 options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        try:
            transformer_config = read_json(self.source / "transformer" / "config.json")
        except FileNotFoundError:
            transformer_config = {"in_channels": 16}
        try:
            vae_config = read_json(self.source / "vae" / "config.json")
        except FileNotFoundError:
            vae_config = {"block_out_channels": [1, 1, 1, 1]}
        vae_scale = 2 ** (len(vae_config.get("block_out_channels", [1, 1, 1, 1])) - 1)
        if o["negative_prompt"] is not None and not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "num_images", "width", "height", "max_sequence_length"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        if o["max_sequence_length"] > 512:
            raise ValueError("max_sequence_length must be at most 512")
        guidance = o["guidance"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, int | float)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError("guidance must be a finite number >= 0")
        # Like the pipeline, rescale the request to the model's pixel budget (about one megapixel).
        width, height = effective_size(o["width"], o["height"], vae_scale)
        if min(width, height) < 2 * vae_scale:
            raise ValueError("width and height are too far from a square for the pixel budget")
        o["requested_width"], o["requested_height"] = o["width"], o["height"]
        o["width"], o["height"] = width, height
        seed, latents = o["seed"], o["latents"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        shape = (
            o["num_images"],
            int(transformer_config.get("in_channels", 64)),
            height // vae_scale,
            width // vae_scale,
        )
        if latents is not None and (
            not isinstance(latents, Tensor)
            or not latents.is_floating_point()
            or tuple(latents.shape) != shape
        ):
            raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        o["latent_shape"] = shape
        return o

    def _load(self) -> None:
        self.components = HiDreamComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the HiDream-I1 architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        seed, latents = o["seed"], o["latents"]
        if latents is None:
            if seed is None:
                seed = secrets.randbits(63)
            latents = torch.randn(
                o["latent_shape"],
                generator=torch.Generator("cpu").manual_seed(seed),
                dtype=torch.float32,
            )
        # The per-step noise comes from its own generator; global random state is never used.
        noise_seed = seed if seed is not None else secrets.randbits(63)
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
            steps=o["steps"],
            guidance=float(o["guidance"]),
            latents=latents.to(torch.float32).to(self.device),
            max_sequence_length=o["max_sequence_length"],
            noise_generator=torch.Generator("cpu").manual_seed(noise_seed),
        )
        pixels = decode_latents(self.components.vae, final)
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
            "scheduler": (
                "flow_match_lcm"
                if self.components.scheduler_config.get("_class_name", "FlowMatchLCMScheduler")
                == "FlowMatchLCMScheduler"
                else "unipc"
            ),
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
