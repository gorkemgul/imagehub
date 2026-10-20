"""Show-o text-to-image backend with discrete image-token denoising."""

from __future__ import annotations

import math
import os
import secrets
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import NativeModel, load_weights, read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.magvit_v2 import MAGVITv2
from imagehub.models.show_o import ShowOModel
from imagehub.tokenization_codegen import CodeGenTokenizer
from imagehub.types import ImageResult, T2IRequest

OPTION_DEFAULTS: dict[str, Any] = {
    "guidance": 10.0,
    "temperature": 1.0,
    "steps": 18,
    "width": 512,
    "height": 512,
    "seed": None,
}
MASK_TOKEN_ID = 58497
LLM_VOCAB_SIZE = 50295
NUM_NEW_SPECIAL_TOKENS = 10
TEXT_LENGTH = 129
IMAGE_TOKENS = 1024
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
ModelT = TypeVar("ModelT", bound=NativeModel)


def create_attention_mask(sequence: Tensor, *, pad_id: int, soi_id: int, eoi_id: int) -> Tensor:
    """Create Show-o's causal-text/bidirectional-image additive attention mask."""
    _, length = sequence.shape
    padding = sequence == pad_id
    start_image = sequence == soi_id
    end_image = sequence == eoi_id
    in_image = (start_image.cumsum(1) > end_image.cumsum(1)) | start_image | end_image
    is_text = ~in_image
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=sequence.device))
    mask = is_text[:, :, None] & causal[None]
    bidirectional = (is_text | in_image)[:, :, None] & (is_text | in_image)[:, None, :]
    start_positions = torch.where(start_image)[1]
    for index in range(sequence.shape[0]):
        pad_positions = torch.where(padding[index])[0]
        if len(pad_positions):
            pad_end = pad_positions[-1]
            mask[index, pad_end + 1 :, : pad_end + 1] = False
            bidirectional[index, start_positions[index] :, pad_positions] = False
    mask[in_image] = bidirectional[in_image]
    inverted = (~mask).to(sequence.dtype)
    inverted.masked_fill_(inverted.bool(), torch.iinfo(sequence.dtype).min)
    return inverted.unsqueeze(1)


def prompt_ids(tokenizer: CodeGenTokenizer, prompt: str, image_tokens: Tensor) -> Tensor:
    text = tokenizer.tokenize(prompt)
    if not text or text[0] != tokenizer.eos_id:
        text = [tokenizer.eos_id] + text
    text = [tokenizer.special_ids["<|t2i|>"], *text, tokenizer.eos_id]
    if len(text) <= TEXT_LENGTH:
        text = [tokenizer.pad_id] * (TEXT_LENGTH - len(text)) + text
    else:
        text = text[: TEXT_LENGTH - 1] + [tokenizer.eos_id]
    prefix = torch.tensor(text, dtype=torch.long, device=image_tokens.device)
    soi = torch.tensor([tokenizer.special_ids["<|soi|>"]], device=image_tokens.device)
    eoi = torch.tensor([tokenizer.special_ids["<|eoi|>"]], device=image_tokens.device)
    return torch.cat((prefix, soi, image_tokens, eoi)).unsqueeze(0)


def _gumbel_noise(tensor: Tensor, generator: torch.Generator) -> Tensor:
    uniform = torch.zeros_like(tensor, device="cpu").uniform_(0, 1, generator=generator)
    return -torch.log(-torch.log(uniform.clamp(min=1e-20))).to(tensor.device)


def mask_by_random_topk(
    mask_len: Tensor, probabilities: Tensor, temperature: float, generator: torch.Generator
) -> Tensor:
    confidence = torch.log(probabilities.clamp(min=1e-20)) + temperature * _gumbel_noise(
        probabilities, generator
    )
    cutoff = torch.gather(torch.sort(confidence, dim=-1).values, 1, mask_len.long())
    return confidence < cutoff


@torch.inference_mode()
def generate_tokens(
    model: ShowOModel,
    input_ids: Tensor,
    unconditional_ids: Tensor | None,
    attention_mask: Tensor,
    *,
    guidance: float,
    temperature: float,
    steps: int,
    generator: torch.Generator,
    image_tokens: int = IMAGE_TOKENS,
) -> Tensor:
    token_ids = input_ids[:, -(image_tokens + 1) : -1].clone()
    token_ids = torch.where(
        token_ids == MASK_TOKEN_ID,
        MASK_TOKEN_ID,
        token_ids - LLM_VOCAB_SIZE - NUM_NEW_SPECIAL_TOKENS,
    )
    unconditional_prefix = (
        unconditional_ids[:, :TEXT_LENGTH] if unconditional_ids is not None else None
    )
    for step in range(steps):
        if unconditional_ids is not None and guidance > 0:
            assert unconditional_prefix is not None
            unconditional_ids = torch.cat((unconditional_prefix, input_ids[:, TEXT_LENGTH:]), 1)
            conditional, unconditional = model(
                torch.cat((input_ids, unconditional_ids)), attention_mask
            ).chunk(2)
            logits = (1 + guidance) * conditional - guidance * unconditional
        else:
            logits = model(input_ids, attention_mask)
        logits = logits[:, -(image_tokens + 1) : -1, LLM_VOCAB_SIZE + NUM_NEW_SPECIAL_TOKENS : -1]
        probabilities = logits.softmax(-1)
        sampled = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]).cpu(), 1, generator=generator
        )[:, 0].to(probabilities.device)
        sampled = sampled.view(*logits.shape[:-1])
        unknown = token_ids == MASK_TOKEN_ID
        sampled = torch.where(unknown, sampled, token_ids)
        ratio = (step + 1) / steps
        mask_ratio = math.cos(ratio * math.pi * 0.5)
        selected = torch.gather(probabilities, -1, sampled[..., None]).squeeze(-1)
        selected = torch.where(unknown, selected, torch.finfo(selected.dtype).max)
        mask_len = torch.tensor(math.floor(image_tokens * mask_ratio), device=logits.device)
        mask_len = torch.maximum(
            torch.tensor([1], device=logits.device),
            torch.minimum(unknown.sum(-1, keepdim=True) - 1, mask_len),
        )
        temperature *= 1.0 - ratio
        masking = mask_by_random_topk(mask_len, selected, temperature, generator)
        input_ids[:, -(image_tokens + 1) : -1] = torch.where(
            masking,
            MASK_TOKEN_ID,
            sampled + LLM_VOCAB_SIZE + NUM_NEW_SPECIAL_TOKENS,
        )
        token_ids = torch.where(masking, MASK_TOKEN_ID, sampled)
    return sampled


@dataclass
class ShowOComponents:
    tokenizer: CodeGenTokenizer
    model: ShowOModel
    vq: MAGVITv2

    @classmethod
    def load(
        cls, directory: str | os.PathLike[str], *, device: torch.device, dtype: torch.dtype
    ) -> ShowOComponents:
        root = Path(directory)
        required = (
            "config.json",
            "pytorch_model.safetensors",
            "vq/config.json",
            "vq/pytorch_model.safetensors",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/added_tokens.json",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise CheckpointError(f"{root} is not a Show-o snapshot (missing {', '.join(missing)})")
        tokenizer = CodeGenTokenizer(root / "tokenizer")
        model = _load_component(ShowOModel, root, device, dtype)
        vq = _load_component(MAGVITv2, root / "vq", device, dtype)
        return cls(tokenizer, model, vq)


def _load_component(
    model_cls: type[ModelT], directory: Path, device: torch.device, dtype: torch.dtype
) -> ModelT:
    with torch.device("meta"):
        model = model_cls(read_json(directory / "config.json"))
    model.load_checkpoint(load_weights(directory / "pytorch_model.safetensors"))
    model.init_non_persistent_buffers()
    return model.to(device=device, dtype=dtype).eval()


class ShowOBackend(TextToImageBackend):
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
        self.components: ShowOComponents | None = None
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
            raise TypeError(f"Unknown Show-o options: {', '.join(unknown)}")
        values = self.option_defaults | merged
        for name in ("width", "height"):
            if values[name] != 512:
                raise ValueError(f"{name} must be 512 for Show-o 512")
        if type(values["steps"]) is not int or values["steps"] <= 0:
            raise ValueError("steps must be a positive int")
        for name in ("guidance", "temperature"):
            value = values[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if values["temperature"] <= 0 or values["guidance"] < 0:
            raise ValueError("temperature must be > 0 and guidance must be >= 0")
        seed = values["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        return values

    def _load(self) -> None:
        self.components = ShowOComponents.load(self.source, device=self.device, dtype=self.dtype)
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    @torch.inference_mode()
    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed = options["seed"] if options["seed"] is not None else secrets.randbits(63)
        generator = torch.Generator("cpu").manual_seed(seed)
        image = torch.full((IMAGE_TOKENS,), MASK_TOKEN_ID, device=self.device)
        conditional = prompt_ids(self.components.tokenizer, inputs.prompt, image)
        unconditional = (
            prompt_ids(self.components.tokenizer, "", image) if options["guidance"] > 0 else None
        )
        mask_input = (
            torch.cat((conditional, unconditional)) if unconditional is not None else conditional
        )
        attention_mask = create_attention_mask(
            mask_input,
            pad_id=self.components.tokenizer.pad_id,
            soi_id=self.components.tokenizer.special_ids["<|soi|>"],
            eoi_id=self.components.tokenizer.special_ids["<|eoi|>"],
        )
        tokens = generate_tokens(
            self.components.model,
            conditional,
            unconditional,
            attention_mask,
            guidance=float(options["guidance"]),
            temperature=float(options["temperature"]),
            steps=options["steps"],
            generator=generator,
            image_tokens=IMAGE_TOKENS,
        ).clamp(0, 8191)
        decoded = self.components.vq.decode_code(tokens)
        pixels = ((decoded + 1) / 2).clamp(0, 1).mul(255).permute(0, 2, 3, 1).to(torch.uint8)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "guidance": float(options["guidance"]),
            "temperature": float(options["temperature"]),
            "steps": options["steps"],
            "width": 512,
            "height": 512,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
