"""Janus-Pro text-to-image backend over a local transformers-layout snapshot."""

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
from imagehub.models.janus import JanusModel
from imagehub.models.janus_vq import postprocess_janus
from imagehub.models.llama import LlamaCache
from imagehub.tokenization_deepseek import DeepSeekTokenizer
from imagehub.types import ImageResult, T2IRequest

OPTION_DEFAULTS: dict[str, Any] = {
    "guidance": 5.0,
    "temperature": 1.0,
    "do_sample": True,
    "width": 384,
    "height": 384,
    "num_images": 1,
    "seed": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "janus-pro-1b"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


@dataclass
class JanusComponents:
    tokenizer: DeepSeekTokenizer
    model: JanusModel
    generation_config: dict[str, Any]

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> JanusComponents:
        root = Path(directory)
        required = ("config.json", "generation_config.json", "tokenizer.json", "model.safetensors")
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Janus snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            return cls(
                tokenizer=DeepSeekTokenizer(root),
                model=JanusModel.from_pretrained(root, device=device, dtype=dtype),
                generation_config=read_json(root / "generation_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error


def _top_k(scores: Tensor, top_k: int) -> Tensor:
    top_k = min(top_k, scores.shape[-1])
    threshold = torch.topk(scores, top_k)[0][..., -1, None]
    return scores.masked_fill(scores < threshold, -float("inf"))


def _top_p(scores: Tensor, top_p: float) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) <= 1 - top_p
    remove[..., -1:] = False
    remove = remove.scatter(1, sorted_indices, remove)
    return scores.masked_fill(remove, -float("inf"))


@torch.inference_mode()
def generate_image_tokens(
    components: JanusComponents,
    *,
    prompt: str,
    guidance: float,
    temperature: float,
    do_sample: bool,
    num_images: int,
    generator: torch.Generator,
) -> Tensor:
    """Generate Janus image tokens with the transformers CFG batch layout."""
    model = components.model
    config = components.generation_config
    input_ids, attention_mask = components.tokenizer(components.tokenizer.image_prompt(prompt))
    device = model.language_model.embed_tokens.weight.device
    input_ids = input_ids.to(device).repeat_interleave(num_images, dim=0)
    attention_mask = attention_mask.to(device).repeat_interleave(num_images, dim=0)
    batch_size, sequence_length = input_ids.shape
    use_cfg = guidance > 1
    input_tokens = input_ids.repeat(2, 1) if use_cfg else input_ids
    if use_cfg:
        bos_token_id = config["bos_token_id"]
        boi_token_id = config["generation_kwargs"]["boi_token_id"]
        unconditional = input_tokens[batch_size:]
        mask = (unconditional != bos_token_id) & (unconditional != boi_token_id)
        unconditional.masked_fill_(mask, config["pad_token_id"])
        attention_mask = attention_mask.repeat(2, 1)

    inputs_embeds = model.language_model.embed_tokens(input_tokens)
    num_image_tokens = model.vqmodel.quantize.quant_state_dims[0] ** 2
    max_cache_len = max(config.get("max_length", 20), num_image_tokens + sequence_length)
    cache = LlamaCache(model.language_model.config, max_cache_len)
    generated_tokens = torch.zeros(
        (batch_size, num_image_tokens), dtype=input_ids.dtype, device=device
    )
    top_k = config.get("top_k", 50)
    top_p = config.get("top_p", 1.0)

    for index in range(num_image_tokens):
        if index == 0:
            cache_position = torch.arange(sequence_length, device=device)
        else:
            cache_position = torch.tensor([sequence_length + index - 1], device=device)
        hidden_states = model.language_model(inputs_embeds, attention_mask, cache, cache_position)
        scores = model.generation_head(hidden_states[:, -1, :].clone())
        if use_cfg:
            conditional, unconditional = scores.split(batch_size, dim=0)
            scores = unconditional + (conditional - unconditional) * guidance
        if do_sample:
            if temperature != 1.0:
                scores = scores / temperature
            if top_k is not None and top_k > 0:
                scores = _top_k(scores, int(top_k))
            if top_p is not None and top_p < 1.0:
                scores = _top_p(scores, float(top_p))
            probabilities = torch.softmax(scores, dim=-1)
            next_token = (
                torch.multinomial(probabilities.cpu(), num_samples=1, generator=generator)
                .squeeze(-1)
                .to(device)
            )
        else:
            next_token = torch.argmax(scores, dim=-1)
        generated_tokens[:, index] = next_token
        embedded_tokens = torch.cat([next_token, next_token]) if use_cfg else next_token
        inputs_embeds = model.prepare_embeddings_for_image_generation(embedded_tokens.unsqueeze(-1))
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
    return generated_tokens


class JanusBackend(TextToImageBackend):
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
        self.components: JanusComponents | None = None
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
            raise TypeError(f"Unknown Janus options: {', '.join(unknown)}")
        values = self.option_defaults | merged
        for name in ("width", "height"):
            if values[name] != 384:
                raise ValueError(f"{name} must be 384 for Janus")
        if type(values["num_images"]) is not int or values["num_images"] <= 0:
            raise ValueError("num_images must be a positive int")
        for name in ("guidance", "temperature"):
            value = values[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite number > 0")
        if type(values["do_sample"]) is not bool:
            raise TypeError("do_sample must be a bool")
        seed = values["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        return values

    def _load(self) -> None:
        self.components = JanusComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the Janus architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    @torch.inference_mode()
    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed = options["seed"]
        if seed is None:
            seed = secrets.randbits(63)
        tokens = generate_image_tokens(
            self.components,
            prompt=inputs.prompt,
            guidance=float(options["guidance"]),
            temperature=float(options["temperature"]),
            do_sample=options["do_sample"],
            num_images=options["num_images"],
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        pixels = postprocess_janus(self.components.model.decode_image_tokens(tokens))
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "guidance": float(options["guidance"]),
            "temperature": float(options["temperature"]),
            "do_sample": options["do_sample"],
            "width": options["width"],
            "height": options["height"],
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
