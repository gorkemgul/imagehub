"""LlamaGen text-to-image backend: Flan-T5-XL captions, an autoregressive GPT, a VQ-16 decoder."""

from __future__ import annotations

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
from torch.nn import functional as F

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import NativeModel, load_weights, read_json, read_snapshot
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.llamagen import LlamaGenGPT, gpt_xl_config
from imagehub.models.llamagen_vq import LlamaGenVQ
from imagehub.models.t5 import T5EncoderModel
from imagehub.tokenization_t5 import T5Tokenizer
from imagehub.types import ImageResult, T2IRequest

VQ_FILE = "vq_ds16_t2i.pt"
TEXT_ENCODER_DIR = "text_encoder"
CAPTION_LENGTH = 120
DOWNSAMPLE = 16
OPTION_DEFAULTS: dict[str, Any] = {
    "cfg_scale": 7.5,
    "temperature": 1.0,
    "top_k": 1000,
    "top_p": 1.0,
    "cfg_interval": -1,
    "width": None,
    "height": None,
    "seed": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_GPT_FILE = re.compile(r"t2i_XL_stage\d+_(\d+)\.pt")


def _hint(root: Path, model_id: str | None) -> str:
    model = model_id or "llamagen-t2i-xl-256"
    return f"Download a snapshot with: python -m imagehub.pretrained {model} {root}"


def find_gpt_checkpoint(root: Path, model_id: str | None = None) -> tuple[Path, int]:
    """The GPT checkpoint of a snapshot and the image size it generates."""
    spec = MODEL_CATALOG.get(model_id) if model_id is not None else None
    if spec is not None and "gpt_checkpoint" in spec.config:
        name = str(spec.config["gpt_checkpoint"])
        found = _GPT_FILE.fullmatch(name)
        assert found is not None
        return root / name, int(found.group(1))
    candidates = sorted(
        path for path in root.glob("t2i_XL_stage*_*.pt") if _GPT_FILE.fullmatch(path.name)
    )
    if len(candidates) != 1:
        raise CheckpointError(
            f"{root} must hold exactly one LlamaGen GPT checkpoint (t2i_XL_stage*_<size>.pt), "
            f"found {len(candidates)}. " + _hint(root, model_id)
        )
    found = _GPT_FILE.fullmatch(candidates[0].name)
    assert found is not None
    return candidates[0], int(found.group(1))


def _load_checkpoint_model(
    model: NativeModel, path: Path, device: torch.device, dtype: torch.dtype
) -> None:
    """Load a published ``{"model": state_dict}`` pickle into a meta-device model strictly."""
    # Fresh (aligned) copies, as ``load_state_dict`` into an initialised model makes.
    state = {
        name: tensor.clone() for name, tensor in load_weights(path, container_key="model").items()
    }
    model.load_checkpoint(state)
    model.init_non_persistent_buffers()
    model.to(device=device, dtype=dtype).eval()


@dataclass
class LlamaGenComponents:
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel
    gpt: LlamaGenGPT
    vq: LlamaGenVQ
    image_size: int

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str | None = None,
    ) -> LlamaGenComponents:
        root = Path(directory)
        gpt_path, image_size = find_gpt_checkpoint(root, model_id)
        missing = [
            name
            for name in (VQ_FILE, gpt_path.name, f"{TEXT_ENCODER_DIR}/config.json")
            if not (root / name).is_file()
        ]
        if missing:
            raise CheckpointError(
                f"{root} is not a LlamaGen snapshot (missing {', '.join(missing)}). "
                + _hint(root, model_id)
            )
        try:
            tokenizer = T5Tokenizer(root / TEXT_ENCODER_DIR)
            text_encoder = T5EncoderModel.from_pretrained(
                root / TEXT_ENCODER_DIR, device=device, dtype=dtype
            )
            latent_size = image_size // DOWNSAMPLE
            # The published checkpoints carry no configuration; an optional ``config.json`` next
            # to them may override the architecture (used by small random-weight snapshots).
            overrides = read_json(root / "config.json") if (root / "config.json").is_file() else {}
            with torch.device("meta"):
                gpt = LlamaGenGPT(gpt_xl_config(latent_size**2) | overrides.get("gpt", {}))
                vq = LlamaGenVQ(overrides.get("vq", {}))
            _load_checkpoint_model(gpt, gpt_path, device, dtype)
            _load_checkpoint_model(vq, root / VQ_FILE, device, dtype)
        except FileNotFoundError as error:
            raise CheckpointError(
                f"{error.filename or error} is missing. " + _hint(root, model_id)
            ) from error
        return cls(tokenizer, text_encoder, gpt, vq, image_size)


# Sampling: ``autoregressive/models/generate.py`` with the random draws taken from an explicit
# generator (the reference draws from the global generator after ``torch.manual_seed``).


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter ``(batch, vocab)`` logits in place with top-k and nucleus (top-p) filtering."""
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def sample(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Draw one token per row from the last position of ``(batch, length, vocab)`` logits."""
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    # The generator lives on the CPU, as the reference's global generator does.
    idx = torch.multinomial(probs.cpu(), num_samples=1, generator=generator).to(probs.device)
    return idx, probs


def _guided(logits: Tensor, cfg_scale: float) -> Tensor:
    cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0)
    return uncond_logits + (cond_logits - uncond_logits) * cfg_scale


def prefill(
    model: LlamaGenGPT,
    cond_idx: Tensor,
    input_pos: Tensor,
    cfg_scale: float,
    **sampling: Any,
) -> Tensor:
    logits = model(None, cond_idx, input_pos)
    if cfg_scale > 1.0:
        logits = _guided(logits, cfg_scale)
    return sample(logits, **sampling)[0]


def decode_one_token(
    model: LlamaGenGPT,
    x: Tensor,
    input_pos: Tensor,
    cfg_scale: float,
    cfg_flag: bool,
    **sampling: Any,
) -> tuple[Tensor, Tensor]:
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        logits = model(torch.cat([x, x]), None, input_pos)
        if cfg_flag:
            logits = _guided(logits, cfg_scale)
        else:
            logits = torch.split(logits, len(logits) // 2, dim=0)[0]
    else:
        logits = model(x, None, input_pos)
    return sample(logits, **sampling)


def decode_n_tokens(
    model: LlamaGenGPT,
    cur_token: Tensor,
    input_pos: Tensor,
    num_new_tokens: int,
    cfg_scale: float,
    cfg_interval: int,
    **sampling: Any,
) -> list[Tensor]:
    new_tokens = []
    cfg_flag = True
    for i in range(num_new_tokens):
        if cfg_interval > -1 and i > cfg_interval:
            cfg_flag = False
        next_token, _ = decode_one_token(
            model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        cur_token = next_token.view(-1, 1)
    return new_tokens


@torch.inference_mode()
def generate(
    model: LlamaGenGPT,
    cond: Tensor,
    max_new_tokens: int,
    emb_masks: Tensor | None = None,
    cfg_scale: float = 1.0,
    cfg_interval: int = -1,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``max_new_tokens`` image tokens per caption; returns ``(batch, tokens)`` int32 ids.

    With ``cfg_scale > 1`` the unconditional half of the guidance batch uses the learned
    ``uncond_embedding`` captions.
    """
    sampling = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "generator": generator,
    }
    if cfg_scale > 1.0:
        cond_null = torch.zeros_like(cond) + cast(Tensor, model.cls_embedding.uncond_embedding)
        cond_combined = torch.cat([cond, cond_null])
    else:
        cond_combined = cond
    caption_length = cond.shape[1]
    total = caption_length + max_new_tokens
    batch = cond.shape[0]
    device = cond.device
    model.setup_caches(
        max_batch_size=batch * 2 if cfg_scale > 1.0 else batch,
        max_seq_length=total,
        dtype=model.tok_embeddings.weight.dtype,
        device=device,
    )
    if emb_masks is not None:
        assert emb_masks.shape[0] == batch
        assert emb_masks.shape[-1] == caption_length
        model.apply_caption_mask(
            torch.cat([emb_masks, emb_masks]) if cfg_scale > 1.0 else emb_masks
        )

    seq = torch.empty((batch, total), dtype=torch.int, device=device)
    input_pos = torch.arange(0, caption_length, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling)
    seq[:, caption_length : caption_length + 1] = next_token
    input_pos = torch.tensor([caption_length], device=device, dtype=torch.int)
    generated = decode_n_tokens(
        model, next_token, input_pos, max_new_tokens - 1, cfg_scale, cfg_interval, **sampling
    )
    if generated:
        seq[:, caption_length + 1 :] = torch.cat(generated, dim=1)
    return seq[:, caption_length:]


# Captions.


@torch.inference_mode()
def encode_prompts(
    components: LlamaGenComponents, prompts: list[str], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Left-padded caption embeddings ``(batch, 120, 2048)`` and the matching masks.

    The prompt is lower-cased and stripped (``T5Embedder`` without text preprocessing), padded to
    120 tokens, encoded with the attention mask, then the padding is moved to the front by
    flipping the masks and rotating the embeddings, and padded positions are zeroed.
    """
    texts = [prompt.lower().strip() for prompt in prompts]
    input_ids, mask = components.tokenizer(texts, CAPTION_LENGTH)
    emb_masks = mask.to(device=device, dtype=torch.long)
    embeddings = components.text_encoder(input_ids.to(device), emb_masks)[0].detach()
    new_masks = torch.flip(emb_masks, dims=[-1])
    rotated = []
    for embedding, row in zip(embeddings, emb_masks, strict=True):
        valid = int(row.sum().item())
        rotated.append(torch.cat([embedding[valid:], embedding[:valid]]))
    new_embeddings = torch.stack(rotated)
    return new_embeddings * new_masks[:, :, None], new_masks


def to_uint8(samples: Tensor) -> Tensor:
    """Pixels in [-1, 1] to uint8 ``(batch, height, width, 3)`` as torchvision's ``save_image``."""
    low, high = -1.0, 1.0
    image = samples.detach().float().clone().clamp_(min=low, max=high)
    image.sub_(low).div_(max(high - low, 1e-5))
    return image.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8)


def _finite_number(value: Any, name: str, *, minimum: float, strict: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < minimum
        or (strict and value == minimum)
    ):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be a finite number {relation} {minimum:g}")


class LlamaGenBackend(TextToImageBackend):
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
        self.components: LlamaGenComponents | None = None
        self.snapshot: dict[str, Any] | None = None
        defaults = (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {}))
            if model_id is not None and model_id in MODEL_CATALOG
            else {}
        )
        self.option_defaults = OPTION_DEFAULTS | defaults

    def _image_size(self) -> int:
        return find_gpt_checkpoint(self.source, self.model_id)[1]

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown LlamaGen options: {', '.join(unknown)}")
        o = self.option_defaults | merged
        _finite_number(o["cfg_scale"], "cfg_scale", minimum=0)
        _finite_number(o["temperature"], "temperature", minimum=0, strict=True)
        if type(o["top_k"]) is not int or o["top_k"] < 0:
            raise ValueError("top_k must be an int >= 0")
        _finite_number(o["top_p"], "top_p", minimum=0, strict=True)
        if o["top_p"] > 1:
            raise ValueError("top_p must be at most 1")
        if type(o["cfg_interval"]) is not int or o["cfg_interval"] < -1:
            raise ValueError("cfg_interval must be an int >= -1")
        seed = o["seed"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        size = self._size_hint()
        for name in ("width", "height"):
            if o[name] is None:
                o[name] = size
            if type(o[name]) is not int or o[name] <= 0:
                raise ValueError(f"{name} must be a positive int")
            if size is not None and o[name] != size:
                raise ValueError(f"{name} must be {size} for this LlamaGen checkpoint")
        return o

    def _size_hint(self) -> int | None:
        try:
            return self._image_size()
        except CheckpointError:
            return None

    def _load(self) -> None:
        self.components = LlamaGenComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)
        recorded = (self.snapshot or {}).get("model_id")
        if recorded is not None and self.model_id is not None and recorded != self.model_id:
            warnings.warn(
                f"{self.source} is a snapshot of {recorded!r} but {self.model_id!r} was selected; "
                "both share the LlamaGen architecture, so it was loaded",
                UserWarning,
                stacklevel=2,
            )

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    @torch.inference_mode()
    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        components = self.components
        if o["width"] is None or o["height"] is None:
            o = o | {"width": components.image_size, "height": components.image_size}
        if (o["width"], o["height"]) != (components.image_size,) * 2:
            raise ValueError(f"width and height must be {components.image_size}")
        seed = o["seed"]
        if seed is None:
            seed = secrets.randbits(63)
        c_indices, c_masks = encode_prompts(components, [inputs.prompt], self.device)
        latent_size = components.image_size // DOWNSAMPLE
        tokens = generate(
            components.gpt,
            c_indices,
            latent_size**2,
            c_masks,
            cfg_scale=float(o["cfg_scale"]),
            cfg_interval=o["cfg_interval"],
            temperature=float(o["temperature"]),
            top_k=o["top_k"],
            top_p=float(o["top_p"]),
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        shape = (
            len(c_indices),
            components.vq.config["codebook_embed_dim"],
            latent_size,
            latent_size,
        )
        pixels = to_uint8(components.vq.decode_code(tokens, shape))
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "cfg_scale": float(o["cfg_scale"]),
            "temperature": float(o["temperature"]),
            "top_k": o["top_k"],
            "top_p": float(o["top_p"]),
            "cfg_interval": o["cfg_interval"],
            "width": o["width"],
            "height": o["height"],
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(pixel) for pixel in pixels), metadata)
