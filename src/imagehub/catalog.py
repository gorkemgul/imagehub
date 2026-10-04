"""Published, distinct model checkpoints, separate from reusable architecture backends.

A model is a released weight set, not an alias, precision variant or configuration.
Families share native code. ``catalog_id`` links each model to the entry in
awesome-text-to-image-architectures that describes its architecture, and ``revision``
is the commit whose files passed the parity tests in docs/validation.json.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from imagehub.types import Task

PROVENANCES = ("author", "community-mirror")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    task: Task
    family: str
    source: str
    repo_id: str | None
    files: tuple[str, ...]
    format: str
    catalog_id: str
    revision: str
    provenance: str
    config: Mapping[str, Any] = field(default_factory=dict)
    gated: bool = False
    license: str = "See model card"

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))
        object.__setattr__(self, "files", tuple(self.files))
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be a 40-character lowercase commit hash")
        if self.provenance not in PROVENANCES:
            raise ValueError(f"provenance must be one of {', '.join(PROVENANCES)}")


_entries: list[ModelSpec] = []

# Stable Diffusion 1.x: one architecture, five released weight sets. Only fp32 EMA weights
# are fetched; fp16, non-EMA, single-file .ckpt and safety checker files are excluded.
_SD1_COMMON = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
    "text_encoder/config.json",
    "unet/config.json",
    "vae/config.json",
)
_SD1_SAFETENSORS = (
    "text_encoder/model.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
)
# v1.1–v1.3 publish fp32 weights only as tensor-only pickles (loaded with weights_only=True).
_SD1_BIN = (
    "text_encoder/pytorch_model.bin",
    "unet/diffusion_pytorch_model.bin",
    "vae/diffusion_pytorch_model.bin",
)
for _version, _repo, _revision, _provenance, _weights in (
    (
        "1",
        "CompVis/stable-diffusion-v1-1",
        "574555a657f0c8d6e970a02c7e095e9d4220dcbf",
        "author",
        _SD1_BIN,
    ),
    (
        "2",
        "CompVis/stable-diffusion-v1-2",
        "f519eebf1372598b85ab317ec273a63ebeddd5eb",
        "author",
        _SD1_BIN,
    ),
    (
        "3",
        "CompVis/stable-diffusion-v1-3",
        "159f26fccae6e3ea26c797ad24c5e7033da5b1ee",
        "author",
        _SD1_BIN,
    ),
    (
        "4",
        "CompVis/stable-diffusion-v1-4",
        "133a221b8aa7292a167afc5127cb63fb5005638b",
        "author",
        _SD1_SAFETENSORS,
    ),
    (
        "5",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        "community-mirror",
        _SD1_SAFETENSORS,
    ),
):
    _entries.append(
        ModelSpec(
            id=f"sd-v1-{_version}",
            task=Task.T2I,
            family="stable-diffusion",
            source=f"https://huggingface.co/{_repo}",
            repo_id=_repo,
            files=_SD1_COMMON + _weights,
            format="diffusers",
            catalog_id="stable-diffusion-1",
            revision=_revision,
            provenance=_provenance,
            license="CreativeML-OpenRAIL-M",
        )
    )

_entries.append(
    ModelSpec(
        id="sdxl-base-1.0",
        task=Task.T2I,
        family="stable-diffusion-xl",
        source="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        revision="462165984030d82259a11f4367a4eed129e94a7b",
        provenance="author",
        catalog_id="sdxl",
        format="diffusers",
        license="CreativeML-OpenRAIL++-M",
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer_2/merges.txt",
            "tokenizer_2/special_tokens_map.json",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/vocab.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

MODEL_CATALOG: Mapping[str, ModelSpec] = MappingProxyType({entry.id: entry for entry in _entries})
if len(MODEL_CATALOG) != len(_entries):
    raise RuntimeError("Duplicate model IDs")


def model_info(model_id: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[model_id]
    except KeyError:
        available = ", ".join(sorted(MODEL_CATALOG))
        raise ValueError(f"Unknown model {model_id!r}. Available: {available}") from None


def available_models(task: Task | str | None = None) -> dict[str, tuple[str, ...]]:
    tasks = (Task(task),) if task is not None else tuple(Task)
    return {
        kind.value: tuple(sorted(spec.id for spec in MODEL_CATALOG.values() if spec.task == kind))
        for kind in tasks
    }
