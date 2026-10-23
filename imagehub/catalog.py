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
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any

from imagehub.types import Task

PROVENANCES = ("author", "community-mirror", "community-conversion")


def _released_reference() -> dict[str, dict[str, str]]:
    return {
        "diffusers": {"version": "0.40.0"},
        "transformers": {"version": "5.18.0"},
    }


def _safe_relative(name: str, field_name: str) -> None:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"{field_name} path {name!r} is not a safe relative path")


@dataclass(frozen=True)
class SubRepo:
    repo_id: str
    revision: str
    files: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))
        if not self.repo_id or self.repo_id.strip() != self.repo_id:
            raise ValueError("repo_id must be nonempty and have no surrounding whitespace")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be a 40-character lowercase commit hash")
        if len(set(self.files)) != len(self.files):
            raise ValueError("subrepository files must not contain duplicates")
        for name in self.files:
            _safe_relative(name, "subrepository file")


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
    reference: Mapping[str, Mapping[str, str]] = field(default_factory=_released_reference)
    gated: bool = False
    license: str = "See model card"
    base: str | None = None
    base_exclude: tuple[str, ...] = ()
    overlay: Mapping[str, str] = field(default_factory=dict)
    subfolders: Mapping[str, SubRepo] = field(default_factory=dict)
    image_input: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))
        object.__setattr__(
            self,
            "reference",
            MappingProxyType(
                {
                    package: MappingProxyType(dict(details))
                    for package, details in self.reference.items()
                }
            ),
        )
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "base_exclude", tuple(self.base_exclude))
        object.__setattr__(self, "overlay", MappingProxyType(dict(self.overlay)))
        object.__setattr__(self, "subfolders", MappingProxyType(dict(self.subfolders)))
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be a 40-character lowercase commit hash")
        if self.provenance not in PROVENANCES:
            raise ValueError(f"provenance must be one of {', '.join(PROVENANCES)}")
        if self.image_input not in (None, "optional", "required"):
            raise ValueError("image_input must be None, 'optional' or 'required'")
        if self.base is not None and not any(entry.id == self.base for entry in _entries):
            raise ValueError(f"base model {self.base!r} does not exist")
        if self.overlay and self.base is None:
            raise ValueError("overlay requires a base model")
        if self.base_exclude and self.base is None:
            raise ValueError("base_exclude requires a base model")
        for name in self.base_exclude:
            _safe_relative(name, "base_exclude")
        for destination, source in self.overlay.items():
            for name in (destination, source):
                _safe_relative(name, "overlay")
            if source not in self.files:
                raise ValueError(f"overlay source {source!r} is not listed in files")
        for subfolder, repository in self.subfolders.items():
            _safe_relative(subfolder, "subfolder")
            if not isinstance(repository, SubRepo):
                raise TypeError("subfolders values must be SubRepo objects")


_entries: list[ModelSpec] = []

_JOYAI_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "processor/chat_template.jinja",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "transformer/diffusion_pytorch_model-00001-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00006-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00007-of-00007.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
_JOYAI_PLUS_FILES = _JOYAI_FILES + ("processor/processor_config.json",)
_JOYAI_BASE_FILES = _JOYAI_FILES + (
    "tokenizer/added_tokens.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/vocab.json",
    "processor/added_tokens.json",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
)

_entries.extend(
    (
        ModelSpec(
            id="joyai-image-edit",
            task=Task.T2I,
            family="joyai-image",
            source="https://huggingface.co/jdopensource/JoyAI-Image-Edit-Diffusers",
            repo_id="jdopensource/JoyAI-Image-Edit-Diffusers",
            revision="4b41fb25d961f37668750178ccbb380da326201c",
            provenance="author",
            catalog_id="joyai-image",
            format="diffusers",
            license="Apache-2.0",
            image_input="optional",
            config={"defaults": {"steps": 40, "guidance": 4.0}},
            files=_JOYAI_BASE_FILES,
        ),
        ModelSpec(
            id="joyai-image-edit-plus",
            task=Task.T2I,
            family="joyai-image",
            source="https://huggingface.co/jdopensource/JoyAI-Image-Edit-Plus-Diffusers",
            repo_id="jdopensource/JoyAI-Image-Edit-Plus-Diffusers",
            revision="c2686460c7b64d8aa11bc4d0da423fb316b33f9e",
            provenance="author",
            catalog_id="joyai-image",
            format="diffusers",
            license="Apache-2.0",
            image_input="required",
            config={"defaults": {"steps": 30, "guidance": 4.0}},
            files=_JOYAI_PLUS_FILES,
        ),
    )
)

_entries.append(
    ModelSpec(
        id="qwen-image",
        task=Task.T2I,
        family="qwen-image",
        source="https://huggingface.co/Qwen/Qwen-Image",
        repo_id="Qwen/Qwen-Image",
        revision="75e0b4be04f60ec59a75f475837eced720f823b6",
        provenance="author",
        catalog_id="qwen-image-2",
        format="diffusers",
        license="Apache-2.0",
        reference={"diffusers": {"version": "0.40.0"}, "transformers": {"version": "5.18.0"}},
        config={
            "defaults": {
                "steps": 50,
                "true_cfg": 4.0,
                "width": 1024,
                "height": 1024,
            }
        },
        files=(
            "model_index.json",
            "LICENSE",
            "scheduler/scheduler_config.json",
            "tokenizer/added_tokens.json",
            "tokenizer/chat_template.jinja",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00004.safetensors",
            "text_encoder/model-00002-of-00004.safetensors",
            "text_encoder/model-00003-of-00004.safetensors",
            "text_encoder/model-00004-of-00004.safetensors",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
            "transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
            "transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
            "transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
            "transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
            "transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
            "transformer/diffusion_pytorch_model-00009-of-00009.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="deepfloyd-if-i-xl",
        task=Task.T2I,
        family="deepfloyd-if",
        source="https://huggingface.co/DeepFloyd/IF-I-XL-v1.0",
        repo_id="DeepFloyd/IF-I-XL-v1.0",
        revision="c03d510e9b75bce9f9db5bb85148c1402ad7e694",
        provenance="author",
        catalog_id="deepfloyd-if",
        format="diffusers",
        license="DeepFloyd IF License (non-commercial)",
        gated=True,
        config={
            "defaults": {
                "steps": 50,
                "guidance": 7.0,
                "superresolution_steps": 50,
                "superresolution_guidance": 4.0,
                "noise_level": 250,
                "width": 256,
                "height": 256,
            }
        },
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "tokenizer/spiece.model",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
        ),
        subfolders={
            "stage_ii": SubRepo(
                repo_id="DeepFloyd/IF-II-L-v1.0",
                revision="609476ce702b2d94aff7d1f944dcc54d4f972901",
                files=(
                    "model_index.json",
                    "image_noising_scheduler/scheduler_config.json",
                    "scheduler/scheduler_config.json",
                    "unet/config.json",
                    "unet/diffusion_pytorch_model.safetensors",
                ),
            )
        },
    )
)

_CHROMA_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)

_entries.extend(
    (
        ModelSpec(
            id="chroma1-base",
            task=Task.T2I,
            family="chroma",
            source="https://huggingface.co/lodestones/Chroma1-Base",
            repo_id="lodestones/Chroma1-Base",
            revision="50ebe125b67794f4ed8e4985d26a4414ca62aed3",
            provenance="author",
            catalog_id="chroma1-base",
            format="diffusers",
            license="Apache-2.0",
            gated=False,
            config={"defaults": {"steps": 40, "guidance": 3.0, "width": 1024, "height": 1024}},
            files=_CHROMA_FILES,
        ),
        ModelSpec(
            id="chroma1-flash",
            task=Task.T2I,
            family="chroma",
            source="https://huggingface.co/lodestones/Chroma1-Flash",
            repo_id="lodestones/Chroma1-Flash",
            revision="093c4f63b60507234aa53a073f1f9f565b3f4336",
            provenance="author",
            catalog_id="chroma1-flash",
            format="diffusers",
            license="Apache-2.0",
            gated=False,
            config={"defaults": {"steps": 4, "guidance": 1.0, "width": 1024, "height": 1024}},
            files=_CHROMA_FILES,
        ),
    )
)

_entries.append(
    ModelSpec(
        id="ldm-text2im-large-256",
        task=Task.T2I,
        family="ldm",
        source="https://huggingface.co/CompVis/ldm-text2im-large-256",
        repo_id="CompVis/ldm-text2im-large-256",
        revision="30de525ca11a880baea4962827fb6cb0bb268955",
        provenance="author",
        catalog_id="ldm",
        format="diffusers",
        license="Apache-2.0",
        config={
            "defaults": {"steps": 50, "guidance": 1.0, "width": 256, "height": 256, "eta": 0.0}
        },
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/vocab.txt",
            "tokenizer/tokenizer_config.json",
            "tokenizer/special_tokens_map.json",
            "bert/config.json",
            "bert/pytorch_model.bin",
            "unet/config.json",
            "unet/diffusion_pytorch_model.bin",
            "vqvae/config.json",
            "vqvae/diffusion_pytorch_model.bin",
        ),
    )
)

# DreamLite publishes bfloat16 safetensors only.  Both releases use the same
# component layout; the mobile checkpoint is distilled for four CFG-free steps.
_DREAMLITE_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model.safetensors",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
for _id, _repo, _revision, _defaults in (
    (
        "dreamlite-base",
        "carlofkl/DreamLite-base",
        "a9a0f151ffd99d3c37f3fd0472f5e8f1b31215aa",
        {"steps": 30, "guidance": 3.5, "image_guidance": 1.5},
    ),
    (
        "dreamlite-mobile",
        "carlofkl/DreamLite-mobile",
        "a48c656c291fe98f74a0405961c9e4e3eee5d6c8",
        {"steps": 4, "guidance": None, "image_guidance": None},
    ),
):
    _entries.append(
        ModelSpec(
            id=_id,
            task=Task.T2I,
            family="dreamlite",
            source=f"https://huggingface.co/{_repo}",
            repo_id=_repo,
            revision=_revision,
            provenance="author",
            catalog_id="dreamlite",
            format="diffusers",
            license="CC-BY-NC-4.0",
            config={"defaults": _defaults, "precision": "bfloat16"},
            files=_DREAMLITE_FILES,
        )
    )

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
        id="dreamshaper-8",
        task=Task.T2I,
        family="stable-diffusion",
        source="https://huggingface.co/Lykon/dreamshaper-8",
        repo_id="Lykon/dreamshaper-8",
        files=_SD1_COMMON + _SD1_SAFETENSORS,
        format="diffusers",
        catalog_id="stable-diffusion-1",
        revision="a7e52b98680b1ba8ff7bce97c7f9f2e2e5337917",
        provenance="author",
        license="CreativeML-OpenRAIL-M",
        config={"defaults": {"steps": 50, "guidance": 7.5}},
    )
)

for _id, _repo, _revision, _precision in (
    (
        "segmind-small-sd",
        "segmind/small-sd",
        "f8b087e68071a29bbe85a4b07e6e1e8998d59aa7",
        "float32",
    ),
    (
        "segmind-tiny-sd",
        "segmind/tiny-sd",
        "cad0bd7495fa6c4bcca01b19a723dc91627fe84f",
        "float16",
    ),
):
    _entries.append(
        ModelSpec(
            id=_id,
            task=Task.T2I,
            family="stable-diffusion",
            source=f"https://huggingface.co/{_repo}",
            repo_id=_repo,
            files=_SD1_COMMON + _SD1_BIN,
            format="diffusers",
            catalog_id="bk-sdm",
            revision=_revision,
            provenance="author",
            license="CreativeML-OpenRAIL-M",
            config={"precision": _precision},
            reference={
                "diffusers": {
                    "version": "0.40.0",
                    "git_commit": "d035dcd7cc7c88e0a154609b62887d50bba9fdc2",
                },
                "segmind-distill-sd": {
                    "repository": "segmind/distill-sd",
                    "git_commit": "c1e97a70d141df09e6fe5cc7dbd66e0cbeae3eeb",
                },
            },
        )
    )

for _variant, _revision in (
    ("base", "0c03b7a0369b49f97d1acf7256d4ee55ced9b2e0"),
    ("small", "572238db7ed3a10858900803f3fc8cca53e893e0"),
    ("tiny", "0364108e53b7f7f4d2585e817a0b7a83dc261cfa"),
):
    _repo = f"nota-ai/bk-sdm-{_variant}"
    _entries.append(
        ModelSpec(
            id=f"bk-sdm-{_variant}",
            task=Task.T2I,
            family="stable-diffusion",
            source=f"https://huggingface.co/{_repo}",
            repo_id=_repo,
            files=_SD1_COMMON + _SD1_SAFETENSORS,
            format="diffusers",
            catalog_id="bk-sdm",
            revision=_revision,
            provenance="author",
            license="CreativeML-OpenRAIL-M",
        )
    )

_entries.append(
    ModelSpec(
        id="sd-v2-1",
        task=Task.T2I,
        family="stable-diffusion",
        source="https://huggingface.co/sd2-community/stable-diffusion-2-1",
        repo_id="sd2-community/stable-diffusion-2-1",
        revision="bb2154823665391b4fb29b0b9cf82a198964ee05",
        provenance="community-mirror",
        catalog_id="stable-diffusion-2",
        format="diffusers",
        license="CreativeML-OpenRAIL++-M",
        config={"defaults": {"steps": 50, "guidance": 7.5}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

for _variant, _revision in (
    ("base", "21915be4c3a83f3fa45f93559750df44943509b1"),
    ("small", "ad82c51a3b568e855a7f79c5b893ea2f9ebb197f"),
    ("tiny", "68277af553777858cd47e133f92e4db47321bc74"),
):
    _repo = f"nota-ai/bk-sdm-v2-{_variant}"
    _entries.append(
        ModelSpec(
            id=f"bk-sdm-v2-{_variant}",
            task=Task.T2I,
            family="stable-diffusion",
            source=f"https://huggingface.co/{_repo}",
            repo_id=_repo,
            revision=_revision,
            provenance="author",
            catalog_id="bk-sdm",
            format="diffusers",
            license="CreativeML-OpenRAIL-M",
            config={"defaults": {"steps": 50, "guidance": 7.5}},
            reference={
                "diffusers": {
                    "version": "0.40.0",
                    "git_commit": "d035dcd7cc7c88e0a154609b62887d50bba9fdc2",
                },
                "transformers": {"version": "5.18.0"},
                "bk-sdm": {
                    "repository": "Nota-NetsPresso/BK-SDM",
                    "git_commit": "e13670c8a7e2b896f508cbece59335930703be6c",
                },
            },
            files=(
                "model_index.json",
                "scheduler/scheduler_config.json",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
            ),
        )
    )

_entries.append(
    ModelSpec(
        id="sd-turbo",
        task=Task.T2I,
        family="stable-diffusion",
        source="https://huggingface.co/stabilityai/sd-turbo",
        repo_id="stabilityai/sd-turbo",
        revision="b261bac6fd2cf515557d5d0707481eafa0485ec2",
        provenance="author",
        catalog_id="sdxl-turbo",
        format="diffusers",
        license="sai-nc-community",
        config={"defaults": {"steps": 1, "guidance": 0.0}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="lcm-dreamshaper-v7",
        task=Task.T2I,
        family="stable-diffusion",
        source="https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7",
        repo_id="SimianLuo/LCM_Dreamshaper_v7",
        revision="a85df6a8bd976cdd08b4fd8f3b73f229c9e54df5",
        provenance="author",
        catalog_id="lcm",
        format="diffusers",
        license="MIT",
        config={"defaults": {"steps": 4, "guidance": 8.0}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="sdxs-512-0.9",
        task=Task.T2I,
        family="stable-diffusion",
        source="https://huggingface.co/IDKiro/sdxs-512-0.9",
        repo_id="IDKiro/sdxs-512-0.9",
        revision="c332f05f60eb4b453de513be52c2a18c48d8cfe6",
        provenance="author",
        catalog_id="sdxs",
        format="diffusers",
        license="CreativeML-OpenRAIL++-M",
        config={"defaults": {"steps": 1, "guidance": 0.0}},
        reference={
            "diffusers": {
                "version": "0.40.0",
                "git_commit": "d035dcd7cc7c88e0a154609b62887d50bba9fdc2",
            },
            "transformers": {"version": "5.18.0"},
        },
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
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

_SDXL_FINETUNE_CONFIGS = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer_2/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/special_tokens_map.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "unet/config.json",
    "vae/config.json",
)
_SDXL_FINETUNE_REFERENCE = {
    "diffusers": {"version": "0.40.0"},
    "transformers": {"version": "5.18.0"},
}
for (
    _finetune_id,
    _finetune_repo,
    _finetune_revision,
    _finetune_license,
    _finetune_precision,
    _finetune_files,
) in (
    (
        "juggernaut-xl-v9",
        "RunDiffusion/Juggernaut-XL-v9",
        "cf419233522daa0b9ea36c3aff98fa2cab1fb0fb",
        "CreativeML-OpenRAIL-M",
        "float16",
        (
            "text_encoder/model.fp16.safetensors",
            "text_encoder_2/model.fp16.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ),
    ),
    (
        "animagine-xl-4.0",
        "cagliostrolab/animagine-xl-4.0",
        "2b7c1b397761bf5bd3cc42e5b39ec99314a75a96",
        "CreativeML-OpenRAIL++-M",
        "float16",
        (
            "text_encoder/model.safetensors",
            "text_encoder_2/model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    ),
    (
        "dreamshaper-xl-1.0",
        "Lykon/dreamshaper-xl-1-0",
        "41e6644752a8c9aa63930e6043c4fd83c7708420",
        "CreativeML-OpenRAIL++-M",
        "float32",
        (
            "text_encoder/model.safetensors",
            "text_encoder_2/model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    ),
):
    _entries.append(
        ModelSpec(
            id=_finetune_id,
            task=Task.T2I,
            family="stable-diffusion-xl",
            source=f"https://huggingface.co/{_finetune_repo}",
            repo_id=_finetune_repo,
            revision=_finetune_revision,
            provenance="author",
            catalog_id="sdxl",
            format="diffusers",
            license=_finetune_license,
            reference=_SDXL_FINETUNE_REFERENCE,
            config={"precision": _finetune_precision},
            files=_SDXL_FINETUNE_CONFIGS + _finetune_files,
        )
    )

_entries.append(
    ModelSpec(
        id="sdxl-refiner-1.0",
        task=Task.T2I,
        family="stable-diffusion-xl",
        source="https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0",
        repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
        revision="5d4cfe854c9a9a87939ff3653551c2b3c99a4356",
        provenance="author",
        catalog_id="sdxl",
        format="diffusers",
        license="CreativeML-OpenRAIL++-M",
        image_input="required",
        config={"pipeline": "refiner"},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors",
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

for _steps in (2, 4, 8):
    _overlay_file = f"sdxl_lightning_{_steps}step_unet.safetensors"
    _entries.append(
        ModelSpec(
            id=f"sdxl-lightning-{_steps}step",
            task=Task.T2I,
            family="stable-diffusion-xl",
            source="https://huggingface.co/ByteDance/SDXL-Lightning",
            repo_id="ByteDance/SDXL-Lightning",
            revision="c9a24f48e1c025556787b0c58dd67a091ece2e44",
            provenance="author",
            catalog_id="sdxl-lightning",
            format="diffusers",
            license="CreativeML-OpenRAIL++-M",
            base="sdxl-base-1.0",
            files=(_overlay_file,),
            overlay={"unet/diffusion_pytorch_model.safetensors": _overlay_file},
            config={
                "defaults": {"steps": _steps, "guidance": 0.0},
                "scheduler": {"timestep_spacing": "trailing"},
            },
        )
    )

_entries.append(
    ModelSpec(
        id="hyper-sdxl-1step",
        task=Task.T2I,
        family="stable-diffusion-xl",
        source="https://huggingface.co/ByteDance/Hyper-SD",
        repo_id="ByteDance/Hyper-SD",
        revision="bc08d970a87c74c71209491d64e3525845698863",
        provenance="author",
        catalog_id="hyper-sd",
        format="diffusers",
        license="Unspecified",
        base="sdxl-base-1.0",
        files=("Hyper-SDXL-1step-Unet.safetensors",),
        overlay={"unet/diffusion_pytorch_model.safetensors": "Hyper-SDXL-1step-Unet.safetensors"},
        config={
            "defaults": {"steps": 1, "guidance": 0.0, "timesteps": [800]},
            "scheduler": {"_class_name": "LCMScheduler"},
        },
    )
)

_entries.append(
    ModelSpec(
        id="dmd2-sdxl-4step",
        task=Task.T2I,
        family="stable-diffusion-xl",
        source="https://huggingface.co/tianweiy/DMD2",
        repo_id="tianweiy/DMD2",
        revision="be22767697a1f3ca656b73c776e15fa335c86c6c",
        provenance="author",
        catalog_id="dmd2",
        format="diffusers",
        license="CC-BY-NC-4.0",
        base="sdxl-base-1.0",
        base_exclude=("unet/diffusion_pytorch_model.safetensors",),
        files=("dmd2_sdxl_4step_unet.bin",),
        overlay={"unet/diffusion_pytorch_model.bin": "dmd2_sdxl_4step_unet.bin"},
        config={
            "defaults": {
                "steps": 4,
                "guidance": 0.0,
                "timesteps": [999, 749, 499, 249],
            },
            "scheduler": {"_class_name": "LCMScheduler"},
        },
    )
)

_entries.append(
    ModelSpec(
        id="dmd2-sdxl-1step",
        task=Task.T2I,
        family="stable-diffusion-xl",
        source="https://huggingface.co/tianweiy/DMD2",
        repo_id="tianweiy/DMD2",
        revision="be22767697a1f3ca656b73c776e15fa335c86c6c",
        provenance="author",
        catalog_id="dmd2",
        format="diffusers",
        license="CC-BY-NC-4.0",
        base="sdxl-base-1.0",
        base_exclude=("unet/diffusion_pytorch_model.safetensors",),
        files=("dmd2_sdxl_1step_unet.bin",),
        overlay={"unet/diffusion_pytorch_model.bin": "dmd2_sdxl_1step_unet.bin"},
        config={
            "defaults": {"steps": 1, "guidance": 0.0, "timesteps": [399]},
            "scheduler": {"_class_name": "LCMScheduler"},
        },
    )
)

_entries.append(
    ModelSpec(
        id="z-image",
        task=Task.T2I,
        family="z-image",
        source="https://huggingface.co/Tongyi-MAI/Z-Image",
        repo_id="Tongyi-MAI/Z-Image",
        revision="04cc4abb7c5069926f75c9bfde9ef43d49423021",
        provenance="author",
        catalog_id="z-image",
        format="diffusers",
        license="Apache-2.0",
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00003.safetensors",
            "text_encoder/model-00002-of-00003.safetensors",
            "text_encoder/model-00003-of-00003.safetensors",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
        config={"defaults": {"steps": 50, "guidance": 4.0}},
    )
)

_entries.append(
    ModelSpec(
        id="z-image-turbo",
        task=Task.T2I,
        family="z-image",
        source="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        revision="f332072aa78be7aecdf3ee76d5c247082da564a6",
        provenance="author",
        catalog_id="z-image",
        format="diffusers",
        license="Apache-2.0",
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00003.safetensors",
            "text_encoder/model-00002-of-00003.safetensors",
            "text_encoder/model-00003-of-00003.safetensors",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="qwen-image-2.1",
        task=Task.T2I,
        family="qwen-image",
        source="https://huggingface.co/Qwen/Qwen-Image-2.1",
        repo_id="Qwen/Qwen-Image-2.1",
        revision="d26bb61231c349cf6b7896fa83353113880e1ba3",
        provenance="author",
        catalog_id="qwen-image-2",
        format="diffusers",
        license="Qwen Research License",
        reference={
            "diffusers": {"git_commit": "7997e4eabb1f880fa5697e2b24de09a892fffad5"},
            "transformers": {"version": "5.18.0"},
        },
        files=(
            "model_index.json",
            "LICENSE",
            "processor/added_tokens.json",
            "processor/chat_template.jinja",
            "processor/merges.txt",
            "processor/preprocessor_config.json",
            "processor/special_tokens_map.json",
            "processor/tokenizer.json",
            "processor/tokenizer_config.json",
            "processor/video_preprocessor_config.json",
            "processor/vocab.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00004.safetensors",
            "text_encoder/model-00002-of-00004.safetensors",
            "text_encoder/model-00003-of-00004.safetensors",
            "text_encoder/model-00004-of-00004.safetensors",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_SD35_COMMON_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "tokenizer_3/special_tokens_map.json",
    "tokenizer_3/spiece.model",
    "tokenizer_3/tokenizer.json",
    "tokenizer_3/tokenizer_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors",
    "text_encoder_3/config.json",
    "text_encoder_3/model.safetensors.index.json",
    "text_encoder_3/model-00001-of-00002.safetensors",
    "text_encoder_3/model-00002-of-00002.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
for _model_id, _repo, _revision, _defaults, _transformer_files in (
    (
        "sd3-medium",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "ea42f8cef0f178587cf766dc8129abd379c90671",
        {"steps": 28, "guidance": 7.0},
        (
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
        ),
    ),
    (
        "sd3.5-medium",
        "stabilityai/stable-diffusion-3.5-medium",
        "b940f670f0eda2d07fbb75229e779da1ad11eb80",
        {"steps": 40, "guidance": 4.5},
        (
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
        ),
    ),
    (
        "sd3.5-large",
        "stabilityai/stable-diffusion-3.5-large",
        "ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f",
        {"steps": 28, "guidance": 3.5},
        (
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        ),
    ),
    (
        "sd3.5-large-turbo",
        "stabilityai/stable-diffusion-3.5-large-turbo",
        "ec07796fc06b096cc56de9762974a28f4c632eda",
        {"steps": 4, "guidance": 0.0},
        (
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        ),
    ),
):
    _entries.append(
        ModelSpec(
            id=_model_id,
            task=Task.T2I,
            family="stable-diffusion-3",
            source=f"https://huggingface.co/{_repo}",
            repo_id=_repo,
            revision=_revision,
            provenance="author",
            catalog_id="stable-diffusion-3",
            format="diffusers",
            license="Stability AI Community License",
            gated=True,
            config={"defaults": _defaults},
            files=_SD35_COMMON_FILES + _transformer_files,
        )
    )

_entries.append(
    ModelSpec(
        id="janus-pro-1b",
        task=Task.T2I,
        family="janus",
        source="https://huggingface.co/deepseek-community/Janus-Pro-1B",
        repo_id="deepseek-community/Janus-Pro-1B",
        revision="1655280bb75959cc1cb85529a2a8b26e7016072e",
        provenance="community-conversion",
        catalog_id="janus-pro",
        format="transformers",
        license="DeepSeek Model License",
        config={"defaults": {"guidance": 5.0}},
        files=(
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="janus-pro-7b",
        task=Task.T2I,
        family="janus",
        source="https://huggingface.co/deepseek-community/Janus-Pro-7B",
        repo_id="deepseek-community/Janus-Pro-7B",
        revision="8f23139ed8d37346501091b3690291cb7b16f9fc",
        provenance="community-conversion",
        catalog_id="janus-pro",
        format="transformers",
        license="DeepSeek Model License",
        config={"defaults": {"guidance": 5.0}},
        files=(
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="flux2-klein-4b",
        task=Task.T2I,
        family="flux2",
        source="https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
        repo_id="black-forest-labs/FLUX.2-klein-4B",
        revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
        provenance="author",
        catalog_id="flux-2",
        format="diffusers",
        license="Apache-2.0",
        config={"defaults": {"steps": 4, "guidance": 1.0}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/added_tokens.json",
            "tokenizer/chat_template.jinja",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="flux2-klein-9b",
        task=Task.T2I,
        family="flux2",
        source="https://huggingface.co/black-forest-labs/FLUX.2-klein-9B",
        repo_id="black-forest-labs/FLUX.2-klein-9B",
        revision="92196c8e11f7b6cf2b7493e037d8c5345c559216",
        provenance="author",
        catalog_id="flux-2",
        format="diffusers",
        license="FLUX Non-Commercial License",
        gated=True,
        config={"defaults": {"steps": 4, "guidance": 1.0}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/added_tokens.json",
            "tokenizer/chat_template.jinja",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00004.safetensors",
            "text_encoder/model-00002-of-00004.safetensors",
            "text_encoder/model-00003-of-00004.safetensors",
            "text_encoder/model-00004-of-00004.safetensors",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="flux1-schnell",
        task=Task.T2I,
        family="flux",
        source="https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        repo_id="black-forest-labs/FLUX.1-schnell",
        revision="741f7c3ce8b383c54771c7003378a50191e9efe9",
        provenance="author",
        catalog_id="flux-1",
        format="diffusers",
        license="Apache-2.0",
        gated=True,
        config={"defaults": {"steps": 4, "guidance": 0.0, "max_sequence_length": 256}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors.index.json",
            "text_encoder_2/model-00001-of-00002.safetensors",
            "text_encoder_2/model-00002-of-00002.safetensors",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer_2/special_tokens_map.json",
            "tokenizer_2/spiece.model",
            "tokenizer_2/tokenizer.json",
            "tokenizer_2/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="flux1-dev",
        task=Task.T2I,
        family="flux",
        source="https://huggingface.co/black-forest-labs/FLUX.1-dev",
        repo_id="black-forest-labs/FLUX.1-dev",
        revision="3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        provenance="author",
        catalog_id="flux-1",
        format="diffusers",
        license="FLUX.1-dev Non-Commercial License",
        gated=True,
        config={"defaults": {"steps": 50, "guidance": 3.5, "max_sequence_length": 512}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors.index.json",
            "text_encoder_2/model-00001-of-00002.safetensors",
            "text_encoder_2/model-00002-of-00002.safetensors",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer_2/special_tokens_map.json",
            "tokenizer_2/spiece.model",
            "tokenizer_2/tokenizer.json",
            "tokenizer_2/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="flux1-krea-dev",
        task=Task.T2I,
        family="flux",
        source="https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev",
        repo_id="black-forest-labs/FLUX.1-Krea-dev",
        revision="8162a9c7b05a641be098422bf2fcf335615c2f28",
        provenance="author",
        catalog_id="flux-1",
        format="diffusers",
        license="FLUX.1-dev Non-Commercial License",
        gated=True,
        config={"defaults": {"steps": 28, "guidance": 4.5, "max_sequence_length": 512}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors.index.json",
            "text_encoder_2/model-00001-of-00002.safetensors",
            "text_encoder_2/model-00002-of-00002.safetensors",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer_2/special_tokens_map.json",
            "tokenizer_2/spiece.model",
            "tokenizer_2/tokenizer.json",
            "tokenizer_2/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="flux-1-lite-8b",
        task=Task.T2I,
        family="flux",
        source="https://huggingface.co/Freepik/flux.1-lite-8B",
        repo_id="Freepik/flux.1-lite-8B",
        revision="16b9373dd3d05bc5f125558f5e2b8470fbb873e8",
        provenance="author",
        catalog_id="flux-1",
        format="diffusers",
        license="FLUX.1-dev Non-Commercial License",
        config={"defaults": {"steps": 28, "guidance": 3.5, "max_sequence_length": 512}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors.index.json",
            "text_encoder_2/model-00001-of-00002.safetensors",
            "text_encoder_2/model-00002-of-00002.safetensors",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer_2/special_tokens_map.json",
            "tokenizer_2/spiece.model",
            "tokenizer_2/tokenizer.json",
            "tokenizer_2/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="flux1-kontext-dev",
        task=Task.T2I,
        family="flux",
        source="https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev",
        repo_id="black-forest-labs/FLUX.1-Kontext-dev",
        revision="24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d",
        provenance="author",
        catalog_id="flux-1-kontext",
        format="diffusers",
        license="FLUX.1-dev Non-Commercial License",
        gated=True,
        image_input="optional",
        config={
            "pipeline": "kontext",
            "defaults": {"steps": 28, "guidance": 3.5, "max_sequence_length": 512},
        },
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors.index.json",
            "text_encoder_2/model-00001-of-00002.safetensors",
            "text_encoder_2/model-00002-of-00002.safetensors",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer_2/special_tokens_map.json",
            "tokenizer_2/spiece.model",
            "tokenizer_2/tokenizer.json",
            "tokenizer_2/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="chroma1-hd",
        task=Task.T2I,
        family="chroma",
        source="https://huggingface.co/lodestones/Chroma1-HD",
        repo_id="lodestones/Chroma1-HD",
        revision="0e0c60ece1e82b17cb7f77342d765ba5024c40c0",
        provenance="author",
        catalog_id="chroma1-hd",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        config={"defaults": {"steps": 35, "guidance": 5.0, "width": 1024, "height": 1024}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/added_tokens.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer/spiece.model",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_HUNYUAN_DIT_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors.index.json",
    "text_encoder_2/model-00001-of-00002.safetensors",
    "text_encoder_2/model-00002-of-00002.safetensors",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/spiece.model",
    "tokenizer_2/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)

_entries.append(
    ModelSpec(
        id="hunyuan-dit-v1.1-distilled",
        task=Task.T2I,
        family="hunyuan-dit",
        source="https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
        repo_id="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
        revision="527cf2ecce7c04021975938f8b0e44e35d2b1ed9",
        provenance="author",
        catalog_id="hunyuan-dit",
        format="diffusers",
        license="Tencent Hunyuan Community License",
        gated=False,
        config={"defaults": {"steps": 25, "guidance": 5.0}},
        files=_HUNYUAN_DIT_FILES,
    )
)

_entries.append(
    ModelSpec(
        id="hunyuan-dit-v1.2",
        task=Task.T2I,
        family="hunyuan-dit",
        source="https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
        repo_id="Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
        revision="5e96094e0ad19e7f475de8711f03634ca0ccc40c",
        provenance="author",
        catalog_id="hunyuan-dit",
        format="diffusers",
        license="Tencent Hunyuan Community License",
        gated=False,
        config={"defaults": {"steps": 50, "guidance": 5.0}},
        files=_HUNYUAN_DIT_FILES,
    )
)

_entries.append(
    ModelSpec(
        id="hunyuan-dit-v1.2-distilled",
        task=Task.T2I,
        family="hunyuan-dit",
        source="https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
        repo_id="Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
        revision="ba991d1546d8c50936c4c16398ed0a87b9b99fb1",
        provenance="author",
        catalog_id="hunyuan-dit",
        format="diffusers",
        license="Tencent Hunyuan Community License",
        gated=False,
        config={"defaults": {"steps": 25, "guidance": 5.0}},
        files=_HUNYUAN_DIT_FILES,
    )
)

_PIXART_COMMON = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
_PIXART_ALPHA_512_FILES = (
    _PIXART_COMMON[:4]
    + tuple(f"text_encoder/model-0000{index}-of-00004.safetensors" for index in range(1, 5))
    + ("tokenizer/added_tokens.json",)
    + _PIXART_COMMON[6:]
)
for _id, _repo, _revision, _catalog_id, _max_length, _size, _files in (
    (
        "pixart-sigma-xl-1024",
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "e102b3591cc82e97071b8b4cb90d834d0c487207",
        "pixart-sigma",
        300,
        1024,
        _PIXART_COMMON,
    ),
    (
        "pixart-alpha-xl-1024",
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        "b89adadeccd9ead2adcb9fa2825d3fabec48d404",
        "pixart-alpha",
        120,
        1024,
        _PIXART_COMMON[:6] + ("tokenizer/added_tokens.json",) + _PIXART_COMMON[6:],
    ),
    (
        "pixart-alpha-xl-512",
        "PixArt-alpha/PixArt-XL-2-512x512",
        "50f702106901db6d0f8b67eb88e814c56ded2692",
        "pixart-alpha",
        120,
        512,
        _PIXART_ALPHA_512_FILES,
    ),
):
    _entries.append(
        ModelSpec(
            id=_id,
            task=Task.T2I,
            family="pixart",
            source=f"https://huggingface.co/{_repo}",
            repo_id=_repo,
            revision=_revision,
            provenance="author",
            catalog_id=_catalog_id,
            format="diffusers",
            license="CreativeML-OpenRAIL++-M",
            gated=False,
            config={
                "defaults": {
                    "steps": 20,
                    "guidance": 4.5,
                    "max_sequence_length": _max_length,
                    **({"width": _size, "height": _size} if _size != 1024 else {}),
                }
            },
            files=_files,
        )
    )

_entries.append(
    ModelSpec(
        id="pixart-lcm-xl-1024",
        task=Task.T2I,
        family="pixart",
        source="https://huggingface.co/PixArt-alpha/PixArt-LCM-XL-2-1024-MS",
        repo_id="PixArt-alpha/PixArt-LCM-XL-2-1024-MS",
        revision="e153060b10990994d2ed31304b3224da47fbd740",
        provenance="author",
        catalog_id="pixart-delta",
        format="diffusers",
        license="CreativeML-OpenRAIL++-M",
        gated=False,
        config={"defaults": {"steps": 4, "guidance": 0.0, "max_sequence_length": 120}},
        files=_PIXART_ALPHA_512_FILES,
    )
)

# The PixArt-Sigma 256, 512 and 2K repositories publish only a transformer; the text encoder,
# tokenizer, VAE and scheduler come from the 1024 repository, as in the model cards.
for _id, _repo, _revision, _size in (
    (
        "pixart-sigma-xl-256",
        "PixArt-alpha/PixArt-Sigma-XL-2-256x256",
        "495cb1bafac48d71ca96761409c5fd7e48354df2",
        256,
    ),
    (
        "pixart-sigma-xl-512",
        "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        "76fb7eb5a9314bc1e4e479d2f13447517fca9be4",
        512,
    ),
    (
        "pixart-sigma-xl-2k",
        "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS",
        "cd274f4c8b01de0dcc87b65887dd357d473bf516",
        2048,
    ),
):
    _entries.append(
        ModelSpec(
            id=_id,
            task=Task.T2I,
            family="pixart",
            source=f"https://huggingface.co/{_repo}",
            repo_id=_repo,
            revision=_revision,
            provenance="author",
            catalog_id="pixart-sigma",
            format="diffusers",
            license="CreativeML-OpenRAIL++-M",
            gated=False,
            base="pixart-sigma-xl-1024",
            config={
                "defaults": {
                    "steps": 20,
                    "guidance": 4.5,
                    "max_sequence_length": 300,
                    "width": _size,
                    "height": _size,
                }
            },
            files=("transformer/config.json", "transformer/diffusion_pytorch_model.safetensors"),
            overlay={
                "transformer/config.json": "transformer/config.json",
                "transformer/diffusion_pytorch_model.safetensors": (
                    "transformer/diffusion_pytorch_model.safetensors"
                ),
            },
        )
    )

_SANA_COMMON = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
)
_SANA_VARIANTS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "sana-1600m-1024",
        "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        "ac0da2ff55fbe434795be0dce883042e4d49e2fc",
        (
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        ),
    ),
    (
        "sana-600m-1024",
        "Efficient-Large-Model/Sana_600M_1024px_diffusers",
        "28f3af7689de15f3883d5863059a2fca0aa9b829",
        ("transformer/diffusion_pytorch_model.safetensors",),
    ),
)
for _sana_id, _sana_repo, _sana_revision, _sana_weights in _SANA_VARIANTS:
    _entries.append(
        ModelSpec(
            id=_sana_id,
            task=Task.T2I,
            family="sana",
            source=f"https://huggingface.co/{_sana_repo}",
            repo_id=_sana_repo,
            revision=_sana_revision,
            provenance="author",
            catalog_id="sana",
            format="diffusers",
            license="Apache-2.0",
            gated=False,
            config={"defaults": {"steps": 20, "guidance": 4.5, "width": 1024, "height": 1024}},
            files=(
                *_SANA_COMMON,
                *_sana_weights,
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
            ),
        )
    )

# Sana-Sprint uses the SCM scheduler and guidance embeddings; defaults to the 2-step
# inference schedule used by the upstream pipeline.
_entries.append(
    ModelSpec(
        id="sana-sprint-1.6b-1024",
        task=Task.T2I,
        family="sana",
        source="https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        repo_id="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        revision="19683c58b7ea290e55cedd8950ae1d86ada7ef96",
        provenance="author",
        catalog_id="sana",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        config={"defaults": {"steps": 2, "guidance": 4.5, "width": 1024, "height": 1024}},
        files=(
            *_SANA_COMMON,
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

# Sana 1.5 4.8B keeps the standard DPMSolverMultistep scheduler but adds qk_norm and
# uses a 60-layer transformer split across two safetensors shards.
_entries.append(
    ModelSpec(
        id="sana-1.5-4.8b-1024",
        task=Task.T2I,
        family="sana",
        source="https://huggingface.co/Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        repo_id="Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
        revision="9468102c3cebb657f8c4b5f1e5a71e989a15f10d",
        provenance="author",
        catalog_id="sana",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        config={"defaults": {"steps": 20, "guidance": 4.5, "width": 1024, "height": 1024}},
        files=(
            *_SANA_COMMON,
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

# Kolors publishes float16 weights only; the loaders read the variant file names.
_entries.append(
    ModelSpec(
        id="kolors",
        task=Task.T2I,
        family="kolors",
        source="https://huggingface.co/Kwai-Kolors/Kolors-diffusers",
        repo_id="Kwai-Kolors/Kolors-diffusers",
        revision="7e091c75199e910a26cd1b51ed52c28de5db3711",
        provenance="author",
        catalog_id="kolors",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        config={"defaults": {"steps": 50, "guidance": 5.0, "width": 1024, "height": 1024}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.fp16.json",
            "text_encoder/model.fp16-00001-of-00003.safetensors",
            "text_encoder/model.fp16-00002-of-00003.safetensors",
            "text_encoder/model.fp16-00003-of-00003.safetensors",
            "tokenizer/tokenizer.model",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.txt",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="lumina-image-2.0",
        task=Task.T2I,
        family="lumina2",
        source="https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0",
        repo_id="Alpha-VLLM/Lumina-Image-2.0",
        revision="53504abd8178b30685b6c4c7a4cd181ff78b73e9",
        provenance="author",
        catalog_id="lumina-image-2",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        config={"defaults": {"steps": 30, "guidance": 4.0, "width": 1024, "height": 1024}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00003.safetensors",
            "text_encoder/model-00002-of-00003.safetensors",
            "text_encoder/model-00003-of-00003.safetensors",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer.model",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="lumina-next",
        task=Task.T2I,
        family="lumina-next",
        source="https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers",
        repo_id="Alpha-VLLM/Lumina-Next-SFT-diffusers",
        revision="0ee5ec90043acf5cb41fe96274af36eb7fad8d95",
        provenance="author",
        catalog_id="lumina-next",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        config={"defaults": {"steps": 30, "guidance": 4.0, "width": 1024, "height": 1024}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer.model",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="lumina-dimoo",
        task=Task.T2I,
        family="lumina-dimoo",
        source="https://huggingface.co/Alpha-VLLM/Lumina-DiMOO",
        repo_id="Alpha-VLLM/Lumina-DiMOO",
        revision="f014549e17679e41b119959754cdd9320af9bdc8",
        provenance="author",
        catalog_id="lumina-dimoo",
        format="safetensors",
        license="Apache-2.0",
        gated=False,
        reference={
            "lumina-dimoo": {
                "repository": "Alpha-VLLM/Lumina-DiMOO",
                "git_commit": "81344e60545ae8d70f11039d3450717d974f831b",
                "class": "LLaDAForMultiModalGeneration/generate_image",
            }
        },
        config={"defaults": {"steps": 64, "guidance": 4.0, "width": 1024, "height": 1024}},
        files=(
            "model_index.json",
            "args.json",
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vqvae/config.json",
            "vqvae/diffusion_pytorch_model.safetensors",
        ),
    )
)

for _amused_id, _amused_revision, _amused_size in (
    ("amused-256", "09b6259bf96dbe6d70a852b70812420fe02df55e", 256),
    ("amused-512", "6f3f309ba188a08e347bff9691a0ac791817ca97", 512),
):
    _entries.append(
        ModelSpec(
            id=_amused_id,
            task=Task.T2I,
            family="amused",
            source=f"https://huggingface.co/amused/{_amused_id}",
            repo_id=f"amused/{_amused_id}",
            revision=_amused_revision,
            provenance="author",
            catalog_id="amused",
            format="diffusers",
            license="OpenRAIL++",
            gated=False,
            config={
                "defaults": {
                    "steps": 12,
                    "guidance": 10.0,
                    "width": _amused_size,
                    "height": _amused_size,
                }
            },
            files=(
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "transformer/config.json",
                "transformer/diffusion_pytorch_model.safetensors",
                "vqvae/config.json",
                "vqvae/diffusion_pytorch_model.safetensors",
            ),
        )
    )

_entries.append(
    ModelSpec(
        id="meissonic",
        task=Task.T2I,
        family="meissonic",
        source="https://huggingface.co/MeissonFlow/Meissonic",
        repo_id="MeissonFlow/Meissonic",
        revision="7ee5068bcebaaf6165e5d77c2969aa3b38e17b38",
        provenance="author",
        catalog_id="meissonic",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        reference={
            "meissonic": {
                "repository": "viiika/Meissonic",
                "git_commit": "a9de06ca8c02c467aec3da275605fc59380299e2",
            },
            "transformers": {"version": "5.18.0"},
        },
        config={"defaults": {"steps": 48, "guidance": 9.0, "width": 1024, "height": 1024}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vqvae/config.json",
            "vqvae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="auraflow-v0.3",
        task=Task.T2I,
        family="auraflow",
        source="https://huggingface.co/fal/AuraFlow-v0.3",
        repo_id="fal/AuraFlow-v0.3",
        revision="2cd8588f04c886002be4571697d84654a50e3af3",
        provenance="author",
        catalog_id="auraflow",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        config={"defaults": {"steps": 50, "guidance": 3.5, "width": 1024, "height": 1024}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "tokenizer/added_tokens.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer.model",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="krea2-turbo",
        task=Task.T2I,
        family="krea2",
        source="https://huggingface.co/krea/Krea-2-Turbo",
        repo_id="krea/Krea-2-Turbo",
        revision="98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7",
        provenance="author",
        catalog_id="krea-2",
        format="diffusers",
        license="Krea 2 Community License",
        gated=True,
        config={"defaults": {"steps": 8, "guidance": 0.0}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "tokenizer/chat_template.jinja",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="longcat-image",
        task=Task.T2I,
        family="longcat-image",
        source="https://huggingface.co/meituan-longcat/LongCat-Image",
        repo_id="meituan-longcat/LongCat-Image",
        revision="d2ea50b79a930074c37b9b97ce45e3b2ea8cf4d8",
        provenance="author",
        catalog_id="longcat-image",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        config={
            "defaults": {
                "steps": 50,
                "guidance": 4.5,
                "enable_cfg_renorm": True,
                "cfg_renorm_min": 0.0,
                "enable_prompt_rewrite": True,
            }
        },
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/model-00001-of-00005.safetensors",
            "text_encoder/model-00002-of-00005.safetensors",
            "text_encoder/model-00003-of-00005.safetensors",
            "text_encoder/model-00004-of-00005.safetensors",
            "text_encoder/model-00005-of-00005.safetensors",
            "text_encoder/model.safetensors.index.json",
            "text_processor/chat_template.json",
            "text_processor/config.json",
            "text_processor/merges.txt",
            "text_processor/preprocessor_config.json",
            "text_processor/special_tokens_map.json",
            "text_processor/tokenizer.json",
            "text_processor/tokenizer_config.json",
            "text_processor/vocab.json",
            "tokenizer/chat_template.json",
            "tokenizer/config.json",
            "tokenizer/merges.txt",
            "tokenizer/preprocessor_config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="cogview4-6b",
        task=Task.T2I,
        family="cogview4",
        source="https://huggingface.co/zai-org/CogView4-6B",
        repo_id="zai-org/CogView4-6B",
        revision="63a52b7f6dace7033380cd6da14d0915eab3e6b5",
        provenance="author",
        catalog_id="cogview4",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        config={"defaults": {"steps": 50, "guidance": 5.0}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00004.safetensors",
            "text_encoder/model-00002-of-00004.safetensors",
            "text_encoder/model-00003-of-00004.safetensors",
            "text_encoder/model-00004-of-00004.safetensors",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="cogview3-plus-3b",
        task=Task.T2I,
        family="cogview3",
        source="https://huggingface.co/zai-org/CogView3-Plus-3B",
        repo_id="zai-org/CogView3-Plus-3B",
        revision="5d70e40732ac0efac98524c51a7fa9c82707f1e5",
        provenance="author",
        catalog_id="cogview3-plus",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        config={"defaults": {"steps": 50, "guidance": 5.0}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00004.safetensors",
            "text_encoder/model-00002-of-00004.safetensors",
            "text_encoder/model-00003-of-00004.safetensors",
            "text_encoder/model-00004-of-00004.safetensors",
            "tokenizer/added_tokens.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer/spiece.model",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="ideogram4-fp8",
        task=Task.T2I,
        family="ideogram4",
        source="https://huggingface.co/ideogram-ai/ideogram-4-fp8",
        repo_id="ideogram-ai/ideogram-4-fp8",
        revision="ee79a7237b519f1402ceacf952f30c8a31ec5073",
        provenance="author",
        catalog_id="ideogram-4",
        format="diffusers",
        license="Ideogram 4 Non-Commercial License",
        gated=True,
        config={"defaults": {"preset": "V4_QUALITY_48"}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/chat_template.jinja",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "unconditional_transformer/config.json",
            "unconditional_transformer/diffusion_pytorch_model.safetensors",
            "unconditional_transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="hidream-o1-image",
        task=Task.T2I,
        family="hidream-o1",
        source="https://huggingface.co/HiDream-ai/HiDream-O1-Image",
        repo_id="HiDream-ai/HiDream-O1-Image",
        revision="0b0901d99f200389e138c61946af1185f5f49a13",
        provenance="author",
        catalog_id="hidream-o1-image",
        format="transformers",
        license="MIT",
        reference={
            "hidream-o1": {
                "repository": "HiDream-ai/HiDream-O1-Image",
                "git_commit": "2c2d29ff729e48f33e41f49edfdbd81d5ac103b4",
            }
        },
        config={"defaults": {"steps": 50, "guidance": 5.0, "shift": 3.0}},
        files=(
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00008.safetensors",
            "model-00002-of-00008.safetensors",
            "model-00003-of-00008.safetensors",
            "model-00004-of-00008.safetensors",
            "model-00005-of-00008.safetensors",
            "model-00006-of-00008.safetensors",
            "model-00007-of-00008.safetensors",
            "model-00008-of-00008.safetensors",
        ),
        image_input=None,
    )
)

_entries.append(
    ModelSpec(
        id="sdxl-turbo",
        task=Task.T2I,
        family="stable-diffusion-xl",
        source="https://huggingface.co/stabilityai/sdxl-turbo",
        repo_id="stabilityai/sdxl-turbo",
        revision="71153311d3dbb46851df1931d3ca6e939de83304",
        provenance="author",
        catalog_id="sdxl-turbo",
        format="diffusers",
        license="sai-nc-community",
        config={"defaults": {"steps": 1, "guidance": 0.0}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer_2/vocab.json",
            "tokenizer_2/merges.txt",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/special_tokens_map.json",
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "unet/config.json",
            "vae/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="ssd-1b",
        task=Task.T2I,
        family="stable-diffusion-xl",
        source="https://huggingface.co/segmind/SSD-1B",
        repo_id="segmind/SSD-1B",
        revision="60987f37e94cd59c36b1cba832b9f97b57395a10",
        provenance="author",
        catalog_id="ssd-1b",
        format="diffusers",
        license="Apache-2.0",
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer_2/vocab.json",
            "tokenizer_2/merges.txt",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/special_tokens_map.json",
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "unet/config.json",
            "vae/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="segmind-vega",
        task=Task.T2I,
        family="stable-diffusion-xl",
        source="https://huggingface.co/segmind/Segmind-Vega",
        repo_id="segmind/Segmind-Vega",
        revision="7714c4363e5856ff974a4f4b068e8691f26d0b40",
        provenance="author",
        catalog_id="segmind-vega",
        format="diffusers",
        license="Apache-2.0",
        reference={
            "diffusers": {
                "version": "0.40.0",
                "git_commit": "d035dcd7cc7c88e0a154609b62887d50bba9fdc2",
            },
            "transformers": {"version": "5.18.0"},
            "segmind-ssd-1b": {
                "repository": "segmind/SSD-1B",
                "git_commit": "2bfc35f41f21531c61e3d3470bca907fd839d035",
            },
        },
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer_2/vocab.json",
            "tokenizer_2/merges.txt",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/special_tokens_map.json",
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "unet/config.json",
            "vae/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

for _id, _repo, _revision in (
    (
        "koala-700m",
        "etri-vilab/koala-700m",
        "5e73a318a1ee1b97cc31fd50f09ac13e018c0b63",
    ),
    (
        "koala-1b",
        "etri-vilab/koala-1b",
        "8ef26f38fe4d141c35fda24966bd48777fc3c8ad",
    ),
):
    _entries.append(
        ModelSpec(
            id=_id,
            task=Task.T2I,
            family="stable-diffusion-xl",
            source=f"https://huggingface.co/{_repo}",
            repo_id=_repo,
            revision=_revision,
            provenance="author",
            catalog_id="koala",
            format="diffusers",
            license="Unspecified",
            reference={
                "diffusers": {
                    "version": "0.40.0",
                    "git_commit": "d035dcd7cc7c88e0a154609b62887d50bba9fdc2",
                },
                "transformers": {"version": "5.18.0"},
                "sdxl-koala": {
                    "repository": "youngwanLEE/sdxl-koala",
                    "git_commit": "df9dac7514cc2bd6882a5161401cd9740349c11b",
                },
            },
            files=(
                "model_index.json",
                "scheduler/scheduler_config.json",
                "tokenizer/vocab.json",
                "tokenizer/merges.txt",
                "tokenizer/tokenizer_config.json",
                "tokenizer/special_tokens_map.json",
                "tokenizer_2/vocab.json",
                "tokenizer_2/merges.txt",
                "tokenizer_2/tokenizer_config.json",
                "tokenizer_2/special_tokens_map.json",
                "text_encoder/config.json",
                "text_encoder_2/config.json",
                "unet/config.json",
                "vae/config.json",
                "text_encoder/model.safetensors",
                "text_encoder_2/model.safetensors",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
            ),
        )
    )

for _id, _repo, _revision in (
    (
        "koala-lightning-700m",
        "etri-vilab/koala-lightning-700m",
        "358e0e3576adf2ebc563b64cfcf8a088a4a6407a",
    ),
    (
        "koala-lightning-1b",
        "etri-vilab/koala-lightning-1b",
        "b4114f7577bb08e08a6b9792eb3a3a8b34c95adb",
    ),
):
    _entries.append(
        ModelSpec(
            id=_id,
            task=Task.T2I,
            family="stable-diffusion-xl",
            source=f"https://huggingface.co/{_repo}",
            repo_id=_repo,
            revision=_revision,
            provenance="author",
            catalog_id="koala",
            format="diffusers",
            license="Unspecified",
            config={
                "defaults": {"steps": 10, "guidance": 3.5},
                "scheduler": {"timestep_spacing": "trailing"},
            },
            reference={
                "diffusers": {
                    "version": "0.40.0",
                    "git_commit": "d035dcd7cc7c88e0a154609b62887d50bba9fdc2",
                },
                "transformers": {"version": "5.18.0"},
                "sdxl-koala": {
                    "repository": "youngwanLEE/sdxl-koala",
                    "git_commit": "df9dac7514cc2bd6882a5161401cd9740349c11b",
                },
            },
            files=(
                "model_index.json",
                "scheduler/scheduler_config.json",
                "tokenizer/vocab.json",
                "tokenizer/merges.txt",
                "tokenizer/tokenizer_config.json",
                "tokenizer/special_tokens_map.json",
                "tokenizer_2/vocab.json",
                "tokenizer_2/merges.txt",
                "tokenizer_2/tokenizer_config.json",
                "tokenizer_2/special_tokens_map.json",
                "text_encoder/config.json",
                "text_encoder_2/config.json",
                "unet/config.json",
                "vae/config.json",
                "text_encoder/model.safetensors",
                "text_encoder_2/model.safetensors",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
            ),
        )
    )

_entries.append(
    ModelSpec(
        id="playground-v2",
        task=Task.T2I,
        family="stable-diffusion-xl",
        source="https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic",
        repo_id="playgroundai/playground-v2-1024px-aesthetic",
        revision="737a24ebe3ae4e5421110879168be188853d86c8",
        provenance="author",
        catalog_id="playground-v2",
        format="diffusers",
        license="playground-v2-community",
        config={"defaults": {"guidance": 3.0}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer_2/vocab.json",
            "tokenizer_2/merges.txt",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/special_tokens_map.json",
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "unet/config.json",
            "vae/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="playground-v2.5",
        task=Task.T2I,
        family="stable-diffusion-xl",
        source="https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic",
        repo_id="playgroundai/playground-v2.5-1024px-aesthetic",
        revision="1e032f13f2fe6db2dc49947dbdbd196e753de573",
        provenance="author",
        catalog_id="playground-v2",
        format="diffusers",
        license="playground-v2dot5-community",
        config={"defaults": {"guidance": 3.0}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer_2/vocab.json",
            "tokenizer_2/merges.txt",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/special_tokens_map.json",
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "unet/config.json",
            "vae/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="kandinsky-2.2",
        task=Task.T2I,
        family="kandinsky22",
        source="https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder",
        repo_id="kandinsky-community/kandinsky-2-2-decoder",
        revision="9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3",
        provenance="author",
        catalog_id="kandinsky-2",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        config={
            "defaults": {
                "steps": 100,
                "guidance": 4.0,
                "prior_steps": 25,
                "prior_guidance": 4.0,
            }
        },
        files=(
            "model_index.json",
            "movq/config.json",
            "movq/diffusion_pytorch_model.safetensors",
            "scheduler/scheduler_config.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
        ),
        subfolders={
            "prior": SubRepo(
                repo_id="kandinsky-community/kandinsky-2-2-prior",
                revision="9fc51ad5732afc5d031724219d22e6c42179c5a8",
                files=(
                    "model_index.json",
                    "image_encoder/config.json",
                    "image_encoder/model.safetensors",
                    "image_processor/preprocessor_config.json",
                    "prior/config.json",
                    "prior/diffusion_pytorch_model.safetensors",
                    "scheduler/scheduler_config.json",
                    "text_encoder/config.json",
                    "text_encoder/model.safetensors",
                    "tokenizer/merges.txt",
                    "tokenizer/special_tokens_map.json",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer/vocab.json",
                ),
            )
        },
    )
)

_entries.append(
    ModelSpec(
        id="kandinsky-2-1",
        task=Task.T2I,
        family="kandinsky21",
        source="https://huggingface.co/kandinsky-community/kandinsky-2-1",
        repo_id="kandinsky-community/kandinsky-2-1",
        revision="04b366659663bf60deacb96fd3e38f1cb39a3be4",
        provenance="author",
        catalog_id="kandinsky-2",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        config={
            "defaults": {
                "steps": 100,
                "guidance": 4.0,
                "prior_steps": 25,
                "prior_guidance": 4.0,
            }
        },
        files=(
            "ddpm_scheduler/scheduler_config.json",
            "model_index.json",
            "movq/config.json",
            "movq/diffusion_pytorch_model.safetensors",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "tokenizer/sentencepiece.bpe.model",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
        ),
        subfolders={
            "prior": SubRepo(
                repo_id="kandinsky-community/kandinsky-2-1-prior",
                revision="b726af62d69111bbf62b5d71bb9d90d9867fbce4",
                files=(
                    "model_index.json",
                    "image_encoder/config.json",
                    "image_encoder/model.safetensors",
                    "image_processor/preprocessor_config.json",
                    "prior/config.json",
                    "prior/diffusion_pytorch_model.safetensors",
                    "scheduler/scheduler_config.json",
                    "text_encoder/config.json",
                    "text_encoder/model.safetensors",
                    "tokenizer/merges.txt",
                    "tokenizer/special_tokens_map.json",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer/vocab.json",
                ),
            )
        },
    )
)

# Kandinsky 3 publishes float16 weights only; the loaders read the variant file names.
_entries.append(
    ModelSpec(
        id="kandinsky-3",
        task=Task.T2I,
        family="kandinsky3",
        source="https://huggingface.co/kandinsky-community/kandinsky-3",
        repo_id="kandinsky-community/kandinsky-3",
        revision="bf79e6c219da8a94abb50235fdc4567eb8fb4632",
        provenance="author",
        catalog_id="kandinsky-3",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        config={"defaults": {"steps": 25, "guidance": 3.0, "width": 1024, "height": 1024}},
        files=(
            "model_index.json",
            "movq/config.json",
            "movq/diffusion_pytorch_model.fp16.safetensors",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.fp16.json",
            "text_encoder/model.fp16-00001-of-00005.safetensors",
            "text_encoder/model.fp16-00002-of-00005.safetensors",
            "text_encoder/model.fp16-00003-of-00005.safetensors",
            "text_encoder/model.fp16-00004-of-00005.safetensors",
            "text_encoder/model.fp16-00005-of-00005.safetensors",
            "tokenizer/added_tokens.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer/spiece.model",
            "tokenizer/tokenizer_config.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ),
    )
)

# Kandinsky 5 publishes bfloat16 transformer and text-encoder weights; the catalog lists them
# as released and the native loaders upcast to float32 for parity testing.
_entries.append(
    ModelSpec(
        id="kandinsky-5-t2i-lite",
        task=Task.T2I,
        family="kandinsky5",
        source="https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers",
        repo_id="kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers",
        revision="25da1e82e6adefdbc83c02a71f762287c5e471a4",
        provenance="author",
        catalog_id="kandinsky-5",
        format="diffusers",
        license="MIT",
        gated=False,
        config={"defaults": {"steps": 50, "guidance": 3.5, "width": 1024, "height": 1024}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00005.safetensors",
            "text_encoder/model-00002-of-00005.safetensors",
            "text_encoder/model-00003-of-00005.safetensors",
            "text_encoder/model-00004-of-00005.safetensors",
            "text_encoder/model-00005-of-00005.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors",
            "tokenizer/preprocessor_config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer_2/special_tokens_map.json",
            "tokenizer_2/tokenizer.json",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/vocab.json",
            "tokenizer_2/merges.txt",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_STABLE_CASCADE_REVISION = "a89f66d459ae653e3b4d4f992a7c3789d0dc4d16"
_STABLE_CASCADE_PRIOR_REVISION = "7ca32c21c3b4d4e35bbb94fcfedfb4fa2259bd91"

for _model_id, _decoder, _prior in (
    ("stable-cascade", "decoder", "prior"),
    ("stable-cascade-lite", "decoder_lite", "prior_lite"),
):
    _entries.append(
        ModelSpec(
            id=_model_id,
            task=Task.T2I,
            family="stable_cascade",
            source="https://huggingface.co/stabilityai/stable-cascade",
            repo_id="stabilityai/stable-cascade",
            revision=_STABLE_CASCADE_REVISION,
            provenance="author",
            catalog_id="wuerstchen",
            format="diffusers",
            license="stable-cascade-nc-community",
            gated=False,
            config={
                "prior_component": _prior,
                "decoder_component": _decoder,
                "defaults": {
                    "steps": 12,
                    "guidance": 0.0,
                    "prior_steps": 60,
                    "prior_guidance": 4.0,
                },
            },
            files=(
                "model_index.json",
                f"{_decoder}/config.json",
                f"{_decoder}/diffusion_pytorch_model.safetensors",
                "scheduler/scheduler_config.json",
                "vqgan/config.json",
                "vqgan/diffusion_pytorch_model.safetensors",
            ),
            subfolders={
                "prior": SubRepo(
                    repo_id="stabilityai/stable-cascade-prior",
                    revision=_STABLE_CASCADE_PRIOR_REVISION,
                    files=(
                        "model_index.json",
                        f"{_prior}/config.json",
                        f"{_prior}/diffusion_pytorch_model.safetensors",
                        "scheduler/scheduler_config.json",
                        "text_encoder/config.json",
                        "text_encoder/model.safetensors",
                        "tokenizer/merges.txt",
                        "tokenizer/special_tokens_map.json",
                        "tokenizer/tokenizer_config.json",
                        "tokenizer/vocab.json",
                    ),
                )
            },
        )
    )

_LLAMAGEN_REVISION = "276f5c5a3d915b922899a03f1912605531574747"
_FLAN_T5_XL_REVISION = "7d6315df2c2fb742f0f5b556879d730926ca9001"

for _llamagen_id, _llamagen_file, _llamagen_size in (
    ("llamagen-t2i-xl-256", "t2i_XL_stage1_256.pt", 256),
    ("llamagen-t2i-xl-512", "t2i_XL_stage2_512.pt", 512),
):
    _entries.append(
        ModelSpec(
            id=_llamagen_id,
            task=Task.T2I,
            family="llamagen",
            source="https://huggingface.co/peizesun/llamagen_t2i",
            repo_id="peizesun/llamagen_t2i",
            revision=_LLAMAGEN_REVISION,
            provenance="author",
            catalog_id="llamagen",
            format="pytorch",
            license="MIT",
            gated=False,
            config={
                "gpt_checkpoint": _llamagen_file,
                "defaults": {
                    "cfg_scale": 7.5,
                    "temperature": 1.0,
                    "top_k": 1000,
                    "top_p": 1.0,
                    "cfg_interval": -1,
                    "width": _llamagen_size,
                    "height": _llamagen_size,
                },
            },
            files=("vq_ds16_t2i.pt", _llamagen_file),
            subfolders={
                "text_encoder": SubRepo(
                    repo_id="google/flan-t5-xl",
                    revision=_FLAN_T5_XL_REVISION,
                    files=(
                        "config.json",
                        "model.safetensors.index.json",
                        "model-00001-of-00002.safetensors",
                        "model-00002-of-00002.safetensors",
                        "spiece.model",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                    ),
                )
            },
        )
    )

_entries.append(
    ModelSpec(
        id="wuerstchen-v2",
        task=Task.T2I,
        family="wuerstchen",
        source="https://huggingface.co/warp-ai/wuerstchen",
        repo_id="warp-ai/wuerstchen",
        revision="c3da41406ddd4d9c48c49aa93981a82354351b83",
        provenance="author",
        catalog_id="wuerstchen",
        format="diffusers",
        license="mit",
        gated=False,
        config={
            "defaults": {
                "steps": 12,
                "guidance": 0.0,
                "prior_steps": 60,
                "prior_guidance": 4.0,
            }
        },
        files=(
            "model_index.json",
            "decoder/config.json",
            "decoder/diffusion_pytorch_model.safetensors",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "vqgan/config.json",
            "vqgan/diffusion_pytorch_model.safetensors",
        ),
        subfolders={
            "prior": SubRepo(
                repo_id="warp-ai/wuerstchen-prior",
                revision="c41be6339e1b93f95ece99825327cdf9f4abc436",
                files=(
                    "model_index.json",
                    "prior/config.json",
                    "prior/diffusion_pytorch_model.safetensors",
                    "scheduler/scheduler_config.json",
                    "text_encoder/config.json",
                    "text_encoder/model.safetensors",
                    "tokenizer/merges.txt",
                    "tokenizer/special_tokens_map.json",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer/vocab.json",
                ),
            )
        },
    )
)

_entries.append(
    ModelSpec(
        id="glm-image",
        task=Task.T2I,
        family="glm-image",
        source="https://huggingface.co/zai-org/GLM-Image",
        repo_id="zai-org/GLM-Image",
        revision="2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4",
        provenance="author",
        catalog_id="glm-image",
        format="diffusers",
        license="MIT",
        gated=False,
        config={"defaults": {"steps": 50, "guidance": 1.5}},
        files=(
            "model_index.json",
            "processor/chat_template.jinja",
            "processor/preprocessor_config.json",
            "processor/tokenizer.json",
            "processor/tokenizer_config.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "tokenizer/added_tokens.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "vision_language_encoder/config.json",
            "vision_language_encoder/generation_config.json",
            "vision_language_encoder/model-00001-of-00004.safetensors",
            "vision_language_encoder/model-00002-of-00004.safetensors",
            "vision_language_encoder/model-00003-of-00004.safetensors",
            "vision_language_encoder/model-00004-of-00004.safetensors",
            "vision_language_encoder/model.safetensors.index.json",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="ernie-image",
        task=Task.T2I,
        family="ernie-image",
        source="https://huggingface.co/baidu/ERNIE-Image",
        repo_id="baidu/ERNIE-Image",
        revision="5346b31d68c9c23758ba56ef8be5e9dc174c7f99",
        provenance="author",
        catalog_id="ernie-image",
        format="diffusers",
        license="Apache-2.0",
        config={"defaults": {"steps": 50, "guidance": 4.0}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="hidream-i1-fast",
        task=Task.T2I,
        family="hidream",
        source="https://huggingface.co/HiDream-ai/HiDream-I1-Fast",
        repo_id="HiDream-ai/HiDream-I1-Fast",
        revision="4856a5d8cd6fbd194780ed9f289bdf696d3afc10",
        provenance="author",
        catalog_id="hidream-i1",
        format="diffusers",
        license="MIT",
        gated=True,
        config={"defaults": {"steps": 16, "guidance": 0.0, "width": 1024, "height": 1024}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors",
            "text_encoder_3/config.json",
            "text_encoder_3/model.safetensors.index.json",
            "text_encoder_3/model-00001-of-00002.safetensors",
            "text_encoder_3/model-00002-of-00002.safetensors",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer_2/merges.txt",
            "tokenizer_2/special_tokens_map.json",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/vocab.json",
            "tokenizer_3/special_tokens_map.json",
            "tokenizer_3/spiece.model",
            "tokenizer_3/tokenizer.json",
            "tokenizer_3/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            *(
                f"transformer/diffusion_pytorch_model-{index:05d}-of-00007.safetensors"
                for index in range(1, 8)
            ),
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
        # The Llama 3.1 encoder and its tokenizer come from Meta's gated repository.
        subfolders={
            "text_encoder_4": SubRepo(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                revision="0e9e39f249a16976918f6564b8830bc894c89659",
                files=(
                    "config.json",
                    "generation_config.json",
                    "model.safetensors.index.json",
                    "model-00001-of-00004.safetensors",
                    "model-00002-of-00004.safetensors",
                    "model-00003-of-00004.safetensors",
                    "model-00004-of-00004.safetensors",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ),
            )
        },
    )
)

# Dev and Full publish the same component layout as Fast.  The Llama encoder remains a
# separately licensed, gated subrepository for all three variants.
for _id, _repo, _revision, _steps, _guidance in (
    (
        "hidream-i1-dev",
        "HiDream-ai/HiDream-I1-Dev",
        "0fad2ea0ccf9a80ddf019ea777eedb27c1ccb232",
        30,
        0.0,
    ),
    (
        "hidream-i1-full",
        "HiDream-ai/HiDream-I1-Full",
        "8ccbbfb270ccdae26d6bb0081df67dc81e4033bf",
        50,
        5.0,
    ),
):
    _fast = _entries[-1]
    _entries.append(
        ModelSpec(
            id=_id,
            task=Task.T2I,
            family="hidream",
            source=f"https://huggingface.co/{_repo}",
            repo_id=_repo,
            revision=_revision,
            provenance="author",
            catalog_id="hidream-i1",
            format="diffusers",
            license="MIT",
            gated=True,
            config={
                "defaults": {"steps": _steps, "guidance": _guidance, "width": 1024, "height": 1024}
            },
            files=_fast.files,
            subfolders=_fast.subfolders,
        )
    )

_entries.append(
    ModelSpec(
        id="omnigen-v1",
        task=Task.T2I,
        family="omnigen",
        source="https://huggingface.co/Shitao/OmniGen-v1-diffusers",
        repo_id="Shitao/OmniGen-v1-diffusers",
        revision="016e2f61d12a98303f6bbdf122687694d7984268",
        provenance="author",
        catalog_id="omnigen",
        format="diffusers",
        license="MIT",
        gated=False,
        image_input="optional",
        config={
            "defaults": {
                "steps": 50,
                "guidance": 2.5,
                "image_guidance": 1.6,
                "width": 1024,
                "height": 1024,
            }
        },
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_entries.append(
    ModelSpec(
        id="ovis-image-7b",
        task=Task.T2I,
        family="ovis-image",
        source="https://huggingface.co/ATH-MaaS/Ovis-Image-7B",
        repo_id="ATH-MaaS/Ovis-Image-7B",
        revision="41be1c5821a92c970d63d7eb595a2fd3fe32b22e",
        provenance="author",
        catalog_id="ovis-image",
        format="diffusers",
        license="Apache-2.0",
        gated=False,
        config={"defaults": {"steps": 50, "guidance": 5.0, "width": 1024, "height": 1024}},
        files=(
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "tokenizer/added_tokens.json",
            "tokenizer/chat_template.jinja",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    )
)

_PRX_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
for _id, _repo, _revision, _size in (
    (
        "prx-1024-t2i-beta",
        "Photoroom/prx-1024-t2i-beta",
        "0370504b9a433fbdff814b8fcba9da03bf72a7a3",
        1024,
    ),
    (
        "prx-512-t2i-sft",
        "Photoroom/prx-512-t2i-sft",
        "2996423bc26e8eaca48774fac1797484214dfea0",
        512,
    ),
):
    _entries.append(
        ModelSpec(
            id=_id,
            task=Task.T2I,
            family="prx",
            source=f"https://huggingface.co/{_repo}",
            repo_id=_repo,
            revision=_revision,
            provenance="author",
            catalog_id="prx",
            format="diffusers",
            license="Apache-2.0",
            config={"defaults": {"width": _size, "height": _size}},
            files=_PRX_FILES,
        )
    )

_entries.append(
    ModelSpec(
        id="hart",
        task=Task.T2I,
        family="hart",
        source="https://huggingface.co/mit-han-lab/hart-0.7b-1024px",
        repo_id="mit-han-lab/hart-0.7b-1024px",
        revision="b1521ff1449d6817c685b08614c554e7a5b5b8eb",
        provenance="author",
        catalog_id="hart",
        format="pytorch",
        license="MIT",
        config={"defaults": {"cfg_scale": 4.5, "width": 1024, "height": 1024}},
        reference={
            "repository": {
                "url": "https://github.com/mit-han-lab/hart",
                "revision": "e28a41fe15025baf3d48ab004f5d3727c88d8fc7",
            }
        },
        files=(
            "llm/config.json",
            "llm/ema_model.bin",
            "tokenizer/config.json",
            "tokenizer/pytorch_model.bin",
        ),
        subfolders={
            "text_encoder": SubRepo(
                repo_id="mit-han-lab/Qwen2-VL-1.5B-Instruct",
                revision="6e5b5399a390528a3b2c9f5109fb795553d3db71",
                files=(
                    "config.json",
                    "generation_config.json",
                    "merges.txt",
                    "model.safetensors.index.json",
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                ),
            )
        },
    )
)

_entries.append(
    ModelSpec(
        id="show-o-512",
        task=Task.T2I,
        family="show-o",
        source="https://huggingface.co/showlab/show-o-512x512",
        repo_id="showlab/show-o-512x512",
        revision="dbe757d951099968fb102d249d35a764528df2dd",
        provenance="author",
        catalog_id="show-o",
        format="safetensors",
        license="Apache-2.0",
        config={
            "defaults": {
                "guidance": 10.0,
                "temperature": 1.0,
                "steps": 18,
                "width": 512,
                "height": 512,
            }
        },
        reference={
            "repository": {
                "url": "https://github.com/showlab/Show-o",
                "revision": "45a5a2de01d1ebd10cd5864d29310a76476cdf23",
            }
        },
        files=("config.json", "pytorch_model.safetensors"),
        subfolders={
            "vq": SubRepo(
                repo_id="showlab/magvitv2",
                revision="5c3fa78f8b3523347c5cd1a4c97f3c4e96f33d5d",
                files=("config.json", "pytorch_model.safetensors"),
            ),
            "tokenizer": SubRepo(
                repo_id="microsoft/phi-1_5",
                revision="77aa61eeac94fbf33d492b9f2744c98b42d5b5eb",
                files=(
                    "added_tokens.json",
                    "config.json",
                    "merges.txt",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                ),
            ),
        },
    )
)

_entries.append(
    ModelSpec(
        id="pixeldit",
        task=Task.T2I,
        family="pixeldit",
        source="https://huggingface.co/nvidia/PixelDiT-1300M-1024px",
        repo_id="nvidia/PixelDiT-1300M-1024px",
        revision="7c63b99a7a399918a1d6478b095698a65f664847",
        provenance="author",
        catalog_id="pixeldit",
        format="pytorch",
        license="NVIDIA License (NSCLv1; non-commercial research or evaluation)",
        config={
            "defaults": {
                "steps": 50,
                "guidance": 2.75,
                "width": 1024,
                "height": 1024,
            }
        },
        reference={
            "repository": {
                "url": "https://github.com/NVlabs/PixelDiT",
                "revision": "f7d7c35f71893d81c6e71d2576ea80601a1e8529",
            }
        },
        files=("LICENSE", "README.md", "config.json", "pixeldit_t2i_v1.pth"),
        subfolders={
            "text_encoder": SubRepo(
                repo_id="Efficient-Large-Model/gemma-2-2b-it",
                revision="569d9809d0c8b6722d4d31b5a77a2ec7a400650a",
                files=(
                    "config.json",
                    "generation_config.json",
                    "model.safetensors.index.json",
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer.model",
                    "tokenizer_config.json",
                ),
            )
        },
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
