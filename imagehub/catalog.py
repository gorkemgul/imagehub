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
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be a 40-character lowercase commit hash")
        if self.provenance not in PROVENANCES:
            raise ValueError(f"provenance must be one of {', '.join(PROVENANCES)}")
        if self.base is not None and not any(entry.id == self.base for entry in _entries):
            raise ValueError(f"base model {self.base!r} does not exist")
        if self.overlay and self.base is None:
            raise ValueError("overlay requires a base model")
        if self.base_exclude and self.base is None:
            raise ValueError("base_exclude requires a base model")
        for name in self.base_exclude:
            path = PurePosixPath(name)
            if not name or path.is_absolute() or "." in path.parts or ".." in path.parts:
                raise ValueError(f"base_exclude path {name!r} is not a safe relative path")
        for destination, source in self.overlay.items():
            for name in (destination, source):
                path = PurePosixPath(name)
                if not name or path.is_absolute() or "." in path.parts or ".." in path.parts:
                    raise ValueError(f"overlay path {name!r} is not a safe relative path")
            if source not in self.files:
                raise ValueError(f"overlay source {source!r} is not listed in files")


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
        files=(
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
        ),
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
for _id, _repo, _revision, _catalog_id, _max_length, _files in (
    (
        "pixart-sigma-xl-1024",
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "e102b3591cc82e97071b8b4cb90d834d0c487207",
        "pixart-sigma",
        300,
        _PIXART_COMMON,
    ),
    (
        "pixart-alpha-xl-1024",
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        "b89adadeccd9ead2adcb9fa2825d3fabec48d404",
        "pixart-alpha",
        120,
        _PIXART_COMMON[:6] + ("tokenizer/added_tokens.json",) + _PIXART_COMMON[6:],
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
                }
            },
            files=_files,
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
