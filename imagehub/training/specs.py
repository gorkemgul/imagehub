"""Training profiles: which catalog models imagehub can fine-tune, and how."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from enum import StrEnum

from imagehub.catalog import MODEL_CATALOG


class TrainingSupport(StrEnum):
    """How far a model's training path is implemented and verified.

    ``NATIVE`` trains from images and captions with an objective checked against the reference
    training script. ``INFERENCE_ONLY`` has no training path yet.
    """

    NATIVE = "native"
    INFERENCE_ONLY = "inference_only"

    @property
    def is_trainable(self) -> bool:
        return self is not TrainingSupport.INFERENCE_ONLY


@dataclass(frozen=True)
class TrainingSpec:
    model_id: str
    family: str
    support: TrainingSupport
    methods: tuple[str, ...] = ()
    target_modules: tuple[str, ...] = ()
    lora_rank: int = 4
    lora_alpha: float = 4.0


_UNET_ATTENTION = ("to_q", "to_k", "to_v", "to_out.0")
# Catalog families with a training family, and their default LoRA targets.
_JOINT_ATTENTION = (
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
)
_FLUX_TARGETS = (
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
)
_KONTEXT_TARGETS = _FLUX_TARGETS + ("proj_mlp",)
_SANA_TARGETS = ("to_k", "to_q", "to_v")
_QWEN_IMAGE_TARGETS = ("to_k", "to_q", "to_v", "to_out.0")
_TRAINABLE = {
    "stable-diffusion": _UNET_ATTENTION,
    "stable-diffusion-xl": _UNET_ATTENTION,
    "stable-diffusion-3": _JOINT_ATTENTION,
    "flux": _FLUX_TARGETS,
    "sana": _SANA_TARGETS,
    "qwen-image": _QWEN_IMAGE_TARGETS,
}
# Few-step distilled checkpoints are not trained with the denoising objective; the tiny
# autoencoder of SDXS has no posterior to sample training latents from; the SDXL refiner is an
# image-to-image model; Playground v2.5 uses the EDM formulation, not DDPM.
# sd3.5-large-turbo is distilled as well.
_EXCLUDED = frozenset(
    {
        "sd-turbo",
        "lcm-dreamshaper-v7",
        "sdxs-512-0.9",
        "sdxl-refiner-1.0",
        "sdxl-lightning-2step",
        "sdxl-lightning-4step",
        "sdxl-lightning-8step",
        "hyper-sdxl-1step",
        "dmd2-sdxl-4step",
        "dmd2-sdxl-1step",
        "sdxl-turbo",
        "koala-lightning-700m",
        "koala-lightning-1b",
        "playground-v2.5",
        "sd3.5-large-turbo",
        "flux1-schnell",
        "sana-sprint-1.6b-1024",
    }
)


def _spec(model_id: str) -> TrainingSpec:
    model = MODEL_CATALOG[model_id]
    targets = _TRAINABLE.get(model.family)
    if model_id == "flux1-kontext-dev":
        targets = _KONTEXT_TARGETS
    if targets is not None and model_id not in _EXCLUDED:
        rank = 16 if model_id == "qwen-image-2.1" else 4
        return TrainingSpec(
            model_id,
            model.family,
            TrainingSupport.NATIVE,
            ("lora",),
            targets,
            rank,
            float(rank),
        )
    return TrainingSpec(model_id, model.family, TrainingSupport.INFERENCE_ONLY)


def get_training_spec(model_id: str) -> TrainingSpec:
    """Return the training profile of a catalog model id."""
    if model_id not in MODEL_CATALOG:
        close = difflib.get_close_matches(model_id, list(MODEL_CATALOG), n=3)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise KeyError(f"No training profile for {model_id!r}.{hint}")
    return _spec(model_id)


def list_training_specs(
    *, support: TrainingSupport | str | None = None
) -> tuple[TrainingSpec, ...]:
    """Every catalog model's profile in catalog order, optionally filtered by support level."""
    wanted = None if support is None else TrainingSupport(support)
    specs = (_spec(model_id) for model_id in MODEL_CATALOG)
    return tuple(spec for spec in specs if wanted is None or spec.support is wanted)
