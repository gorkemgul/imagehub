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


_UNET_ATTENTION = ("to_q", "to_k", "to_v", "to_out.0")
# Few-step distilled checkpoints are not trained with the denoising objective, and the tiny
# autoencoder of SDXS has no posterior to sample training latents from.
_EXCLUDED = frozenset({"sd-turbo", "lcm-dreamshaper-v7", "sdxs-512-0.9"})


def _spec(model_id: str) -> TrainingSpec:
    model = MODEL_CATALOG[model_id]
    if model.family == "stable-diffusion" and model_id not in _EXCLUDED:
        return TrainingSpec(
            model_id, model.family, TrainingSupport.NATIVE, ("lora",), _UNET_ATTENTION
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
