"""What differs between model families during training, behind one interface."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from imagehub.training.arguments import TrainingArguments

Collator = Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Tensor]]


class Objective:
    """A family's training loss.

    ``objective(batch, generator)`` draws the step's random tensors with :meth:`draw` (in the
    reference script's order) and returns :meth:`loss_from`. Reference tests and the distributed
    replay call the two halves separately with recorded draws.
    """

    def draw(self, batch: Mapping[str, Tensor], generator: torch.Generator) -> dict[str, Tensor]:
        raise NotImplementedError

    def loss_from(self, batch: Mapping[str, Tensor], draws: Mapping[str, Tensor]) -> Tensor:
        raise NotImplementedError

    def __call__(self, batch: Mapping[str, Tensor], generator: torch.Generator) -> Tensor:
        return self.loss_from(batch, self.draw(batch, generator))


class TrainingFamily:
    """Loading, modules, batches, loss and sample images for one model family."""

    #: Name used in messages.
    name: str
    #: Adapter key prefix in the diffusers layout (``unet`` or ``transformer``).
    prefix: str
    #: Keys the collator must produce.
    batch_keys: tuple[str, ...]
    #: Block classes recomputed in the backward pass with ``gradient_checkpointing``.
    checkpoint_blocks: tuple[type, ...] = ()

    def validate(self, args: TrainingArguments) -> None:
        """Reject settings the family's objective does not use."""

    def load(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        model_id: str,
    ) -> Any:
        raise NotImplementedError

    def trainable(self, components: Any) -> nn.Module:
        """The module that receives the LoRA adapter."""
        raise NotImplementedError

    def frozen(self, components: Any) -> list[nn.Module]:
        """Modules that run without gradients (text encoders, autoencoder)."""
        raise NotImplementedError

    def collator(
        self,
        components: Any,
        args: TrainingArguments,
        *,
        generator: torch.Generator,
        train: bool,
    ) -> Collator:
        """The default collator; ``train=False`` gives the deterministic evaluation transform."""
        raise NotImplementedError

    def objective(self, components: Any, args: TrainingArguments) -> Objective:
        raise NotImplementedError

    def sample(
        self, components: Any, prompt: str, args: TrainingArguments, device: torch.device
    ) -> Tensor:
        """One validation image, ``(H, W, 3)`` uint8, as the family's pipeline renders it."""
        raise NotImplementedError


_FAMILIES = {
    "stable-diffusion": "imagehub.training.sd1:StableDiffusionFamily",
    "stable-diffusion-xl": "imagehub.training.sdxl:StableDiffusionXLFamily",
    "stable-diffusion-3": "imagehub.training.sd3:StableDiffusion3Family",
    "flux": "imagehub.training.flux:Flux1Family",
    "sana": "imagehub.training.sana:SanaFamily",
    "qwen-image": "imagehub.training.qwen_image:QwenImageFamily",
    "z-image": "imagehub.training.z_image:ZImageFamily",
}

FLOW_MATCHING_OPTIONS = (
    "weighting_scheme",
    "logit_mean",
    "logit_std",
    "mode_scale",
    "precondition_outputs",
)
DDPM_OPTIONS = ("prediction_type", "snr_gamma", "noise_offset")


def reject_options(args: TrainingArguments, names: tuple[str, ...], family: str) -> None:
    """Raise when any of ``names`` differs from its default for a family that ignores it."""
    defaults = {"noise_offset": 0.0, "guidance_scale": 3.5, "complex_human_instruction": None}
    used = [name for name in names if getattr(args, name) != defaults.get(name)]
    if used:
        raise ValueError(f"{', '.join(used)} do not apply to {family} training")


def get_family(name: str) -> TrainingFamily:
    """The training family registered for a catalog family name."""
    if name not in _FAMILIES:
        raise KeyError(f"No training family for {name!r}")
    module, attribute = _FAMILIES[name].split(":")
    family = getattr(import_module(module), attribute)()
    assert isinstance(family, TrainingFamily)
    return family


def training_families() -> tuple[str, ...]:
    return tuple(_FAMILIES)
