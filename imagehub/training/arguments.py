"""Training settings, with defaults close to diffusers' ``train_text_to_image_lora.py``."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LR_SCHEDULERS = ("constant", "constant_with_warmup", "linear", "cosine")
MIXED_PRECISION = ("no", "fp16", "bf16")
PREDICTION_TYPES = ("epsilon", "v_prediction")
REPORTERS = ("none", "wandb")


def _positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")


def _non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")


def _number(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return float(value)


def _choice(name: str, value: Any, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")


@dataclass(frozen=True)
class TrainingArguments:
    """Settings for :class:`imagehub.training.Trainer`. Only ``output_dir`` and ``max_steps``
    are required."""

    output_dir: str | os.PathLike[str]
    max_steps: int
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    mixed_precision: str = "no"
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False
    lora_rank: int = 4
    lora_alpha: float = 4.0
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] | None = None
    noise_offset: float = 0.0
    snr_gamma: float | None = None
    prediction_type: str | None = None
    logging_steps: int = 10
    save_steps: int = 500
    report_to: str = "none"
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for name in (
            "max_steps",
            "train_batch_size",
            "gradient_accumulation_steps",
            "resolution",
            "lora_rank",
            "logging_steps",
            "save_steps",
        ):
            _positive_int(name, getattr(self, name))
        _non_negative_int("lr_warmup_steps", self.lr_warmup_steps)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an int")
        for name in ("learning_rate", "adam_epsilon", "lora_alpha"):
            object.__setattr__(self, name, _number(name, getattr(self, name), positive=True))
        for name in ("max_grad_norm", "adam_weight_decay", "noise_offset", "lora_dropout"):
            object.__setattr__(self, name, _number(name, getattr(self, name)))
        for name in ("adam_beta1", "adam_beta2"):
            value = _number(name, getattr(self, name))
            if value >= 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.lora_dropout >= 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.snr_gamma is not None:
            object.__setattr__(
                self, "snr_gamma", _number("snr_gamma", self.snr_gamma, positive=True)
            )
        _choice("lr_scheduler", self.lr_scheduler, LR_SCHEDULERS)
        _choice("mixed_precision", self.mixed_precision, MIXED_PRECISION)
        _choice("report_to", self.report_to, REPORTERS)
        if self.prediction_type is not None:
            _choice("prediction_type", self.prediction_type, PREDICTION_TYPES)
        for name in ("center_crop", "random_flip"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if self.lora_target_modules is not None:
            targets = tuple(self.lora_target_modules)
            if not targets or not all(isinstance(t, str) and t for t in targets):
                raise ValueError("lora_target_modules must be non-empty strings")
            object.__setattr__(self, "lora_target_modules", targets)
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a device name such as 'cpu' or 'cuda'")

    def to_dict(self) -> dict[str, Any]:
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values["output_dir"] = str(self.output_dir)
        if values["lora_target_modules"] is not None:
            values["lora_target_modules"] = list(values["lora_target_modules"])
        return values
