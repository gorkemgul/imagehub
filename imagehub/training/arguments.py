"""Training settings, with defaults close to diffusers' ``train_text_to_image_lora.py``."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imagehub.training.flow_matching import WEIGHTING_SCHEMES
from imagehub.training.reporting import REPORTERS

EVAL_STRATEGIES = ("no", "steps")
LR_SCHEDULERS = ("constant", "constant_with_warmup", "linear", "cosine")
MIXED_PRECISION = ("no", "fp16", "bf16")
PREDICTION_TYPES = ("epsilon", "v_prediction")


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


def _reporters(value: Any) -> tuple[str, ...]:
    """``"none"``, one tracker name or a sequence of them, as a tuple of names."""
    names = (value,) if isinstance(value, str) else tuple(value)
    names = tuple(name for name in names if name != "none")
    for name in names:
        _choice("report_to", name, REPORTERS)
    if len(set(names)) != len(names):
        raise ValueError("report_to lists a tracker twice")
    return names


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
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
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
    gradient_checkpointing: bool = False
    noise_offset: float = 0.0
    snr_gamma: float | None = None
    prediction_type: str | None = None
    weighting_scheme: str | None = None
    logit_mean: float | None = None
    logit_std: float | None = None
    mode_scale: float | None = None
    precondition_outputs: bool | None = None
    max_sequence_length: int | None = None
    guidance_scale: float = 3.5
    logging_steps: int = 10
    save_steps: int = 500
    eval_strategy: str = "no"
    eval_steps: int | None = None
    validation_prompts: tuple[str, ...] = ()
    validation_num_inference_steps: int = 25
    validation_guidance_scale: float = 7.5
    report_to: str | tuple[str, ...] = "none"
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for name in (
            "max_steps",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "resolution",
            "lora_rank",
            "logging_steps",
            "save_steps",
            "validation_num_inference_steps",
        ):
            _positive_int(name, getattr(self, name))
        if self.eval_steps is not None:
            _positive_int("eval_steps", self.eval_steps)
        object.__setattr__(
            self,
            "validation_guidance_scale",
            _number("validation_guidance_scale", self.validation_guidance_scale),
        )
        object.__setattr__(self, "guidance_scale", _number("guidance_scale", self.guidance_scale))
        prompts = tuple(self.validation_prompts)
        if not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ValueError("validation_prompts must be non-empty strings")
        object.__setattr__(self, "validation_prompts", prompts)
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
        object.__setattr__(self, "report_to", _reporters(self.report_to))
        _choice("eval_strategy", self.eval_strategy, EVAL_STRATEGIES)
        if self.prediction_type is not None:
            _choice("prediction_type", self.prediction_type, PREDICTION_TYPES)
        if self.weighting_scheme is not None:
            _choice("weighting_scheme", self.weighting_scheme, WEIGHTING_SCHEMES)
        for name in ("logit_mean", "logit_std", "mode_scale"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise ValueError(f"{name} must be a number")
                if not math.isfinite(value):
                    raise ValueError(f"{name} must be a finite number")
                object.__setattr__(self, name, float(value))
        if self.logit_std is not None and self.logit_std <= 0:
            raise ValueError("logit_std must be positive")
        if self.precondition_outputs is not None and not isinstance(
            self.precondition_outputs, bool
        ):
            raise ValueError("precondition_outputs must be a bool")
        if self.max_sequence_length is not None:
            _positive_int("max_sequence_length", self.max_sequence_length)
        for name in ("center_crop", "random_flip", "gradient_checkpointing"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if self.lora_target_modules is not None:
            targets = tuple(self.lora_target_modules)
            if not targets or not all(isinstance(t, str) and t for t in targets):
                raise ValueError("lora_target_modules must be non-empty strings")
            object.__setattr__(self, "lora_target_modules", targets)
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a device name such as 'cpu' or 'cuda'")

    @property
    def evaluation_interval(self) -> int | None:
        """Optimizer steps between evaluations, or ``None`` when evaluation is off."""
        if self.eval_strategy == "no":
            return None
        return self.eval_steps or self.logging_steps

    def to_dict(self) -> dict[str, Any]:
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values["output_dir"] = str(self.output_dir)
        values["validation_prompts"] = list(values["validation_prompts"])
        values["report_to"] = list(values["report_to"])
        if values["lora_target_modules"] is not None:
            values["lora_target_modules"] = list(values["lora_target_modules"])
        return values
