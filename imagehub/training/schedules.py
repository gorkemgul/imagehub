"""Learning-rate multipliers with the formulas of ``diffusers.optimization.get_scheduler``."""

from __future__ import annotations

import math
from collections.abc import Callable


def lr_lambda(name: str, warmup: int, total: int) -> Callable[[int], float]:
    """The ``LambdaLR`` multiplier for schedule ``name`` over ``total`` optimizer steps."""

    def warming(step: int) -> float | None:
        if step < warmup:
            return float(step) / float(max(1, warmup))
        return None

    if name == "constant":
        return lambda step: 1.0
    if name == "constant_with_warmup":
        return lambda step: w if (w := warming(step)) is not None else 1.0
    if name == "linear":

        def linear(step: int) -> float:
            if (w := warming(step)) is not None:
                return w
            return max(0.0, float(total - step) / float(max(1, total - warmup)))

        return linear
    if name == "cosine":

        def cosine(step: int) -> float:
            if (w := warming(step)) is not None:
                return w
            progress = float(step - warmup) / float(max(1, total - warmup))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))

        return cosine
    raise ValueError(f"Unknown learning-rate schedule {name!r}")
