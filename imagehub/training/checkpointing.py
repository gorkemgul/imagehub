"""Gradient checkpointing for native models without changing their code."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def enable_gradient_checkpointing(model: nn.Module, block_types: tuple[type, ...]) -> int:
    """Recompute every ``block_types`` block in the backward pass; returns the block count.

    Activations inside the blocks are dropped after the forward pass and recomputed when the
    gradients are needed, trading compute for memory. Without gradients (evaluation, sampling)
    the blocks run normally.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, block_types) and not getattr(module, "_checkpointed", False):
            forward = module.forward

            def run(*args: Any, _forward: Any = forward, **kwargs: Any) -> Any:
                if torch.is_grad_enabled():
                    return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
                return _forward(*args, **kwargs)

            module.forward = run
            setattr(module, "_checkpointed", True)  # noqa: B010
            count += 1
    return count
