"""Process-group helpers for data-parallel training under ``torchrun``."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from imagehub.backends.stable_diffusion import resolve_device


@dataclass(frozen=True)
class DistributedState:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(device: str) -> tuple[DistributedState, torch.device]:
    """Join the ``torchrun`` process group when ``WORLD_SIZE > 1``; return the process device."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedState(), resolve_device(device)
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device.startswith("cuda"):
        selected = torch.device("cuda", local_rank)
        torch.cuda.set_device(selected)
        backend = "nccl"
    elif device == "cpu":
        selected = torch.device("cpu")
        backend = "gloo"
    else:
        raise ValueError("Distributed training runs on CUDA or the CPU")
    if not dist.is_initialized():
        dist.init_process_group(backend)
    return DistributedState(rank, world_size, local_rank), selected


def average_gradients(parameters: Iterable[Tensor], state: DistributedState) -> None:
    """Replace every gradient with its mean over the processes."""
    if not state.enabled:
        return
    import torch.distributed as dist

    for parameter in parameters:
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(state.world_size)


def reduce_sum(values: list[float], state: DistributedState, device: torch.device) -> list[float]:
    """Sum floats over the processes (in float64)."""
    if not state.enabled:
        return values
    import torch.distributed as dist

    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def barrier(state: DistributedState) -> None:
    if state.enabled:
        import torch.distributed as dist

        dist.barrier()


def gather_tensor(tensor: Tensor, state: DistributedState, device: torch.device) -> list[Tensor]:
    """Every process's ``tensor`` (same shape and dtype everywhere), in rank order, on the CPU.

    Generator states are gathered this way rather than with ``all_gather_object``, which needs
    numpy.
    """
    if not state.enabled:
        return [tensor]
    import torch.distributed as dist

    local = tensor.to(device)
    values = [torch.empty_like(local) for _ in range(state.world_size)]
    dist.all_gather(values, local)
    return [value.cpu() for value in values]
