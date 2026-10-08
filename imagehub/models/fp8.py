"""Weight-only float8 linear layers and checkpoint loading."""

from __future__ import annotations

import warnings

import torch
from torch import nn
from torch.nn import functional as F

FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"


def quantize_weight_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a two-dimensional weight with one float32 scale per row."""
    w = weight.detach().to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_E4M3_MAX
    q = (w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
    return q, scale.squeeze(1).to(torch.float32)


def is_fp8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    """Return whether a state dictionary contains weight-only float8 tensors."""
    return any(k.endswith(FP8_SCALE_SUFFIX) for k in state_dict) or any(
        v.dtype == FP8_WEIGHT_DTYPE for v in state_dict.values()
    )


class Fp8Linear(nn.Module):
    """Linear layer with an e4m3 weight and per-output-row float32 scale."""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE),
        )
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def swap_linears_to_fp8(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    compute_dtype: torch.dtype,
    *,
    prefix: str = "",
) -> None:
    """Replace linears for which the checkpoint contains a float8 scale."""
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}{name}"
        if isinstance(child, nn.Linear) and f"{child_prefix}{FP8_SCALE_SUFFIX}" in state_dict:
            setattr(
                module,
                name,
                Fp8Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                ),
            )
        else:
            swap_linears_to_fp8(child, state_dict, compute_dtype, prefix=f"{child_prefix}.")


def load_fp8_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    *,
    assign: bool = False,
    strict: bool = True,
) -> None:
    """Load float8 weights without casting them or their float32 scales."""
    prepared: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if v.dtype == FP8_WEIGHT_DTYPE:
            prepared[k] = v.to(device=device)
        elif k.endswith(FP8_SCALE_SUFFIX):
            prepared[k] = v.to(device=device, dtype=torch.float32)
        elif v.is_floating_point():
            prepared[k] = v.to(device=device, dtype=dtype)
        else:
            prepared[k] = v.to(device=device)

    missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
    if unexpected:
        raise RuntimeError(f"unexpected keys after fp8 load: {unexpected[:10]}")
    if missing:
        if strict:
            raise RuntimeError(f"missing keys after fp8 load: {missing[:10]}")
        warnings.warn(f"missing keys after fp8 load: {missing[:10]}", stacklevel=2)

    model.to(device)
