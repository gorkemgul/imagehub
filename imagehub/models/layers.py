"""Shared convolutional diffusion blocks (ResNet, attention, resampling)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def timestep_embedding(
    timesteps: Tensor, dim: int, *, flip_sin_to_cos: bool, shift: float
) -> Tensor:
    """Sinusoidal timestep features, numerically matching diffusers.get_timestep_embedding."""
    half = dim // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half - shift)
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
    return emb


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        temb_channels: int | None,
        eps: float,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels) if temb_channels else None
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x: Tensor, temb: Tensor | None = None) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if self.time_emb_proj is not None and temb is not None:
            h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class Downsample(nn.Module):
    """Stride-2 convolution. ``padding=0`` pads right/bottom by one, as in the LDM autoencoder."""

    def __init__(self, channels: int, *, padding: int) -> None:
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        if self.padding == 0:
            x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor, size: tuple[int, int] | None = None) -> Tensor:
        if size is None:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        else:
            x = F.interpolate(x, size=size, mode="nearest")
        return self.conv(x)


def attention(q: Tensor, k: Tensor, v: Tensor, heads: int) -> Tensor:
    """Multi-head scaled dot-product attention over (batch, tokens, channels) inputs."""
    b = q.shape[0]
    q, k, v = (t.view(b, -1, heads, t.shape[-1] // heads).transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(b, -1, heads * out.shape[-1])
