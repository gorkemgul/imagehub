"""Infinity's d32 binary-spherical-quantized image decoder."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

INFINITY_VAE_CONFIG: dict[str, Any] = {
    "base_channels": 128,
    "channel_multipliers": (1, 2, 4, 4, 4),
    "num_res_blocks": 2,
    "latent_channels": 32,
    "patch_size": 16,
}


def _swish(value: Tensor) -> Tensor:
    return value * torch.sigmoid(value)


class InfinityConv(nn.Module):
    """The published wrapper around a 2D convolution (keeps the ``conv`` key level)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.conv(value)


class InfinityNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 32 if channels % 32 == 0 else 24
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6, affine=True)

    def forward(self, value: Tensor) -> Tensor:
        return self.norm(value)


class InfinityResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = InfinityNorm(in_channels)
        self.conv1 = InfinityConv(in_channels, out_channels, 3, padding=1)
        self.norm2 = InfinityNorm(out_channels)
        self.conv2 = InfinityConv(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = InfinityConv(in_channels, out_channels, 1)

    def forward(self, value: Tensor) -> Tensor:
        hidden = self.conv1(_swish(self.norm1(value)))
        hidden = self.conv2(_swish(self.norm2(hidden)))
        if hasattr(self, "nin_shortcut"):
            value = self.nin_shortcut(value)
        return value + hidden


class InfinityDownsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = InfinityConv(channels, channels, 3, stride=2)

    def forward(self, value: Tensor) -> Tensor:
        return self.conv(F.pad(value, (0, 1, 0, 1)))


class InfinityUpsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = InfinityConv(channels, channels, 3, padding=1)

    def forward(self, value: Tensor) -> Tensor:
        return self.conv(F.interpolate(value, scale_factor=2.0, mode="nearest"))


class _Level(nn.Module):
    def __init__(self, blocks: Sequence[nn.Module], sample: nn.Module | None, name: str) -> None:
        super().__init__()
        self.block = nn.ModuleList(blocks)
        self.attn = nn.ModuleList()
        if sample is not None:
            setattr(self, name, sample)


class _Middle(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block_1 = InfinityResnetBlock(channels, channels)
        self.block_2 = InfinityResnetBlock(channels, channels)

    def forward(self, value: Tensor) -> Tensor:
        return self.block_2(self.block_1(value))


class InfinityEncoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        base = config["base_channels"]
        multipliers = config["channel_multipliers"]
        blocks = config["num_res_blocks"]
        max_down = int(math.log2(config["patch_size"]))
        self.conv_in = InfinityConv(3, base, 3, padding=1)
        self.down = nn.ModuleList()
        block_in = base
        for level, multiplier in enumerate(multipliers):
            block_out = base * multiplier
            level_blocks = []
            for _ in range(blocks):
                level_blocks.append(InfinityResnetBlock(block_in, block_out))
                block_in = block_out
            sample = InfinityDownsample(block_in) if level < max_down else None
            self.down.append(_Level(level_blocks, sample, "downsample"))
        self.mid = _Middle(block_in)
        self.norm_out = InfinityNorm(block_in)
        self.conv_out = InfinityConv(block_in, config["latent_channels"], 3, padding=1)

    def forward(self, value: Tensor) -> Tensor:
        hidden = self.conv_in(value)
        for level in self.down:
            for block in cast(nn.ModuleList, level.block):
                hidden = block(hidden)
            if hasattr(level, "downsample"):
                hidden = cast(nn.Module, level._modules["downsample"])(hidden)
        return self.conv_out(_swish(self.norm_out(self.mid(hidden))))


class InfinityDecoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        base = config["base_channels"]
        multipliers = config["channel_multipliers"]
        blocks = config["num_res_blocks"]
        max_up = int(math.log2(config["patch_size"]))
        block_in = base * multipliers[-1]
        self.conv_in = InfinityConv(config["latent_channels"], block_in, 3, padding=1)
        self.mid = _Middle(block_in)
        levels: list[nn.Module] = []
        for level in reversed(range(len(multipliers))):
            block_out = base * multipliers[level]
            level_blocks = []
            for _ in range(blocks + 1):
                level_blocks.append(InfinityResnetBlock(block_in, block_out))
                block_in = block_out
            sample = InfinityUpsample(block_in) if 1 <= level <= max_up else None
            levels.insert(0, _Level(level_blocks, sample, "upsample"))
        self.up = nn.ModuleList(levels)
        self.norm_out = InfinityNorm(block_in)
        self.conv_out = InfinityConv(block_in, 3, 3, padding=1)

    def forward(self, value: Tensor) -> Tensor:
        hidden = self.mid(self.conv_in(value))
        for level_index in reversed(range(len(self.up))):
            level = self.up[level_index]
            for block in cast(nn.ModuleList, level.block):
                hidden = block(hidden)
            if hasattr(level, "upsample"):
                hidden = cast(nn.Module, level._modules["upsample"])(hidden)
        return self.conv_out(_swish(self.norm_out(hidden)))


class InfinityVAE(NativeModel):
    """The published Infinity image tokenizer; generation uses its decoder and bit converter."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(INFINITY_VAE_CONFIG | dict(config or {}))
        self.encoder = InfinityEncoder(self.config)
        self.decoder = InfinityDecoder(self.config)

    def prepare_state_dict(self, state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
        state = dict(state_dict)
        # Quantizer buffers are deterministic and the d32 checkpoint may store them.
        for name in tuple(state):
            if name.startswith(("quantizer.", "quantize.", "perceptual_model.")):
                state.pop(name)
        return state

    def bits_to_codes(self, bits: Tensor) -> Tensor:
        scale = 1.0 / math.sqrt(self.config["latent_channels"])
        codes = bits * (2.0 * scale) - scale
        codes = F.normalize(codes, dim=-1)
        return codes.movedim(-1, 1)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent).clamp(-1, 1)
