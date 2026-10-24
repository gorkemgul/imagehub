"""Spatiotemporal Emu3 vision tokenizer with the published tensor layout."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel


def _silu(value: Tensor) -> Tensor:
    return value * torch.sigmoid(value)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        inputs: int,
        outputs: int,
        kernel: int | tuple[int, int, int] = (3, 1, 1),
        stride: int | tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        if isinstance(kernel, int):
            kernel = (kernel,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        padding: tuple[int, ...] = ()
        for amount in [k - s for k, s in zip(kernel[1:], stride[1:], strict=True)][::-1]:
            padding += (amount // 2 + amount % 2, amount // 2)
        self.padding = padding + (2, 0)
        self.conv = nn.Conv3d(inputs, outputs, kernel, stride=stride)

    def forward(self, value: Tensor) -> Tensor:
        return self.conv(F.pad(value, self.padding))


class TemporalResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm3d(channels)
        self.conv1 = CausalConv3d(channels, channels, (3, 3, 3))
        self.norm2 = nn.BatchNorm3d(channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(channels, channels, (3, 3, 3))

    def forward(self, value: Tensor) -> Tensor:
        hidden = self.conv1(_silu(self.norm1(value)))
        hidden = self.conv2(self.dropout(_silu(self.norm2(hidden))))
        return value + hidden


class TemporalConv(nn.Module):
    def __init__(self, channels: int, *, downsample: bool) -> None:
        super().__init__()
        kernel = (4, 3, 3) if downsample else (3, 3, 3)
        stride = (2, 1, 1) if downsample else (1, 1, 1)
        self.conv = CausalConv3d(channels, channels, kernel, stride)

    def forward(self, value: Tensor) -> Tensor:
        return self.conv(value)


class SpatialNorm(nn.Module):
    def __init__(self, channels: int, latent_channels: int) -> None:
        super().__init__()
        self.norm_layer = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_y = nn.Conv2d(latent_channels, channels, 1)
        self.conv_b = nn.Conv2d(latent_channels, channels, 1)

    def forward(self, value: Tensor, latent: Tensor) -> Tensor:
        latent = F.interpolate(latent, size=value.shape[-2:], mode="nearest")
        return self.norm_layer(value) * self.conv_y(latent) + self.conv_b(latent)


class ResBlock(nn.Module):
    def __init__(
        self, inputs: int, outputs: int, dropout: float, latent_channels: int | None = None
    ) -> None:
        super().__init__()
        norm = (
            (lambda channels: nn.GroupNorm(32, channels, eps=1e-6))
            if latent_channels is None
            else (lambda channels: SpatialNorm(channels, latent_channels))
        )
        self.conditioned = latent_channels is not None
        self.norm1 = norm(inputs)
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.norm2 = norm(outputs)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
        if inputs != outputs:
            self.nin_shortcut = nn.Conv2d(inputs, outputs, 1)

    def forward(self, value: Tensor, latent: Tensor | None = None) -> Tensor:
        args = (latent,) if self.conditioned else ()
        hidden = self.conv1(_silu(self.norm1(value, *args)))
        hidden = self.conv2(self.dropout(_silu(self.norm2(hidden, *args))))
        residual = self.nin_shortcut(value) if hasattr(self, "nin_shortcut") else value
        return residual + hidden


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, latent_channels: int | None = None) -> None:
        super().__init__()
        self.conditioned = latent_channels is not None
        self.norm = (
            nn.GroupNorm(32, channels, eps=1e-6)
            if latent_channels is None
            else SpatialNorm(channels, latent_channels)
        )
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, value: Tensor, latent: Tensor | None = None) -> Tensor:
        hidden = self.norm(value, *((latent,) if self.conditioned else ()))
        query, key, val = self.q(hidden), self.k(hidden), self.v(hidden)
        batch, channels, height, width = query.shape
        weights = torch.bmm(
            query.reshape(batch, channels, -1).permute(0, 2, 1),
            key.reshape(batch, channels, -1),
        )
        weights = weights / channels**0.5
        weights = weights.softmax(2)
        hidden = torch.bmm(val.reshape(batch, channels, -1), weights.permute(0, 2, 1))
        return value + self.proj_out(hidden.reshape(batch, channels, height, width))


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2)

    def forward(self, value: Tensor) -> Tensor:
        return self.conv(F.pad(value, (0, 1, 0, 1)))


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: Tensor) -> Tensor:
        return self.conv(F.interpolate(value, scale_factor=2.0, mode="nearest"))


def _container(**modules: nn.Module) -> nn.Module:
    result = nn.Module()
    for name, module in modules.items():
        setattr(result, name, module)
    return result


class Encoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        base, multipliers = config["ch"], config["ch_mult"]
        self.levels, self.blocks = len(multipliers), config["num_res_blocks"]
        self.conv_in = nn.Conv2d(config["in_channels"], base, 3, padding=1)
        self.down = nn.ModuleList()
        block_in = base
        for level, multiplier in enumerate(multipliers):
            block_out = base * multiplier
            blocks = nn.ModuleList()
            attention = nn.ModuleList()
            for _ in range(self.blocks):
                blocks.append(ResBlock(block_in, block_out, config["dropout"]))
                block_in = block_out
                if level in config["attn_resolutions"]:
                    attention.append(AttentionBlock(block_in))
            parts: dict[str, nn.Module] = {"block": blocks, "attn": attention}
            if level != self.levels - 1:
                parts["downsample"] = Downsample(block_in)
            self.down.append(_container(**parts))
        self.mid = _container(
            block_1=ResBlock(block_in, block_in, config["dropout"]),
            attn_1=AttentionBlock(block_in),
            block_2=ResBlock(block_in, block_in, config["dropout"]),
        )
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        channels = config["z_channels"] * (2 if config["double_z"] else 1)
        self.conv_out = nn.Conv2d(block_in, channels, 3, padding=1)
        temporal = int(math.log2(config["temporal_downsample_factor"]))
        self.time_conv = nn.ModuleList(
            TemporalConv(channels, downsample=True) for _ in range(temporal)
        )
        self.time_res_stack = nn.Sequential(
            *(TemporalResBlock(channels, config["dropout"]) for _ in range(self.blocks))
        )

    def forward(self, value: Tensor) -> Tensor:
        time = value.shape[1]
        hidden = self.conv_in(value.flatten(0, 1))
        for level in range(self.levels):
            down = cast(Any, self.down[level])
            for block in range(self.blocks):
                hidden = down.block[block](hidden)
                if len(down.attn):
                    hidden = down.attn[block](hidden)
            if level != self.levels - 1:
                hidden = down.downsample(hidden)
        mid = cast(Any, self.mid)
        hidden = mid.block_2(mid.attn_1(mid.block_1(hidden)))
        hidden = self.conv_out(_silu(self.norm_out(hidden)))
        hidden = hidden.reshape(-1, time, *hidden.shape[1:]).permute(0, 2, 1, 3, 4)
        for conv in self.time_conv:
            hidden = _silu(conv(hidden))
        return self.time_res_stack(hidden).permute(0, 2, 1, 3, 4)


class Decoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        base, multipliers = config["ch"], config["ch_mult"]
        self.levels, self.blocks = len(multipliers), config["num_res_blocks"]
        latent, channels = config["embed_dim"], config["z_channels"]
        self.time_res_stack = nn.Sequential(
            *(TemporalResBlock(channels, config["dropout"]) for _ in range(self.blocks))
        )
        temporal = int(math.log2(config["temporal_downsample_factor"]))
        self.time_conv = nn.ModuleList(
            TemporalConv(channels, downsample=False) for _ in range(temporal)
        )
        block_in = base * multipliers[-1]
        self.conv_in = nn.Conv2d(channels, block_in, 3, padding=1)
        self.mid = _container(
            block_1=ResBlock(block_in, block_in, config["dropout"], latent),
            attn_1=AttentionBlock(block_in, latent),
            block_2=ResBlock(block_in, block_in, config["dropout"], latent),
        )
        self.up = nn.ModuleList()
        for level in reversed(range(self.levels)):
            block_out = base * multipliers[level]
            blocks, attention = nn.ModuleList(), nn.ModuleList()
            for _ in range(self.blocks + 1):
                blocks.append(ResBlock(block_in, block_out, config["dropout"], latent))
                block_in = block_out
                if level in config["attn_resolutions"]:
                    attention.append(AttentionBlock(block_in, latent))
            parts: dict[str, nn.Module] = {"block": blocks, "attn": attention}
            if level != 0:
                parts["upsample"] = Upsample(block_in)
            self.up.insert(0, _container(**parts))
        self.norm_out = SpatialNorm(block_in, latent)
        self.conv_out = nn.Conv2d(block_in, config["out_channels"], 3, padding=1)

    def forward(self, quantized: Tensor, latent: Tensor) -> Tensor:
        combined = torch.cat((quantized, latent)).permute(0, 2, 1, 3, 4)
        combined = self.time_res_stack(combined)
        for conv in self.time_conv:
            batch, channels, time, height, width = combined.shape
            combined = combined.permute(0, 1, 3, 4, 2).reshape(batch, -1, time)
            combined = F.interpolate(combined, scale_factor=2.0, mode="nearest")
            combined = combined.view(batch, channels, height, width, -1).permute(0, 1, 4, 2, 3)
            combined = _silu(conv(combined.contiguous()))
        hidden, latent = torch.chunk(combined.permute(0, 2, 1, 3, 4), 2)
        hidden = self.conv_in(hidden.flatten(0, 1))
        latent = latent.flatten(0, 1)
        mid = cast(Any, self.mid)
        hidden = mid.block_2(mid.attn_1(mid.block_1(hidden, latent), latent), latent)
        for level in reversed(range(self.levels)):
            up = cast(Any, self.up[level])
            for block in range(self.blocks + 1):
                hidden = up.block[block](hidden, latent)
                if len(up.attn):
                    hidden = up.attn[block](hidden, latent)
            if level != 0:
                hidden = up.upsample(hidden)
        return self.conv_out(_silu(self.norm_out(hidden, latent)))


class VectorQuantizer(nn.Module):
    def __init__(self, size: int, dimensions: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(size, dimensions)


class Emu3VisionVQ(NativeModel):
    """Complete Emu3 vision tokenizer; image generation uses ``decode``."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.quantize = VectorQuantizer(config["codebook_size"], config["embed_dim"])
        self.quant_conv = CausalConv3d(config["z_channels"], config["embed_dim"])
        self.post_quant_conv = CausalConv3d(config["embed_dim"], config["z_channels"])
        self.spatial_scale_factor = 2 ** (len(config["ch_mult"]) - 1)

    def decode(self, tokens: Tensor) -> Tensor:
        ndim = tokens.ndim
        if ndim == 3:
            tokens = tokens.unsqueeze(1)
        batch, time, height, width = tokens.shape
        latent = self.quantize.embedding(tokens.flatten()).view(
            batch, time, height, width, self.config["embed_dim"]
        )
        latent_3d = latent.permute(0, 4, 1, 2, 3).contiguous()
        quantized = self.post_quant_conv(latent_3d)
        video = self.decoder(quantized.permute(0, 2, 1, 3, 4), latent_3d.permute(0, 2, 1, 3, 4))
        video = video.reshape(
            batch,
            time * self.config["temporal_downsample_factor"],
            self.config["out_channels"],
            height * self.spatial_scale_factor,
            width * self.spatial_scale_factor,
        )
        return video[:, 0] if ndim == 3 else video
