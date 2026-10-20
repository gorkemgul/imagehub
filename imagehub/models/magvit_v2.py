"""MAGVIT-v2 lookup-free image tokenizer used by Show-o."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

MAGVIT_V2_CONFIG: dict[str, Any] = {
    "ch": 128,
    "encoder_ch_mult": [1, 2, 2, 4, 4],
    "encoder_num_res_blocks": [4, 3, 4, 3, 4],
    "decoder_ch_mult": [1, 1, 2, 2, 4],
    "decoder_num_res_blocks": [4, 4, 3, 4, 3],
    "attn_resolutions": [5],
    "resolution": 256,
    "z_channels": 13,
    "out_channels": 3,
}


def _normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True)


def _swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = _normalize(out_channels)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.conv1(_swish(self.norm1(x)))
        hidden = self.conv2(self.dropout(_swish(self.norm2(hidden))))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden


class AttnBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = _normalize(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.norm(x)
        query, key, value = self.q(hidden), self.k(hidden), self.v(hidden)
        batch, channels, height, width = query.shape
        query = query.reshape(batch, channels, height * width).permute(0, 2, 1)
        key = key.reshape(batch, channels, height * width)
        weights = torch.bmm(query, key) * (int(channels) ** -0.5)
        weights = F.softmax(weights, dim=2).permute(0, 2, 1)
        value = value.reshape(batch, channels, height * width)
        hidden = torch.bmm(value, weights).reshape(batch, channels, height, width)
        return x + self.proj_out(hidden)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class MidBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels)
        self.attn_1 = AttnBlock(channels)
        self.block_2 = ResnetBlock(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.block_2(self.attn_1(self.block_1(x)))


class Level(nn.Module):
    def __init__(
        self, blocks: nn.ModuleList, attention: nn.ModuleList, sample: nn.Module | None, name: str
    ) -> None:
        super().__init__()
        self.block = blocks
        self.attn = attention
        if sample is not None:
            setattr(self, name, sample)


class VQGANEncoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        ch = config["ch"]
        multipliers = config["encoder_ch_mult"]
        block_counts = config["encoder_num_res_blocks"]
        attention_resolutions = config["attn_resolutions"]
        resolution = config["resolution"]
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        self.down = nn.ModuleList()
        block_in = ch
        current_resolution = resolution
        for level, multiplier in enumerate(multipliers):
            blocks, attention = nn.ModuleList(), nn.ModuleList()
            block_out = ch * multiplier
            for _ in range(block_counts[level]):
                blocks.append(ResnetBlock(block_in, block_out))
                block_in = block_out
                if current_resolution in attention_resolutions:
                    attention.append(AttnBlock(block_in))
            sample = Downsample(block_in) if level != len(multipliers) - 1 else None
            self.down.append(Level(blocks, attention, sample, "downsample"))
            if sample is not None:
                current_resolution //= 2
        self.mid = MidBlock(block_in)
        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, config["z_channels"], 3, padding=1)
        self.quant_conv = nn.Conv2d(config["z_channels"], config["z_channels"], 1)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.conv_in(x)
        for level_index, module in enumerate(self.down):
            level = cast(Level, module)
            for block_index, block in enumerate(level.block):
                hidden = block(hidden)
                if len(level.attn):
                    hidden = level.attn[block_index](hidden)
            if level_index != len(self.down) - 1:
                hidden = cast(Downsample, level.downsample)(hidden)
        hidden = self.mid(hidden)
        return self.quant_conv(self.conv_out(_swish(self.norm_out(hidden))))


class VQGANDecoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        ch = config["ch"]
        multipliers = config["decoder_ch_mult"]
        block_counts = config["decoder_num_res_blocks"]
        attention_resolutions = config["attn_resolutions"]
        current_resolution = config["resolution"] // (2 ** (len(multipliers) - 1))
        block_in = ch * multipliers[-1]
        self.post_quant_conv = nn.Conv2d(config["z_channels"], config["z_channels"], 1)
        self.conv_in = nn.Conv2d(config["z_channels"], block_in, 3, padding=1)
        self.mid = MidBlock(block_in)
        self.up = nn.ModuleList()
        for level in reversed(range(len(multipliers))):
            blocks, attention = nn.ModuleList(), nn.ModuleList()
            block_out = ch * multipliers[level]
            for _ in range(block_counts[level]):
                blocks.append(ResnetBlock(block_in, block_out))
                block_in = block_out
                if current_resolution in attention_resolutions:
                    attention.append(AttnBlock(block_in))
            sample = Upsample(block_in) if level != 0 else None
            self.up.insert(0, Level(blocks, attention, sample, "upsample"))
            if sample is not None:
                current_resolution *= 2
        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, config["out_channels"], 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        hidden = self.conv_in(self.post_quant_conv(z))
        hidden = self.mid(hidden)
        for level_index in reversed(range(len(self.up))):
            level = cast(Level, self.up[level_index])
            for block_index, block in enumerate(level.block):
                hidden = block(hidden)
                if len(level.attn):
                    hidden = level.attn[block_index](hidden)
            if level_index != 0:
                hidden = cast(Upsample, level.upsample)(hidden)
        return self.conv_out(_swish(self.norm_out(hidden)))


class LFQuantizer(nn.Module):
    embedding: Tensor

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        size = 2**dimensions
        indices = torch.arange(size)
        bits = (indices[:, None] >> torch.arange(dimensions - 1, -1, -1)) & 1
        self.register_buffer("embedding", bits.float() * 2 - 1)
        self.register_buffer("power_vals", 2 ** torch.arange(dimensions - 1, -1, -1))

    def get_codebook_entry(self, indices: Tensor, shape: tuple[int, int] | None = None) -> Tensor:
        if shape is None:
            side = math.isqrt(indices.shape[-1])
            shape = (side, side)
        batch = indices.shape[0]
        return (
            self.embedding[indices.reshape(-1)]
            .view(batch, shape[0], shape[1], -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class MAGVITv2(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(MAGVIT_V2_CONFIG | dict(config))
        self.encoder = VQGANEncoder(self.config)
        self.decoder = VQGANDecoder(self.config)
        self.quantize = LFQuantizer(self.config["z_channels"])

    def init_non_persistent_buffers(self) -> None:
        dimensions = self.config["z_channels"]
        indices = torch.arange(2**dimensions)
        bits = (indices[:, None] >> torch.arange(dimensions - 1, -1, -1)) & 1
        self.quantize.embedding = bits.float() * 2 - 1

    def decode_code(self, indices: Tensor, shape: tuple[int, int] | None = None) -> Tensor:
        return self.decoder(self.quantize.get_codebook_entry(indices, shape))
