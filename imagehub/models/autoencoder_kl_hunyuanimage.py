"""KL autoencoder with the published HunyuanImage 2.1 tensor names (32x compression)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

HUNYUANIMAGE_VAE_DEFAULTS: dict[str, Any] = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 64,
    "block_out_channels": [128, 256, 512, 512, 1024, 1024],
    "layers_per_block": 2,
    "spatial_compression_ratio": 32,
    "sample_size": 384,
    "scaling_factor": None,
    "downsample_match_channel": True,
    "upsample_match_channel": True,
}
_GROUPS = 32
_EPS = 1e-6


class HunyuanImageResnetBlock(nn.Module):
    """Two 3x3 convolutions with an identity residual (the published blocks never change width)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_GROUPS, channels, eps=_EPS)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_GROUPS, channels, eps=_EPS)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + x


class HunyuanImageAttentionBlock(nn.Module):
    """Single-head self-attention over all spatial positions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_GROUPS, channels, eps=_EPS)
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Conv2d(channels, channels, 1)
        self.to_v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.norm(x)
        query, key, value = self.to_q(x), self.to_k(x), self.to_v(x)
        batch_size, channels, height, width = query.shape
        query, key, value = (
            item.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels).contiguous()
            for item in (query, key, value)
        )
        x = F.scaled_dot_product_attention(query, key, value)
        x = x.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return self.proj(x) + identity


class HunyuanImageDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % 4:
            raise ValueError("Downsample output channels must be divisible by 4")
        self.conv = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.group_size = 4 * in_channels // out_channels

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv(x)
        batch, channels, height, width = h.shape
        h = h.reshape(batch, channels, height // 2, 2, width // 2, 2)
        h = h.permute(0, 3, 5, 1, 2, 4).reshape(batch, 4 * channels, height // 2, width // 2)
        batch, channels, height, width = x.shape
        shortcut = x.reshape(batch, channels, height // 2, 2, width // 2, 2)
        shortcut = shortcut.permute(0, 3, 5, 1, 2, 4)
        shortcut = shortcut.reshape(batch, 4 * channels, height // 2, width // 2)
        batch, channels, height, width = shortcut.shape
        shortcut = shortcut.view(batch, h.shape[1], self.group_size, height, width).mean(dim=2)
        return h + shortcut


class HunyuanImageUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1)
        self.repeats = 4 * out_channels // in_channels

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv(x)
        batch, channels, height, width = h.shape
        h = h.reshape(batch, 2, 2, channels // 4, height, width)
        h = h.permute(0, 3, 4, 1, 5, 2).reshape(batch, channels // 4, height * 2, width * 2)
        shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
        batch, channels, height, width = shortcut.shape
        shortcut = shortcut.reshape(batch, 2, 2, channels // 4, height, width)
        shortcut = shortcut.permute(0, 3, 4, 1, 5, 2)
        shortcut = shortcut.reshape(batch, channels // 4, height * 2, width * 2)
        return h + shortcut


class HunyuanImageMidBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [HunyuanImageResnetBlock(channels), HunyuanImageResnetBlock(channels)]
        )
        self.attentions = nn.ModuleList([HunyuanImageAttentionBlock(channels)])

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


class HunyuanImageEncoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        channels = list(config["block_out_channels"])
        z_channels = config["latent_channels"]
        if channels[-1] % (2 * z_channels):
            raise ValueError("The last block width must be divisible by twice the latent channels")
        self.group_size = channels[-1] // (2 * z_channels)
        self.conv_in = nn.Conv2d(config["in_channels"], channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        block_in = channels[0]
        for index, block_out in enumerate(channels):
            for _ in range(config["layers_per_block"]):
                self.down_blocks.append(HunyuanImageResnetBlock(block_in))
            if (
                index < math.log2(config["spatial_compression_ratio"])
                and index != len(channels) - 1
            ):
                block_out = channels[index + 1]
                self.down_blocks.append(HunyuanImageDownsample(block_in, block_out))
                block_in = block_out
        self.mid_block = HunyuanImageMidBlock(channels[-1])
        self.norm_out = nn.GroupNorm(_GROUPS, channels[-1], eps=_EPS)
        self.conv_out = nn.Conv2d(channels[-1], 2 * z_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        batch, channels, height, width = x.shape
        residual = x.view(batch, channels // self.group_size, self.group_size, height, width)
        residual = residual.mean(dim=2)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x + residual


class HunyuanImageDecoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        channels = list(reversed(config["block_out_channels"]))
        z_channels = config["latent_channels"]
        if channels[0] % z_channels:
            raise ValueError("The first decoder width must be divisible by the latent channels")
        self.repeat = channels[0] // z_channels
        self.conv_in = nn.Conv2d(z_channels, channels[0], 3, padding=1)
        self.mid_block = HunyuanImageMidBlock(channels[0])
        self.up_blocks = nn.ModuleList()
        block_in = channels[0]
        for index, block_out in enumerate(channels):
            for _ in range(config["layers_per_block"] + 1):
                self.up_blocks.append(HunyuanImageResnetBlock(block_in))
            if (
                index < math.log2(config["spatial_compression_ratio"])
                and index != len(channels) - 1
            ):
                block_out = channels[index + 1]
                self.up_blocks.append(HunyuanImageUpsample(block_in, block_out))
                block_in = block_out
        self.norm_out = nn.GroupNorm(_GROUPS, channels[-1], eps=_EPS)
        self.conv_out = nn.Conv2d(channels[-1], config["out_channels"], 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x) + x.repeat_interleave(repeats=self.repeat, dim=1)
        h = self.mid_block(h)
        for block in self.up_blocks:
            h = block(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class AutoencoderKLHunyuanImage(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = HUNYUANIMAGE_VAE_DEFAULTS | {
            name: value for name, value in config.items() if not name.startswith("_")
        }
        if not (cfg["downsample_match_channel"] and cfg["upsample_match_channel"]):
            raise ValueError("Unsupported HunyuanImage VAE: channels must match when resampling")
        super().__init__(cfg)
        self.encoder = HunyuanImageEncoder(cfg)
        self.decoder = HunyuanImageDecoder(cfg)

    @property
    def scaling_factor(self) -> float:
        return float(self.config["scaling_factor"])

    def encode(self, image: Tensor) -> Tensor:
        """Return the posterior mean (unscaled latents) for images in [-1, 1]."""
        return self.encoder(image).chunk(2, dim=1)[0]

    def decode(self, latents: Tensor) -> Tensor:
        """Decode unscaled latents to images in roughly [-1, 1]."""
        return self.decoder(latents)
