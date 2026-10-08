"""Paella VQ autoencoder used as Stable Cascade stage A."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from imagehub.checkpoints import NativeModel

PAELLA_VQ_DEFAULTS: dict[str, Any] = {
    "in_channels": 3,
    "out_channels": 3,
    "up_down_scale_factor": 2,
    "levels": 2,
    "bottleneck_blocks": 12,
    "embed_dim": 384,
    "latent_channels": 4,
    "num_vq_embeddings": 8192,
    "scale_factor": 0.3764,
}
_EPS = 1e-6


class MixingResidualBlock(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False, eps=_EPS)
        self.depthwise = nn.Sequential(
            nn.ReplicationPad2d(1), nn.Conv2d(channels, channels, kernel_size=3, groups=channels)
        )
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False, eps=_EPS)
        self.channelwise = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )
        self.gammas = nn.Parameter(torch.zeros(6))

    def forward(self, x: Tensor) -> Tensor:
        mods = self.gammas
        hidden = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * (1 + mods[0]) + mods[1]
        x = x + self.depthwise(hidden) * mods[2]
        hidden = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * (1 + mods[3]) + mods[4]
        return x + self.channelwise(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * mods[5]


class VectorQuantizer(nn.Module):
    """Holds the codebook; decoding uses the latents without quantization."""

    def __init__(self, embeddings: int, dimension: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(embeddings, dimension)


class PaellaVQModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in PAELLA_VQ_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported Paella VQ setting {unknown[0]}")
        super().__init__(PAELLA_VQ_DEFAULTS | dict(config))
        cfg = self.config
        levels = cfg["levels"]
        scale = cfg["up_down_scale_factor"]
        latent = cfg["latent_channels"]
        if levels < 1 or cfg["bottleneck_blocks"] < 1 or scale < 1:
            raise ValueError("Unsupported Paella VQ setting: levels, bottleneck_blocks, scale")
        self.scale_factor = float(cfg["scale_factor"])
        channels = [cfg["embed_dim"] // (2**i) for i in reversed(range(levels))]

        self.in_block = nn.Sequential(
            nn.PixelUnshuffle(scale),
            nn.Conv2d(cfg["in_channels"] * scale**2, channels[0], kernel_size=1),
        )
        down: list[nn.Module] = []
        for i in range(levels):
            if i > 0:
                down.append(nn.Conv2d(channels[i - 1], channels[i], 4, stride=2, padding=1))
            down.append(MixingResidualBlock(channels[i], channels[i] * 4))
        down.append(
            nn.Sequential(
                nn.Conv2d(channels[-1], latent, kernel_size=1, bias=False),
                nn.BatchNorm2d(latent),
            )
        )
        self.down_blocks = nn.Sequential(*down)
        self.vquantizer = VectorQuantizer(cfg["num_vq_embeddings"], latent)

        up: list[nn.Module] = [nn.Sequential(nn.Conv2d(latent, channels[-1], kernel_size=1))]
        for i in range(levels):
            width = channels[levels - 1 - i]
            for _ in range(cfg["bottleneck_blocks"] if i == 0 else 1):
                up.append(MixingResidualBlock(width, width * 4))
            if i < levels - 1:
                up.append(
                    nn.ConvTranspose2d(width, channels[levels - 2 - i], 4, stride=2, padding=1)
                )
        self.up_blocks = nn.Sequential(*up)
        self.out_block = nn.Sequential(
            nn.Conv2d(channels[0], cfg["out_channels"] * scale**2, kernel_size=1),
            nn.PixelShuffle(scale),
        )

    def encode(self, images: Tensor) -> Tensor:
        return self.down_blocks(self.in_block(images))

    def decode(self, latents: Tensor) -> Tensor:
        return self.out_block(self.up_blocks(latents))
