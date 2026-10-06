"""KL autoencoder with the published FLUX.2 tensor names and latent layout."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.autoencoder_kl import DecoderBlock, EncoderBlock
from imagehub.models.layers import ResnetBlock, attention

FLUX2_VAE_DEFAULTS: dict[str, Any] = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
    "block_out_channels": [128, 256, 512, 512],
    "decoder_block_out_channels": None,
    "layers_per_block": 2,
    "act_fn": "silu",
    "latent_channels": 32,
    "norm_num_groups": 32,
    "sample_size": 1024,
    "force_upcast": True,
    "use_quant_conv": True,
    "use_post_quant_conv": True,
    "mid_block_add_attention": True,
    "batch_norm_eps": 1e-4,
    "batch_norm_momentum": 0.1,
    "patch_size": [2, 2],
}
_EPS = 1e-6


class Flux2VAEAttention(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, eps=_EPS)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels), nn.Dropout(0.0)])

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        residual = x
        hidden = self.group_norm(x).view(batch, channels, height * width).transpose(1, 2)
        hidden = attention(self.to_q(hidden), self.to_k(hidden), self.to_v(hidden), heads=1)
        hidden = self.to_out[1](self.to_out[0](hidden))
        return hidden.transpose(1, 2).reshape(batch, channels, height, width) + residual


class Flux2VAEMidBlock(nn.Module):
    def __init__(self, channels: int, groups: int, add_attention: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(channels, channels, temb_channels=None, eps=_EPS, groups=groups)
            for _ in range(2)
        )
        self.attentions = nn.ModuleList(
            [Flux2VAEAttention(channels, groups)] if add_attention else []
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnets[0](x)
        if self.attentions:
            x = self.attentions[0](x)
        return self.resnets[1](x)


class Flux2Encoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        channels = config["block_out_channels"]
        groups = config["norm_num_groups"]
        layers = config["layers_per_block"]
        self.conv_in = nn.Conv2d(config["in_channels"], channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(
            EncoderBlock(
                channels[max(index - 1, 0)],
                channels[index],
                layers,
                groups,
                down=index < len(channels) - 1,
            )
            for index in range(len(channels))
        )
        self.mid_block = Flux2VAEMidBlock(channels[-1], groups, config["mid_block_add_attention"])
        self.conv_norm_out = nn.GroupNorm(groups, channels[-1], eps=_EPS)
        self.conv_out = nn.Conv2d(channels[-1], 2 * config["latent_channels"], 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class Flux2Decoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        channels = config["block_out_channels"]
        groups = config["norm_num_groups"]
        layers = config["layers_per_block"]
        reversed_channels = list(reversed(channels))
        self.conv_in = nn.Conv2d(config["latent_channels"], reversed_channels[0], 3, padding=1)
        self.mid_block = Flux2VAEMidBlock(
            reversed_channels[0], groups, config["mid_block_add_attention"]
        )
        self.up_blocks = nn.ModuleList(
            DecoderBlock(
                reversed_channels[max(index - 1, 0)],
                reversed_channels[index],
                layers + 1,
                groups,
                up=index < len(channels) - 1,
            )
            for index in range(len(channels))
        )
        self.conv_norm_out = nn.GroupNorm(groups, channels[0], eps=_EPS)
        self.conv_out = nn.Conv2d(channels[0], config["out_channels"], 3, padding=1)

    def forward(self, latents: Tensor) -> Tensor:
        x = self.mid_block(self.conv_in(latents))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


def patchify_latents(latents: Tensor) -> Tensor:
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)


def unpatchify_latents(latents: Tensor) -> Tensor:
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)


def normalize_latents(
    latents: Tensor, running_mean: Tensor, running_var: Tensor, batch_norm_eps: float
) -> Tensor:
    mean = running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    std = torch.sqrt(running_var.view(1, -1, 1, 1) + batch_norm_eps).to(
        latents.device, latents.dtype
    )
    return (latents - mean) / std


def denormalize_latents(
    latents: Tensor, running_mean: Tensor, running_var: Tensor, batch_norm_eps: float
) -> Tensor:
    mean = running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    std = torch.sqrt(running_var.view(1, -1, 1, 1) + batch_norm_eps).to(
        latents.device, latents.dtype
    )
    return latents * std + mean


class AutoencoderKLFlux2(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(FLUX2_VAE_DEFAULTS | dict(config))
        latent_channels = self.config["latent_channels"]
        self.encoder = Flux2Encoder(self.config)
        decoder_config = self.config | {
            "block_out_channels": self.config["decoder_block_out_channels"]
            or self.config["block_out_channels"]
        }
        self.decoder = Flux2Decoder(decoder_config)
        if self.config["use_quant_conv"]:
            self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        if self.config["use_post_quant_conv"]:
            self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.bn = nn.BatchNorm2d(
            math.prod(self.config["patch_size"]) * latent_channels,
            eps=self.config["batch_norm_eps"],
            momentum=self.config["batch_norm_momentum"],
            affine=False,
            track_running_stats=True,
        )

    @property
    def batch_norm_eps(self) -> float:
        return float(self.config["batch_norm_eps"])

    @property
    def patch_size(self) -> tuple[int, int]:
        return tuple(self.config["patch_size"])

    def encode(self, image: Tensor) -> Tensor:
        """Return the posterior mean (unscaled latents) for images in [-1, 1]."""
        moments = self.encoder(image)
        if self.config["use_quant_conv"]:
            moments = self.quant_conv(moments)
        return moments.chunk(2, dim=1)[0]

    def decode(self, latents: Tensor) -> Tensor:
        """Decode unscaled latents to images in roughly [-1, 1]."""
        if self.config["use_post_quant_conv"]:
            latents = self.post_quant_conv(latents)
        return self.decoder(latents)
