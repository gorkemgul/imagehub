"""Vector-quantized autoencoder with MoVQ spatial normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.layers import Downsample, ResnetBlock, Upsample, attention

VQ_DEFAULTS: dict[str, Any] = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ["DownEncoderBlock2D"],
    "up_block_types": ["UpDecoderBlock2D"],
    "block_out_channels": [64],
    "layers_per_block": 1,
    "act_fn": "silu",
    "latent_channels": 3,
    "sample_size": 32,
    "num_vq_embeddings": 256,
    "norm_num_groups": 32,
    "vq_embed_dim": None,
    "scaling_factor": 0.18215,
    "norm_type": "group",
    "mid_block_add_attention": True,
    "lookup_from_codebook": False,
    "force_upcast": False,
}
_EPS = 1e-6


class SpatialNorm(nn.Module):
    def __init__(self, channels: int, latent_channels: int) -> None:
        super().__init__()
        self.norm_layer = nn.GroupNorm(32, channels, eps=_EPS)
        self.conv_y = nn.Conv2d(latent_channels, channels, 1)
        self.conv_b = nn.Conv2d(latent_channels, channels, 1)

    def forward(self, x: Tensor, latents: Tensor) -> Tensor:
        latents = F.interpolate(latents, size=x.shape[-2:], mode="nearest")
        return self.norm_layer(x) * self.conv_y(latents) + self.conv_b(latents)


class VQAttention(nn.Module):
    def __init__(self, channels: int, groups: int, latent_channels: int | None = None) -> None:
        super().__init__()
        self.group_norm: nn.GroupNorm | None
        self.spatial_norm: SpatialNorm | None
        if latent_channels is None:
            self.group_norm = nn.GroupNorm(groups, channels, eps=_EPS)
            self.spatial_norm = None
        else:
            self.group_norm = None
            self.spatial_norm = SpatialNorm(channels, latent_channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels), nn.Identity()])

    def forward(self, x: Tensor, latents: Tensor | None = None) -> Tensor:
        batch, channels, height, width = x.shape
        residual = x
        if self.spatial_norm is not None:
            assert latents is not None
            x = self.spatial_norm(x, latents)
        x = x.view(batch, channels, height * width).transpose(1, 2)
        if self.group_norm is not None:
            x = self.group_norm(x.transpose(1, 2)).transpose(1, 2)
        x = attention(self.to_q(x), self.to_k(x), self.to_v(x), heads=1)
        x = self.to_out[0](x).transpose(1, 2).reshape(batch, channels, height, width)
        return x + residual


class SpatialResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, latent_channels: int) -> None:
        super().__init__()
        self.norm1 = SpatialNorm(in_channels, latent_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = SpatialNorm(out_channels, latent_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x: Tensor, latents: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(x, latents)))
        hidden = self.conv2(F.silu(self.norm2(hidden, latents)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + hidden


class VQMidBlock(nn.Module):
    def __init__(
        self, channels: int, groups: int, add_attention: bool, latent_channels: int | None
    ) -> None:
        super().__init__()
        block = (
            (lambda: ResnetBlock(channels, channels, temb_channels=None, eps=_EPS, groups=groups))
            if latent_channels is None
            else (lambda: SpatialResnetBlock(channels, channels, latent_channels))
        )
        self.resnets = nn.ModuleList([block(), block()])
        self.attentions = nn.ModuleList(
            [VQAttention(channels, groups, latent_channels)] if add_attention else []
        )
        self.spatial = latent_channels is not None

    def forward(self, x: Tensor, latents: Tensor | None = None) -> Tensor:
        x = self.resnets[0](x, latents) if self.spatial else self.resnets[0](x)
        if self.attentions:
            x = self.attentions[0](x, latents)
        return self.resnets[1](x, latents) if self.spatial else self.resnets[1](x)


class EncoderBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, layers: int, groups: int, down: bool, attention: bool
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(
                in_ch if i == 0 else out_ch,
                out_ch,
                temb_channels=None,
                eps=_EPS,
                groups=groups,
            )
            for i in range(layers)
        )
        self.attentions = (
            nn.ModuleList(VQAttention(out_ch, groups) for _ in range(layers)) if attention else None
        )
        self.downsamplers = nn.ModuleList([Downsample(out_ch, padding=0)]) if down else None

    def forward(self, x: Tensor) -> Tensor:
        for i, resnet in enumerate(self.resnets):
            x = resnet(x)
            if self.attentions is not None:
                x = self.attentions[i](x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        channels = cfg["block_out_channels"]
        groups = cfg["norm_num_groups"]
        layers = cfg["layers_per_block"]
        self.conv_in = nn.Conv2d(cfg["in_channels"], channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(
            EncoderBlock(
                channels[max(i - 1, 0)],
                channels[i],
                layers,
                groups,
                down=i < len(channels) - 1,
                attention=cfg["down_block_types"][i] == "AttnDownEncoderBlock2D",
            )
            for i in range(len(channels))
        )
        self.mid_block = VQMidBlock(channels[-1], groups, cfg["mid_block_add_attention"], None)
        self.conv_norm_out = nn.GroupNorm(groups, channels[-1], eps=_EPS)
        self.conv_out = nn.Conv2d(channels[-1], cfg["latent_channels"], 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        layers: int,
        latent_channels: int,
        up: bool,
        attention: bool,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            SpatialResnetBlock(in_ch if i == 0 else out_ch, out_ch, latent_channels)
            for i in range(layers)
        )
        self.attentions = (
            nn.ModuleList(VQAttention(out_ch, 32, latent_channels) for _ in range(layers))
            if attention
            else None
        )
        self.upsamplers = nn.ModuleList([Upsample(out_ch)]) if up else None

    def forward(self, x: Tensor, latents: Tensor) -> Tensor:
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, latents)
            if self.attentions is not None:
                x = self.attentions[i](x, latents)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        channels = cfg["block_out_channels"]
        groups = cfg["norm_num_groups"]
        layers = cfg["layers_per_block"]
        latent = cfg["latent_channels"]
        reverse = list(reversed(channels))
        self.conv_in = nn.Conv2d(latent, reverse[0], 3, padding=1)
        self.mid_block = VQMidBlock(reverse[0], groups, cfg["mid_block_add_attention"], latent)
        self.up_blocks = nn.ModuleList(
            DecoderBlock(
                reverse[max(i - 1, 0)],
                reverse[i],
                layers + 1,
                latent,
                up=i < len(reverse) - 1,
                attention=cfg["up_block_types"][i] == "AttnUpDecoderBlock2D",
            )
            for i in range(len(reverse))
        )
        self.conv_norm_out = SpatialNorm(channels[0], latent)
        self.conv_out = nn.Conv2d(channels[0], cfg["out_channels"], 3, padding=1)

    def forward(self, x: Tensor, latents: Tensor) -> Tensor:
        x = self.mid_block(self.conv_in(x), latents)
        for block in self.up_blocks:
            x = block(x, latents)
        return self.conv_out(F.silu(self.conv_norm_out(x, latents)))


class VectorQuantizer(nn.Module):
    def __init__(self, embeddings: int, dimension: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(embeddings, dimension)
        self.embedding.weight.data.uniform_(-1.0 / embeddings, 1.0 / embeddings)

    def forward(self, latents: Tensor) -> Tensor:
        moved = latents.permute(0, 2, 3, 1).contiguous()
        flat = moved.view(-1, moved.shape[-1])
        indices = torch.argmin(torch.cdist(flat, self.embedding.weight), dim=1)
        quantized = self.embedding(indices).view(moved.shape)
        quantized = moved + (quantized - moved).detach()
        return quantized.permute(0, 3, 1, 2).contiguous()


class VQModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in VQ_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported VQ setting {unknown[0]}")
        super().__init__(VQ_DEFAULTS | dict(config))
        cfg = self.config
        if cfg["norm_type"] != "spatial":
            raise ValueError(f"Unsupported VQ setting norm_type={cfg['norm_type']!r}")
        if cfg["act_fn"] != "silu":
            raise ValueError(f"Unsupported VQ setting act_fn={cfg['act_fn']!r}")
        if not set(cfg["down_block_types"]) <= {
            "DownEncoderBlock2D",
            "AttnDownEncoderBlock2D",
        } or not set(cfg["up_block_types"]) <= {"UpDecoderBlock2D", "AttnUpDecoderBlock2D"}:
            raise ValueError("Unsupported VQ block type")
        dimension = cfg["vq_embed_dim"] or cfg["latent_channels"]
        self.encoder = Encoder(cfg)
        self.quant_conv = nn.Conv2d(cfg["latent_channels"], dimension, 1)
        self.quantize = VectorQuantizer(cfg["num_vq_embeddings"], dimension)
        self.post_quant_conv = nn.Conv2d(dimension, cfg["latent_channels"], 1)
        self.decoder = Decoder(cfg)

    def encode(self, image: Tensor) -> Tensor:
        return self.quant_conv(self.encoder(image))

    def decode(self, latents: Tensor, *, force_not_quantize: bool = False) -> Tensor:
        quantized = latents if force_not_quantize else self.quantize(latents)
        return self.decoder(self.post_quant_conv(quantized), quantized)
