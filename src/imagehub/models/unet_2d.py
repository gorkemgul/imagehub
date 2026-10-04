"""Stable Diffusion 1.x conditional U-Net with diffusers ``UNet2DConditionModel`` weight names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.layers import (
    Downsample,
    ResnetBlock,
    Upsample,
    attention,
    timestep_embedding,
)

UNET_DEFAULTS: dict[str, Any] = {
    "in_channels": 4,
    "out_channels": 4,
    "block_out_channels": [320, 640, 1280, 1280],
    "down_block_types": ["CrossAttnDownBlock2D"] * 3 + ["DownBlock2D"],
    "up_block_types": ["UpBlock2D"] + ["CrossAttnUpBlock2D"] * 3,
    "layers_per_block": 2,
    # SD 1.x configs store the number of heads under this historical name.
    "attention_head_dim": 8,
    "cross_attention_dim": 768,
    "norm_num_groups": 32,
    "norm_eps": 1e-5,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
}


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        context = x if context is None else context
        out = attention(self.to_q(x), self.to_k(context), self.to_v(context), self.heads)
        return self.to_out[0](out)


class GEGLU(nn.Module):
    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner * 2)

    def forward(self, x: Tensor) -> Tensor:
        hidden, gate = self.proj(x).chunk(2, dim=-1)
        return hidden * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        # Index 1 is dropout in the reference; the slot keeps ``net.2`` weight names aligned.
        self.net = nn.ModuleList([GEGLU(dim, dim * 4), nn.Identity(), nn.Linear(dim * 4, dim)])

    def forward(self, x: Tensor) -> Tensor:
        return self.net[2](self.net[0](x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, context_dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim, heads)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context) + x
        return self.ff(self.norm3(x)) + x


class SpatialTransformer(nn.Module):
    """diffusers ``Transformer2DModel`` with 1x1 convolution projections (SD 1.x)."""

    def __init__(self, channels: int, context_dim: int, heads: int, groups: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(channels, context_dim, heads)])
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.proj_in(self.norm(x))
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        for block in self.transformer_blocks:
            x = block(x, context)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return self.proj_out(x) + residual


def _transformers(cfg: Mapping[str, Any], channels: int, count: int) -> nn.ModuleList:
    return nn.ModuleList(
        SpatialTransformer(
            channels, cfg["cross_attention_dim"], cfg["attention_head_dim"], cfg["norm_num_groups"]
        )
        for _ in range(count)
    )


class DownBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, *, cfg: Mapping[str, Any], temb: int, attn: bool, down: bool
    ) -> None:
        super().__init__()
        layers, eps, groups = cfg["layers_per_block"], cfg["norm_eps"], cfg["norm_num_groups"]
        self.resnets = nn.ModuleList(
            ResnetBlock(
                in_ch if i == 0 else out_ch, out_ch, temb_channels=temb, eps=eps, groups=groups
            )
            for i in range(layers)
        )
        self.attentions = _transformers(cfg, out_ch, layers) if attn else None
        self.downsamplers = nn.ModuleList([Downsample(out_ch, padding=1)]) if down else None

    def forward(self, x: Tensor, temb: Tensor, context: Tensor) -> tuple[Tensor, list[Tensor]]:
        skips = []
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, temb)
            if self.attentions is not None:
                x = self.attentions[i](x, context)
            skips.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            skips.append(x)
        return x, skips


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        prev_ch: int,
        out_ch: int,
        *,
        cfg: Mapping[str, Any],
        temb: int,
        attn: bool,
        up: bool,
    ) -> None:
        super().__init__()
        layers, eps, groups = cfg["layers_per_block"] + 1, cfg["norm_eps"], cfg["norm_num_groups"]
        self.resnets = nn.ModuleList()
        for i in range(layers):
            skip_ch = in_ch if i == layers - 1 else out_ch
            res_in = prev_ch if i == 0 else out_ch
            self.resnets.append(
                ResnetBlock(res_in + skip_ch, out_ch, temb_channels=temb, eps=eps, groups=groups)
            )
        self.attentions = _transformers(cfg, out_ch, layers) if attn else None
        self.upsamplers = nn.ModuleList([Upsample(out_ch)]) if up else None

    def forward(
        self,
        x: Tensor,
        skips: list[Tensor],
        temb: Tensor,
        context: Tensor,
        size: tuple[int, int] | None,
    ) -> Tensor:
        for i, resnet in enumerate(self.resnets):
            x = resnet(torch.cat([x, skips.pop()], dim=1), temb)
            if self.attentions is not None:
                x = self.attentions[i](x, context)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, size)
        return x


class MidBlock(nn.Module):
    def __init__(self, ch: int, *, cfg: Mapping[str, Any], temb: int) -> None:
        super().__init__()
        eps, groups = cfg["norm_eps"], cfg["norm_num_groups"]
        self.resnets = nn.ModuleList(
            ResnetBlock(ch, ch, temb_channels=temb, eps=eps, groups=groups) for _ in range(2)
        )
        self.attentions = _transformers(cfg, ch, 1)

    def forward(self, x: Tensor, temb: Tensor, context: Tensor) -> Tensor:
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        return self.resnets[1](x, temb)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


class UNet2DConditionModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(UNET_DEFAULTS | dict(config))
        cfg = self.config
        ch = cfg["block_out_channels"]
        temb = ch[0] * 4
        self.conv_in = nn.Conv2d(cfg["in_channels"], ch[0], 3, padding=1)
        self.time_embedding = TimestepEmbedding(ch[0], temb)
        self.down_blocks = nn.ModuleList()
        out_ch = ch[0]
        for i, kind in enumerate(cfg["down_block_types"]):
            in_ch, out_ch = out_ch, ch[i]
            self.down_blocks.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    cfg=cfg,
                    temb=temb,
                    attn=kind == "CrossAttnDownBlock2D",
                    down=i < len(ch) - 1,
                )
            )
        self.mid_block = MidBlock(ch[-1], cfg=cfg, temb=temb)
        rev = list(reversed(ch))
        self.up_blocks = nn.ModuleList()
        out_ch = rev[0]
        for i, kind in enumerate(cfg["up_block_types"]):
            prev_ch, out_ch = out_ch, rev[i]
            in_ch = rev[min(i + 1, len(ch) - 1)]
            self.up_blocks.append(
                UpBlock(
                    in_ch,
                    prev_ch,
                    out_ch,
                    cfg=cfg,
                    temb=temb,
                    attn=kind == "CrossAttnUpBlock2D",
                    up=i < len(ch) - 1,
                )
            )
        self.conv_norm_out = nn.GroupNorm(cfg["norm_num_groups"], ch[0], eps=cfg["norm_eps"])
        self.conv_out = nn.Conv2d(ch[0], cfg["out_channels"], 3, padding=1)

    def forward(self, sample: Tensor, timestep: Tensor, context: Tensor) -> Tensor:
        timesteps = timestep.reshape(-1).to(sample.device).expand(sample.shape[0])
        t = timestep_embedding(
            timesteps,
            self.config["block_out_channels"][0],
            flip_sin_to_cos=self.config["flip_sin_to_cos"],
            shift=self.config["freq_shift"],
        ).to(sample.dtype)
        temb = self.time_embedding(t)
        x = self.conv_in(sample)
        skips = [x]
        for down in self.down_blocks:
            x, block_skips = down(x, temb, context)
            skips.extend(block_skips)
        x = self.mid_block(x, temb, context)
        n = self.config["layers_per_block"] + 1
        for up in self.up_blocks:
            block_skips, skips = skips[-n:], skips[:-n]
            size = tuple(skips[-1].shape[2:]) if skips else None
            x = up(x, block_skips, temb, context, size)
        return self.conv_out(F.silu(self.conv_norm_out(x)))
