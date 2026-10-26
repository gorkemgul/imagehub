"""Kandinsky 3 U-Net: BigGAN-deep style bottleneck blocks with conditional group normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.layers import timestep_embedding

KANDINSKY3_UNET_DEFAULTS: dict[str, Any] = {
    "in_channels": 4,
    "time_embedding_dim": 1536,
    "groups": 32,
    "attention_head_dim": 64,
    "layers_per_block": 3,
    "block_out_channels": [384, 768, 1536, 3072],
    "cross_attention_dim": 4096,
    "encoder_hid_dim": 4096,
}
_EXPANSION_RATIO = 4
_COMPRESSION_RATIO = 2
_ADD_CROSS_ATTENTION = (False, True, True, True)
_ADD_SELF_ATTENTION = (False, True, True, True)
# Resolution change of one block of a bottleneck residual unit: None keeps the size, True
# upsamples before the convolution, False downsamples after it.
_Resolutions = list[bool | None]


class Attention(nn.Module):
    """Multi-head attention with bias-free projections; the heads follow from ``out_dim``."""

    def __init__(self, query_dim: int, context_dim: int, head_dim: int, out_dim: int) -> None:
        super().__init__()
        self.heads = out_dim // head_dim
        self.to_q = nn.Linear(query_dim, out_dim, bias=False)
        self.to_k = nn.Linear(context_dim, out_dim, bias=False)
        self.to_v = nn.Linear(context_dim, out_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(out_dim, out_dim, bias=False)])

    def forward(self, hidden_states: Tensor, context: Tensor, mask: Tensor | None) -> Tensor:
        """``mask`` is the additive context mask of shape (batch, 1, context tokens)."""
        batch = hidden_states.shape[0]
        query = self.to_q(hidden_states)
        key = self.to_k(context)
        value = self.to_v(context)
        head_dim = key.shape[-1] // self.heads
        query = query.view(batch, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, self.heads, head_dim).transpose(1, 2)
        attn_mask = None
        if mask is not None:
            attn_mask = mask.to(dtype=hidden_states.dtype)[:, None]
            attn_mask = attn_mask.expand(-1, self.heads, -1, -1).contiguous()
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, -1, self.heads * head_dim)
        return self.to_out[0](hidden_states)


class EncoderProj(nn.Module):
    def __init__(self, encoder_hid_dim: int, cross_attention_dim: int) -> None:
        super().__init__()
        self.projection_linear = nn.Linear(encoder_hid_dim, cross_attention_dim, bias=False)
        self.projection_norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection_norm(self.projection_linear(x))


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: Tensor) -> Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class ConditionalGroupNorm(nn.Module):
    def __init__(self, groups: int, channels: int, context_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, affine=False)
        self.context_mlp = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, 2 * channels))

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        context = self.context_mlp(context)
        for _ in range(x.ndim - 2):
            context = context.unsqueeze(-1)
        scale, shift = context.chunk(2, dim=1)
        return self.norm(x) * (scale + 1.0) + shift


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        kernel_size: int,
        groups: int,
        up_resolution: bool | None,
    ) -> None:
        super().__init__()
        self.group_norm = ConditionalGroupNorm(groups, in_channels, time_embed_dim)
        self.up_sample: nn.Module = (
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            if up_resolution is not None and up_resolution
            else nn.Identity()
        )
        self.projection = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size > 1)
        )
        self.down_sample: nn.Module = (
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
            if up_resolution is not None and not up_resolution
            else nn.Identity()
        )

    def forward(self, x: Tensor, time_embed: Tensor) -> Tensor:
        x = F.silu(self.group_norm(x, time_embed))
        return self.down_sample(self.projection(self.up_sample(x)))


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        groups: int,
        up_resolutions: _Resolutions,
    ) -> None:
        super().__init__()
        hidden = max(in_channels, out_channels) // _COMPRESSION_RATIO
        pairs = [(in_channels, hidden), (hidden, hidden), (hidden, hidden), (hidden, out_channels)]
        self.resnet_blocks = nn.ModuleList(
            Block(a, b, time_embed_dim, kernel, groups, resolution)
            for (a, b), kernel, resolution in zip(pairs, (1, 3, 3, 1), up_resolutions, strict=True)
        )
        self.shortcut_up_sample: nn.Module = (
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            if True in up_resolutions
            else nn.Identity()
        )
        self.shortcut_projection: nn.Module = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.shortcut_down_sample: nn.Module = (
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
            if False in up_resolutions
            else nn.Identity()
        )

    def forward(self, x: Tensor, time_embed: Tensor) -> Tensor:
        out = x
        for block in self.resnet_blocks:
            out = block(out, time_embed)
        x = self.shortcut_down_sample(self.shortcut_projection(self.shortcut_up_sample(x)))
        return x + out


class AttentionPooling(nn.Module):
    def __init__(self, channels: int, context_dim: int, head_dim: int) -> None:
        super().__init__()
        self.attention = Attention(context_dim, context_dim, head_dim, channels)

    def forward(self, x: Tensor, context: Tensor, context_mask: Tensor) -> Tensor:
        pooled = self.attention(context.mean(dim=1, keepdim=True), context, context_mask)
        return x + pooled.squeeze(1)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        context_dim: int | None,
        groups: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.in_norm = ConditionalGroupNorm(groups, channels, time_embed_dim)
        self.attention = Attention(channels, context_dim or channels, head_dim, channels)
        hidden = _EXPANSION_RATIO * channels
        self.out_norm = ConditionalGroupNorm(groups, channels, time_embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(
        self,
        x: Tensor,
        time_embed: Tensor,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        height, width = x.shape[-2:]
        out = self.in_norm(x, time_embed)
        out = out.reshape(x.shape[0], -1, height * width).permute(0, 2, 1)
        out = self.attention(out, context if context is not None else out, context_mask)
        out = out.permute(0, 2, 1).unsqueeze(-1).reshape(out.shape[0], -1, height, width)
        x = x + out
        return x + self.feed_forward(self.out_norm(x, time_embed))


def _attention_or_identity(
    enabled: bool,
    channels: int,
    time_embed_dim: int,
    context_dim: int | None,
    groups: int,
    head_dim: int,
) -> nn.Module:
    if not enabled:
        return nn.Identity()
    return AttentionBlock(channels, time_embed_dim, context_dim, groups, head_dim)


class DownSampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        context_dim: int | None,
        num_blocks: int,
        groups: int,
        head_dim: int,
        down_sample: bool,
        self_attention: bool,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.context_dim = context_dim
        attentions = [
            _attention_or_identity(
                self_attention, in_channels, time_embed_dim, None, groups, head_dim
            )
        ]
        plain: _Resolutions = [None] * 4
        last: _Resolutions = [None, None, False if down_sample else None, None]
        resolutions = [plain] * (num_blocks - 1) + [last]
        pairs = [(in_channels, out_channels)] + [(out_channels, out_channels)] * (num_blocks - 1)
        resnets_in = []
        resnets_out = []
        for (a, b), resolution in zip(pairs, resolutions, strict=True):
            resnets_in.append(ResNetBlock(a, b, time_embed_dim, groups, [None] * 4))
            attentions.append(
                _attention_or_identity(
                    context_dim is not None, b, time_embed_dim, context_dim, groups, head_dim
                )
            )
            resnets_out.append(ResNetBlock(b, b, time_embed_dim, groups, resolution))
        self.attentions = nn.ModuleList(attentions)
        self.resnets_in = nn.ModuleList(resnets_in)
        self.resnets_out = nn.ModuleList(resnets_out)

    def forward(
        self, x: Tensor, time_embed: Tensor, context: Tensor, context_mask: Tensor
    ) -> Tensor:
        if self.self_attention:
            x = self.attentions[0](x, time_embed)
        for attention, resnet_in, resnet_out in zip(
            self.attentions[1:], self.resnets_in, self.resnets_out, strict=True
        ):
            x = resnet_in(x, time_embed)
            if self.context_dim is not None:
                x = attention(x, time_embed, context, context_mask)
            x = resnet_out(x, time_embed)
        return x


class UpSampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cat_dim: int,
        out_channels: int,
        time_embed_dim: int,
        context_dim: int | None,
        num_blocks: int,
        groups: int,
        head_dim: int,
        up_sample: bool,
        self_attention: bool,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.context_dim = context_dim
        resolutions: list[_Resolutions] = [[None, True if up_sample else None, None, None]] + [
            [None] * 4
        ] * (num_blocks - 1)
        pairs = (
            [(in_channels + cat_dim, in_channels)]
            + [(in_channels, in_channels)] * (num_blocks - 2)
            + [(in_channels, out_channels)]
        )
        attentions = [
            _attention_or_identity(
                self_attention, out_channels, time_embed_dim, None, groups, head_dim
            )
        ]
        resnets_in = []
        resnets_out = []
        for (a, b), resolution in zip(pairs, resolutions, strict=True):
            resnets_in.append(ResNetBlock(a, a, time_embed_dim, groups, resolution))
            attentions.append(
                _attention_or_identity(
                    context_dim is not None, a, time_embed_dim, context_dim, groups, head_dim
                )
            )
            resnets_out.append(ResNetBlock(a, b, time_embed_dim, groups, [None] * 4))
        self.attentions = nn.ModuleList(attentions)
        self.resnets_in = nn.ModuleList(resnets_in)
        self.resnets_out = nn.ModuleList(resnets_out)

    def forward(
        self, x: Tensor, time_embed: Tensor, context: Tensor, context_mask: Tensor
    ) -> Tensor:
        for attention, resnet_in, resnet_out in zip(
            self.attentions[1:], self.resnets_in, self.resnets_out, strict=True
        ):
            x = resnet_in(x, time_embed)
            if self.context_dim is not None:
                x = attention(x, time_embed, context, context_mask)
            x = resnet_out(x, time_embed)
        if self.self_attention:
            x = self.attentions[0](x, time_embed)
        return x


class Kandinsky3UNet(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in KANDINSKY3_UNET_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported Kandinsky 3 U-Net setting {unknown[0]}")
        super().__init__(KANDINSKY3_UNET_DEFAULTS | dict(config))
        cfg = self.config
        if type(cfg["layers_per_block"]) is not int or cfg["layers_per_block"] < 2:
            raise ValueError("Unsupported Kandinsky 3 U-Net setting layers_per_block")
        if len(cfg["block_out_channels"]) != len(_ADD_CROSS_ATTENTION):
            raise ValueError("Unsupported Kandinsky 3 U-Net setting block_out_channels")
        in_channels = cfg["in_channels"]
        time_dim = cfg["time_embedding_dim"]
        context_dim = cfg["cross_attention_dim"]
        groups = cfg["groups"]
        head_dim = cfg["attention_head_dim"]
        blocks = list(cfg["block_out_channels"])
        init_channels = blocks[0] // 2
        self.init_channels = init_channels
        self.time_embedding = TimestepEmbedding(init_channels, time_dim)
        self.add_time_condition = AttentionPooling(time_dim, context_dim, head_dim)
        self.conv_in = nn.Conv2d(in_channels, init_channels, kernel_size=3, padding=1)
        self.encoder_hid_proj = EncoderProj(cfg["encoder_hid_dim"], context_dim)

        hidden_dims = [init_channels, *blocks]
        in_out_dims = list(zip(hidden_dims[:-1], hidden_dims[1:], strict=True))
        text_dims = [context_dim if exists else None for exists in _ADD_CROSS_ATTENTION]
        num_blocks = len(blocks) * [cfg["layers_per_block"]]
        self.num_levels = len(in_out_dims)

        cat_dims = []
        self.down_blocks = nn.ModuleList()
        for level, ((in_dim, out_dim), count, text_dim, self_attention) in enumerate(
            zip(in_out_dims, num_blocks, text_dims, _ADD_SELF_ATTENTION, strict=True)
        ):
            last = level == self.num_levels - 1
            cat_dims.append(0 if last else out_dim)
            self.down_blocks.append(
                DownSampleBlock(
                    in_dim,
                    out_dim,
                    time_dim,
                    text_dim,
                    count,
                    groups,
                    head_dim,
                    not last,
                    self_attention,
                )
            )
        self.up_blocks = nn.ModuleList()
        for level, ((out_dim, in_dim), count, text_dim, self_attention) in enumerate(
            zip(
                reversed(in_out_dims),
                reversed(num_blocks),
                reversed(text_dims),
                reversed(_ADD_SELF_ATTENTION),
                strict=True,
            )
        ):
            self.up_blocks.append(
                UpSampleBlock(
                    in_dim,
                    cat_dims.pop(),
                    out_dim,
                    time_dim,
                    text_dim,
                    count,
                    groups,
                    head_dim,
                    level != 0,
                    self_attention,
                )
            )
        self.conv_norm_out = nn.GroupNorm(groups, init_channels)
        self.conv_out = nn.Conv2d(init_channels, in_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: Tensor,
        timestep: Tensor | float | int,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
    ) -> Tensor:
        """``encoder_attention_mask`` is (batch, tokens) with 1 for tokens to attend to."""
        mask = ((1 - encoder_attention_mask.to(sample.dtype)) * -10000.0).unsqueeze(1)
        if not torch.is_tensor(timestep):
            dtype = torch.float32 if isinstance(timestep, float) else torch.int32
            timestep = torch.tensor([timestep], dtype=dtype, device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])
        features = timestep_embedding(
            timestep, self.init_channels, flip_sin_to_cos=False, shift=1
        ).to(sample.dtype)
        time_embed = self.time_embedding(features)

        context = self.encoder_hid_proj(encoder_hidden_states)
        time_embed = self.add_time_condition(time_embed, context, mask)

        skips = []
        sample = self.conv_in(sample)
        for level, block in enumerate(self.down_blocks):
            sample = block(sample, time_embed, context, mask)
            if level != self.num_levels - 1:
                skips.append(sample)
        for level, block in enumerate(self.up_blocks):
            if level != 0:
                sample = torch.cat([sample, skips.pop()], dim=1)
            sample = block(sample, time_embed, context, mask)
        return self.conv_out(F.silu(self.conv_norm_out(sample)))
