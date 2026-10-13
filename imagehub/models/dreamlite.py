"""DreamLite conditional U-Net with published diffusers tensor names."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.layers import Downsample, Upsample, timestep_embedding
from imagehub.models.unet_2d import TimestepEmbedding


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(dtype) * self.weight


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, expand: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch * expand, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class DreamLiteResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, groups: int, eps: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch, eps=eps)
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch, expand=2)
        self.time_emb_proj = nn.Linear(temb_ch, out_ch * 2)
        self.norm2 = nn.GroupNorm(groups, out_ch * 2, eps=eps)
        self.conv2 = DepthwiseSeparableConv(out_ch * 2, out_ch)
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        residual = x
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden


class DreamLiteAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, heads: int, kv_heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.head_dim * kv_heads, bias=False)
        self.to_v = nn.Linear(context_dim, self.head_dim * kv_heads, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Identity()])
        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

    def forward(
        self, x: Tensor, context: Tensor | None = None, mask: Tensor | None = None
    ) -> Tensor:
        context = x if context is None else context
        batch = x.shape[0]
        query = self.norm_q(self.to_q(x).view(batch, -1, self.heads, self.head_dim))
        key = self.norm_k(self.to_k(context).view(batch, -1, self.kv_heads, self.head_dim))
        value = self.to_v(context).view(batch, -1, self.kv_heads, self.head_dim)
        if self.kv_heads != self.heads:
            repeat = self.heads // self.kv_heads
            key = torch.repeat_interleave(key, repeat, dim=2)
            value = torch.repeat_interleave(value, repeat, dim=2)
        attention_mask = mask[:, None, None, :].bool() if mask is not None else None
        hidden = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
        )
        hidden = hidden.transpose(1, 2).reshape(batch, -1, self.heads * self.head_dim)
        return self.to_out[0](hidden)


class GEGLU(nn.Module):
    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner * 2)

    def forward(self, x: Tensor) -> Tensor:
        hidden, gate = self.proj(x).chunk(2, dim=-1)
        return hidden * F.gelu(gate)


class DreamLiteFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            [GEGLU(dim, dim * mult), nn.Identity(), nn.Linear(dim * mult, dim)]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net[2](self.net[0](x))


class DreamLiteTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, context_dim: int, heads: int, kv_heads: int, mult: int, self_attention: bool
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.norm1 = nn.LayerNorm(dim)
        if self_attention:
            self.attn1 = DreamLiteAttention(dim, dim, heads, kv_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = DreamLiteAttention(dim, context_dim, heads, kv_heads)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = DreamLiteFeedForward(dim, mult)

    def forward(self, x: Tensor, context: Tensor, context_mask: Tensor | None = None) -> Tensor:
        norm = self.norm1(x)
        x = self.attn1(norm) + x if self.self_attention else norm
        x = self.attn2(self.norm2(x), context, context_mask) + x
        return self.ff(self.norm3(x)) + x


class DreamLiteSpatialTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        context_dim: int,
        heads: int,
        groups: int,
        layers: int,
        kv_heads: int,
        mult: int,
        self_attention: bool,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks = nn.ModuleList(
            DreamLiteTransformerBlock(channels, context_dim, heads, kv_heads, mult, self_attention)
            for _ in range(layers)
        )
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: Tensor, context: Tensor, context_mask: Tensor | None = None) -> Tensor:
        batch, channels, height, width = x.shape
        residual = x
        hidden = self.norm(x).permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        hidden = self.proj_in(hidden)
        for block in self.transformer_blocks:
            hidden = block(hidden, context, context_mask)
        hidden = self.proj_out(hidden)
        hidden = hidden.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return hidden + residual


def _depths(value: int | Sequence[int], count: int) -> list[int]:
    result = [value] * count if isinstance(value, int) else list(value)
    if len(result) != count:
        raise ValueError("DreamLite transformer depth must match the number of residual blocks")
    return [int(item) for item in result]


class DreamLiteDownBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        cfg: Mapping[str, Any],
        temb: int,
        layers: int,
        transformer_layers: int | Sequence[int],
        heads: int,
        context_dim: int,
        self_attention: bool,
        down: bool,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            DreamLiteResnetBlock(
                in_ch if index == 0 else out_ch,
                out_ch,
                temb,
                cfg["norm_num_groups"],
                cfg["norm_eps"],
            )
            for index in range(layers)
        )
        depths = _depths(transformer_layers, layers)
        self.attentions = nn.ModuleList(
            DreamLiteSpatialTransformer(
                out_ch,
                context_dim,
                heads,
                cfg["norm_num_groups"],
                depths[index],
                cfg["num_kv_heads"],
                cfg["ff_mult"],
                self_attention,
            )
            for index in range(layers)
        )
        self.downsamplers = nn.ModuleList([Downsample(out_ch, padding=1)]) if down else None

    def forward(
        self, x: Tensor, temb: Tensor, context: Tensor, context_mask: Tensor | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        skips = []
        for resnet, attention in zip(self.resnets, self.attentions, strict=True):
            x = attention(resnet(x, temb), context, context_mask)
            skips.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            skips.append(x)
        return x, skips


class DreamLiteUpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        prev_ch: int,
        out_ch: int,
        *,
        cfg: Mapping[str, Any],
        temb: int,
        layers: int,
        transformer_layers: int | Sequence[int],
        heads: int,
        context_dim: int,
        self_attention: bool | None,
        up: bool,
    ) -> None:
        super().__init__()
        layers += 1
        self.resnets = nn.ModuleList()
        for index in range(layers):
            skip_ch = in_ch if index == layers - 1 else out_ch
            residual_ch = prev_ch if index == 0 else out_ch
            self.resnets.append(
                DreamLiteResnetBlock(
                    residual_ch + skip_ch,
                    out_ch,
                    temb,
                    cfg["norm_num_groups"],
                    cfg["norm_eps"],
                )
            )
        if self_attention is None:
            self.attentions = None
        else:
            depths = _depths(transformer_layers, layers)
            self.attentions = nn.ModuleList(
                DreamLiteSpatialTransformer(
                    out_ch,
                    context_dim,
                    heads,
                    cfg["norm_num_groups"],
                    depths[index],
                    cfg["num_kv_heads"],
                    cfg["ff_mult"],
                    self_attention,
                )
                for index in range(layers)
            )
        self.upsamplers = nn.ModuleList([Upsample(out_ch)]) if up else None

    def forward(
        self,
        x: Tensor,
        skips: list[Tensor],
        temb: Tensor,
        context: Tensor,
        size: tuple[int, int] | None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        for index, resnet in enumerate(self.resnets):
            x = resnet(torch.cat([x, skips.pop()], dim=1), temb)
            if self.attentions is not None:
                x = self.attentions[index](x, context, context_mask)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, size)
        return x


class DreamLiteMidBlock(nn.Module):
    def __init__(self, channels: int, *, cfg: Mapping[str, Any], temb: int, heads: int) -> None:
        super().__init__()
        count = cfg["num_mid_layers"]
        self.resnets = nn.ModuleList(
            DreamLiteResnetBlock(channels, channels, temb, cfg["norm_num_groups"], cfg["norm_eps"])
            for _ in range(count + 1)
        )
        depth = cfg["transformer_layers_per_block"][-1]
        self.attentions = nn.ModuleList(
            DreamLiteSpatialTransformer(
                channels,
                cfg["cross_attention_dim"],
                heads,
                cfg["norm_num_groups"],
                depth,
                cfg["num_kv_heads"],
                cfg["ff_mult"],
                True,
            )
            for _ in range(count)
        )

    def forward(
        self, x: Tensor, temb: Tensor, context: Tensor, context_mask: Tensor | None = None
    ) -> Tensor:
        x = self.resnets[0](x, temb)
        for attention, resnet in zip(self.attentions, self.resnets[1:], strict=True):
            x = resnet(attention(x, context, context_mask), temb)
        return x


_DEFAULTS: dict[str, Any] = {
    "in_channels": 4,
    "out_channels": 4,
    "block_out_channels": [320, 640, 1280],
    "layers_per_block": 2,
    "attention_head_dim": [4, 8, 20],
    "cross_attention_dim": 2048,
    "transformer_layers_per_block": [1, 1, 1],
    "reverse_transformer_layers_per_block": None,
    "norm_num_groups": 32,
    "norm_eps": 1e-5,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "encoder_hid_dim": 2048,
    "encoder_hid_dim_type": "text_proj_rms",
    "addition_embed_type": "time",
    "addition_time_embed_dim": 256,
    "projection_class_embeddings_input_dim": 512,
    "num_kv_heads": 1,
    "ff_mult": 6,
    "num_mid_layers": 1,
}


class DreamLiteUNetModel(NativeModel):
    """Published DreamLite U-Net, restricted to the released configuration family."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(_DEFAULTS | dict(config))
        cfg = self.config
        channels = list(cfg["block_out_channels"])
        blocks = len(channels)

        def per_block(value: Any) -> list[Any]:
            return [value] * blocks if isinstance(value, int) else list(value)

        layers = per_block(cfg["layers_per_block"])
        depths = per_block(cfg["transformer_layers_per_block"])
        heads = per_block(cfg["attention_head_dim"])
        temb = channels[0] * 4
        self.conv_in = nn.Conv2d(cfg["in_channels"], channels[0], 3, padding=1)
        self.time_embedding = TimestepEmbedding(channels[0], temb)
        self.encoder_hid_proj = nn.Sequential(
            nn.Linear(cfg["encoder_hid_dim"], cfg["cross_attention_dim"]),
            RMSNorm(cfg["cross_attention_dim"]),
        )
        self.add_embedding = TimestepEmbedding(cfg["projection_class_embeddings_input_dim"], temb)
        self.down_blocks = nn.ModuleList()
        out_ch = channels[0]
        for index, out_ch in enumerate(channels):
            self.down_blocks.append(
                DreamLiteDownBlock(
                    channels[max(index - 1, 0)],
                    out_ch,
                    cfg=cfg,
                    temb=temb,
                    layers=layers[index],
                    transformer_layers=depths[index],
                    heads=heads[index],
                    context_dim=cfg["cross_attention_dim"],
                    self_attention=index == blocks - 1,
                    down=index < blocks - 1,
                )
            )
        self.mid_block = DreamLiteMidBlock(channels[-1], cfg=cfg, temb=temb, heads=heads[-1])
        reverse_channels = list(reversed(channels))
        reverse_layers = list(reversed(layers))
        reverse_heads = list(reversed(heads))
        reverse_depths = (
            list(reversed(depths))
            if cfg["reverse_transformer_layers_per_block"] is None
            else list(cfg["reverse_transformer_layers_per_block"])
        )
        self.up_blocks = nn.ModuleList()
        previous = reverse_channels[0]
        for index, out_ch in enumerate(reverse_channels):
            input_ch = reverse_channels[min(index + 1, blocks - 1)]
            self_attention: bool | None = True if index == 0 else False if index == 1 else None
            self.up_blocks.append(
                DreamLiteUpBlock(
                    input_ch,
                    previous,
                    out_ch,
                    cfg=cfg,
                    temb=temb,
                    layers=reverse_layers[index],
                    transformer_layers=reverse_depths[index],
                    heads=reverse_heads[index],
                    context_dim=cfg["cross_attention_dim"],
                    self_attention=self_attention,
                    up=index < blocks - 1,
                )
            )
            previous = out_ch
        self.conv_norm_out = nn.GroupNorm(cfg["norm_num_groups"], channels[0], eps=cfg["norm_eps"])
        self.conv_out = nn.Conv2d(channels[0], cfg["out_channels"], 3, padding=1)

    def forward(
        self,
        sample: Tensor,
        timestep: Tensor,
        context: Tensor,
        *,
        time_ids: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        timesteps = timestep.reshape(-1).to(sample.device).expand(sample.shape[0])
        time = timestep_embedding(
            timesteps,
            self.config["block_out_channels"][0],
            flip_sin_to_cos=self.config["flip_sin_to_cos"],
            shift=self.config["freq_shift"],
        ).to(sample.dtype)
        temb = self.time_embedding(time)
        add = timestep_embedding(
            time_ids.flatten(),
            self.config["addition_time_embed_dim"],
            flip_sin_to_cos=self.config["flip_sin_to_cos"],
            shift=self.config["freq_shift"],
        ).reshape(sample.shape[0], -1)
        temb = temb + self.add_embedding(add.to(temb.dtype))
        context = self.encoder_hid_proj(context)
        hidden = self.conv_in(sample)
        skips = [hidden]
        for block in self.down_blocks:
            hidden, block_skips = block(hidden, temb, context, context_mask)
            skips.extend(block_skips)
        hidden = self.mid_block(hidden, temb, context, context_mask)
        for block in self.up_blocks:
            block = cast(DreamLiteUpBlock, block)
            count = len(block.resnets)
            block_skips, skips = skips[-count:], skips[:-count]
            size = tuple(skips[-1].shape[2:]) if skips else None
            hidden = block(hidden, block_skips, temb, context, size, context_mask)
        return self.conv_out(F.silu(self.conv_norm_out(hidden)))
