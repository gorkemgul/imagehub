"""Stable Diffusion 1.x conditional U-Net with diffusers ``UNet2DConditionModel`` weight names."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    "transformer_layers_per_block": 1,
    "reverse_transformer_layers_per_block": None,
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "num_attention_heads": None,
    "use_linear_projection": False,
    "addition_embed_type": None,
    "addition_time_embed_dim": None,
    "projection_class_embeddings_input_dim": None,
    "sample_size": None,
    "norm_num_groups": 32,
    "norm_eps": 1e-5,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "time_cond_proj_dim": None,
}
_UNSUPPORTED_DEFAULTS: dict[str, Any] = {
    "center_input_sample": False,
    "only_cross_attention": False,
    "downsample_padding": 1,
    "mid_block_scale_factor": 1,
    "dropout": 0.0,
    "act_fn": "silu",
    "dual_cross_attention": False,
    "class_embed_type": None,
    "num_class_embeds": None,
    "upcast_attention": False,
    "resnet_time_scale_shift": "default",
    "resnet_skip_time_act": False,
    "resnet_out_scale_factor": 1.0,
    "time_embedding_type": "positional",
    "time_embedding_dim": None,
    "time_embedding_act_fn": None,
    "timestep_post_act": None,
    "conv_in_kernel": 3,
    "conv_out_kernel": 3,
    "encoder_hid_dim": None,
    "encoder_hid_dim_type": None,
    "attention_type": "default",
    "class_embeddings_concat": False,
    "mid_block_only_cross_attention": None,
    "cross_attention_norm": None,
    "addition_embed_type_num_heads": 64,
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
    """The continuous-input path of diffusers ``Transformer2DModel``."""

    def __init__(
        self,
        channels: int,
        context_dim: int,
        heads: int,
        groups: int,
        layers: int,
        linear_projection: bool,
    ) -> None:
        super().__init__()
        self.linear_projection = linear_projection
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
        projection = nn.Linear if linear_projection else lambda x, y: nn.Conv2d(x, y, 1)
        self.proj_in = projection(channels, channels)
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(channels, context_dim, heads) for _ in range(layers)
        )
        self.proj_out = projection(channels, channels)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        if self.linear_projection:
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            x = self.proj_in(x)
        else:
            x = self.proj_in(x)
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        for block in self.transformer_blocks:
            x = block(x, context)
        if self.linear_projection:
            x = self.proj_out(x)
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        else:
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            x = self.proj_out(x)
        return x + residual


def _transformers(
    cfg: Mapping[str, Any],
    channels: int,
    count: int,
    *,
    context_dim: int,
    heads: int,
    layers: int | Sequence[int],
) -> nn.ModuleList:
    depths = [layers] * count if isinstance(layers, int) else [int(v) for v in layers]
    if len(depths) != count:
        raise ValueError(
            "U-Net transformer_layers_per_block must have one depth per attention layer"
        )
    return nn.ModuleList(
        SpatialTransformer(
            channels,
            context_dim,
            heads,
            cfg["norm_num_groups"],
            depth,
            cfg["use_linear_projection"],
        )
        for depth in depths
    )


class DownBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        cfg: Mapping[str, Any],
        temb: int,
        layers: int,
        attn: bool,
        down: bool,
        context_dim: int,
        heads: int,
        transformer_layers: int,
    ) -> None:
        super().__init__()
        eps, groups = cfg["norm_eps"], cfg["norm_num_groups"]
        self.resnets = nn.ModuleList(
            ResnetBlock(
                in_ch if i == 0 else out_ch, out_ch, temb_channels=temb, eps=eps, groups=groups
            )
            for i in range(layers)
        )
        self.attentions = (
            _transformers(
                cfg,
                out_ch,
                layers,
                context_dim=context_dim,
                heads=heads,
                layers=transformer_layers,
            )
            if attn
            else None
        )
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
        layers: int,
        attn: bool,
        up: bool,
        context_dim: int,
        heads: int,
        transformer_layers: int,
    ) -> None:
        super().__init__()
        layers, eps, groups = layers + 1, cfg["norm_eps"], cfg["norm_num_groups"]
        self.resnets = nn.ModuleList()
        for i in range(layers):
            skip_ch = in_ch if i == layers - 1 else out_ch
            res_in = prev_ch if i == 0 else out_ch
            self.resnets.append(
                ResnetBlock(res_in + skip_ch, out_ch, temb_channels=temb, eps=eps, groups=groups)
            )
        self.attentions = (
            _transformers(
                cfg,
                out_ch,
                layers,
                context_dim=context_dim,
                heads=heads,
                layers=transformer_layers,
            )
            if attn
            else None
        )
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
    def __init__(
        self,
        ch: int,
        *,
        cfg: Mapping[str, Any],
        temb: int,
        context_dim: int,
        heads: int,
        transformer_layers: int | Sequence[int],
    ) -> None:
        super().__init__()
        eps, groups = cfg["norm_eps"], cfg["norm_num_groups"]

        def resnet() -> ResnetBlock:
            return ResnetBlock(ch, ch, temb_channels=temb, eps=eps, groups=groups)

        self.attentions: nn.ModuleList | None
        if cfg["mid_block_type"] == "UNetMidBlock2DCrossAttn":
            self.resnets = nn.ModuleList([resnet(), resnet()])
            self.attentions = _transformers(
                cfg,
                ch,
                1,
                context_dim=context_dim,
                heads=heads,
                layers=transformer_layers,
            )
        else:
            # diffusers builds the conditional UNet's UNetMidBlock2D with a single residual
            # block and an empty attention list.
            self.resnets = nn.ModuleList([resnet()])
            self.attentions = None

    def forward(self, x: Tensor, temb: Tensor, context: Tensor) -> Tensor:
        x = self.resnets[0](x, temb)
        if self.attentions is not None:
            x = self.attentions[0](x, context)
        for resnet in self.resnets[1:]:
            x = resnet(x, temb)
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, dim: int, cond_proj_dim: int | None = None) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, dim)
        self.cond_proj = (
            nn.Linear(cond_proj_dim, in_dim, bias=False) if cond_proj_dim is not None else None
        )
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: Tensor, condition: Tensor | None = None) -> Tensor:
        if condition is not None:
            assert self.cond_proj is not None
            x = x + self.cond_proj(condition)
        return self.linear_2(F.silu(self.linear_1(x)))


class UNet2DConditionModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key
            for key in config
            if not key.startswith("_")
            and key not in UNET_DEFAULTS
            and key not in _UNSUPPORTED_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported U-Net setting {unknown[0]}")
        super().__init__(UNET_DEFAULTS | _UNSUPPORTED_DEFAULTS | dict(config))
        cfg = self.config
        ch = cfg["block_out_channels"]
        blocks = len(ch)
        if len(cfg["down_block_types"]) != blocks or len(cfg["up_block_types"]) != blocks:
            raise ValueError("U-Net block type lists must match block_out_channels")
        supported_blocks = {"DownBlock2D", "CrossAttnDownBlock2D"}
        supported_up_blocks = {"UpBlock2D", "CrossAttnUpBlock2D"}
        if (
            not set(cfg["down_block_types"]) <= supported_blocks
            or not set(cfg["up_block_types"]) <= supported_up_blocks
        ):
            raise ValueError("Unsupported U-Net block type")
        for key, default in _UNSUPPORTED_DEFAULTS.items():
            if key == "upcast_attention" and not cfg[key]:
                continue
            if cfg[key] != default:
                raise ValueError(f"Unsupported U-Net setting {key}={cfg[key]!r}")
        if cfg["addition_embed_type"] not in (None, "text_time"):
            raise ValueError(
                f"Unsupported U-Net setting addition_embed_type={cfg['addition_embed_type']!r}"
            )
        if cfg["mid_block_type"] not in ("UNetMidBlock2DCrossAttn", "UNetMidBlock2D"):
            raise ValueError(f"Unsupported U-Net setting mid_block_type={cfg['mid_block_type']!r}")

        def per_block(value: Any, name: str) -> list[Any]:
            values = [value] * blocks if isinstance(value, int) else list(value)
            if len(values) != blocks:
                raise ValueError(f"U-Net {name} must have one value per block")
            return values

        layers = per_block(cfg["layers_per_block"], "layers_per_block")
        transformer_layers = per_block(
            cfg["transformer_layers_per_block"], "transformer_layers_per_block"
        )
        head_config = (
            cfg["num_attention_heads"]
            if cfg["num_attention_heads"] is not None
            else cfg["attention_head_dim"]
        )
        heads = per_block(head_config, "attention heads")
        context_dims = per_block(cfg["cross_attention_dim"], "cross_attention_dim")
        temb = ch[0] * 4
        self.conv_in = nn.Conv2d(cfg["in_channels"], ch[0], 3, padding=1)
        self.time_embedding = TimestepEmbedding(ch[0], temb, cfg["time_cond_proj_dim"])
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
                    layers=layers[i],
                    attn=kind == "CrossAttnDownBlock2D",
                    down=i < len(ch) - 1,
                    context_dim=context_dims[i],
                    heads=heads[i],
                    transformer_layers=transformer_layers[i],
                )
            )
        self.mid_block = MidBlock(
            ch[-1],
            cfg=cfg,
            temb=temb,
            context_dim=context_dims[-1],
            heads=heads[-1],
            transformer_layers=transformer_layers[-1],
        )
        rev = list(reversed(ch))
        rev_layers = list(reversed(layers))
        reverse_config = cfg["reverse_transformer_layers_per_block"]
        if reverse_config is None:
            rev_transformer_layers = list(reversed(transformer_layers))
        else:
            rev_transformer_layers = list(reverse_config)
            if len(rev_transformer_layers) != blocks:
                raise ValueError(
                    "U-Net reverse_transformer_layers_per_block must have one value per block"
                )
        rev_heads = list(reversed(heads))
        rev_context_dims = list(reversed(context_dims))
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
                    layers=rev_layers[i],
                    attn=kind == "CrossAttnUpBlock2D",
                    up=i < len(ch) - 1,
                    context_dim=rev_context_dims[i],
                    heads=rev_heads[i],
                    transformer_layers=rev_transformer_layers[i],
                )
            )
        if cfg["addition_embed_type"] == "text_time":
            if (
                cfg["addition_time_embed_dim"] is None
                or cfg["projection_class_embeddings_input_dim"] is None
            ):
                raise ValueError("text_time conditioning requires both embedding dimensions")
            self.add_time_proj = cfg["addition_time_embed_dim"]
            self.add_embedding = TimestepEmbedding(
                cfg["projection_class_embeddings_input_dim"], temb
            )
        self.conv_norm_out = nn.GroupNorm(cfg["norm_num_groups"], ch[0], eps=cfg["norm_eps"])
        self.conv_out = nn.Conv2d(ch[0], cfg["out_channels"], 3, padding=1)

    def forward(
        self,
        sample: Tensor,
        timestep: Tensor,
        context: Tensor,
        *,
        timestep_cond: Tensor | None = None,
        text_embeds: Tensor | None = None,
        time_ids: Tensor | None = None,
    ) -> Tensor:
        timesteps = timestep.reshape(-1).to(sample.device).expand(sample.shape[0])
        t = timestep_embedding(
            timesteps,
            self.config["block_out_channels"][0],
            flip_sin_to_cos=self.config["flip_sin_to_cos"],
            shift=self.config["freq_shift"],
        ).to(sample.dtype)
        temb = self.time_embedding(t, timestep_cond)
        if self.config["addition_embed_type"] == "text_time":
            if text_embeds is None:
                raise ValueError("text_time conditioning requires text_embeds")
            if time_ids is None:
                raise ValueError("text_time conditioning requires time_ids")
            time_features = timestep_embedding(
                time_ids.flatten(),
                self.add_time_proj,
                flip_sin_to_cos=self.config["flip_sin_to_cos"],
                shift=self.config["freq_shift"],
            ).reshape(text_embeds.shape[0], -1)
            add_embeds = torch.cat([text_embeds, time_features], dim=-1).to(temb.dtype)
            temb = temb + self.add_embedding(add_embeds)
        x = self.conv_in(sample)
        skips = [x]
        for down in self.down_blocks:
            x, block_skips = down(x, temb, context)
            skips.extend(block_skips)
        x = self.mid_block(x, temb, context)
        for up in self.up_blocks:
            assert isinstance(up, UpBlock)
            n = len(up.resnets)
            block_skips, skips = skips[-n:], skips[:-n]
            size = tuple(skips[-1].shape[2:]) if skips else None
            x = up(x, block_skips, temb, context, size)
        return self.conv_out(F.silu(self.conv_norm_out(x)))
