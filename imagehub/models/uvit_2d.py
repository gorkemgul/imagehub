"""Native aMUSEd U-ViT: a convolutional stem, a transformer stack and a masked-token head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.layers import attention, timestep_embedding

UVIT_DEFAULTS: dict[str, Any] = {
    "hidden_size": 1024,
    "use_bias": False,
    "hidden_dropout": 0.0,
    "cond_embed_dim": 768,
    "micro_cond_encode_dim": 256,
    "micro_cond_embed_dim": 1280,
    "encoder_hidden_size": 768,
    "vocab_size": 8256,
    "codebook_size": 8192,
    "in_channels": 768,
    "block_out_channels": 768,
    "num_res_blocks": 3,
    "downsample": False,
    "upsample": False,
    "block_num_heads": 12,
    "num_hidden_layers": 22,
    "num_attention_heads": 16,
    "attention_dropout": 0.0,
    "intermediate_size": 2816,
    "layer_norm_eps": 1e-6,
    "ln_elementwise_affine": True,
    "sample_size": 64,
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight: nn.Parameter | None
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is None:
            return hidden_states.to(input_dtype)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight


class GlobalResponseNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class Attention(nn.Module):
    def __init__(
        self, query_dim: int, heads: int, dim_head: int, cross_dim: int | None, bias: bool
    ) -> None:
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner, bias=bias)
        self.to_k = nn.Linear(cross_dim or query_dim, inner, bias=bias)
        self.to_v = nn.Linear(cross_dim or query_dim, inner, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner, query_dim, bias=bias), nn.Identity()])

    def forward(self, hidden_states: Tensor, context: Tensor | None = None) -> Tensor:
        context = hidden_states if context is None else context
        out = attention(
            self.to_q(hidden_states), self.to_k(context), self.to_v(context), heads=self.heads
        )
        return self.to_out[1](self.to_out[0](out.to(hidden_states.dtype)))


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int, bias: bool) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            [GEGLU(dim, inner_dim, bias), nn.Identity(), nn.Linear(inner_dim, dim, bias=bias)]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class AdaNormContinuous(nn.Module):
    """Normalization whose scale and shift come from a conditioning embedding."""

    def __init__(
        self, dim: int, cond_dim: int, affine: bool, eps: float, bias: bool, kind: str
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(cond_dim, dim * 2, bias=bias)
        self.norm: nn.Module
        if kind == "rms_norm":
            self.norm = RMSNorm(dim, eps, affine)
        else:
            self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=affine, bias=bias)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        emb = self.linear(F.silu(conditioning).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class TransformerBlock(nn.Module):
    def __init__(
        self, dim: int, heads: int, inner_dim: int, eps: float, affine: bool, bias: bool
    ) -> None:
        super().__init__()
        head_dim = dim // heads
        self.norm1 = AdaNormContinuous(dim, dim, affine, eps, bias, "rms_norm")
        self.attn1 = Attention(dim, heads, head_dim, None, bias)
        self.norm2 = AdaNormContinuous(dim, dim, affine, eps, bias, "rms_norm")
        self.attn2 = Attention(dim, heads, head_dim, dim, bias)
        self.norm3 = AdaNormContinuous(dim, dim, affine, eps, bias, "layer_norm")
        self.ff = FeedForward(dim, inner_dim, bias)

    def forward(self, hidden_states: Tensor, context: Tensor, pooled: Tensor) -> Tensor:
        hidden_states = self.attn1(self.norm1(hidden_states, pooled)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states, pooled), context) + hidden_states
        return self.ff(self.norm3(hidden_states, pooled)) + hidden_states


class SkipFFTransformerBlock(nn.Module):
    """Two cross-attentions over the text states without a feed-forward layer."""

    def __init__(self, dim: int, heads: int, kv_dim: int, bias: bool) -> None:
        super().__init__()
        head_dim = dim // heads
        self.kv_mapper = nn.Linear(kv_dim, dim, bias=bias) if kv_dim != dim else None
        self.norm1 = RMSNorm(dim, 1e-6)
        self.attn1 = Attention(dim, heads, head_dim, dim, bias)
        self.norm2 = RMSNorm(dim, 1e-6)
        self.attn2 = Attention(dim, heads, head_dim, dim, bias)

    def forward(self, hidden_states: Tensor, context: Tensor) -> Tensor:
        if self.kv_mapper is not None:
            context = self.kv_mapper(F.silu(context))
        hidden_states = self.attn1(self.norm1(hidden_states), context) + hidden_states
        return self.attn2(self.norm2(hidden_states), context) + hidden_states


class ConvNextBlock(nn.Module):
    def __init__(self, channels: int, eps: float, affine: bool, bias: bool, hidden: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)
        self.norm = RMSNorm(channels, eps, affine)
        self.channelwise_linear_1 = nn.Linear(channels, channels * 4, bias=bias)
        self.channelwise_act = nn.GELU()
        self.channelwise_norm = GlobalResponseNorm(channels * 4)
        self.channelwise_linear_2 = nn.Linear(channels * 4, channels, bias=bias)
        self.cond_embeds_mapper = nn.Linear(hidden, channels * 2, bias=bias)

    def forward(self, x: Tensor, cond_embeds: Tensor) -> Tensor:
        x_res = x
        x = self.depthwise(x).permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.channelwise_linear_1(x)
        x = self.channelwise_act(x)
        x = self.channelwise_norm(x)
        x = self.channelwise_linear_2(x)
        x = x.permute(0, 3, 1, 2) + x_res
        scale, shift = self.cond_embeds_mapper(F.silu(cond_embeds)).chunk(2, dim=1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class Downsample(nn.Module):
    def __init__(self, channels: int, eps: float, affine: bool, bias: bool) -> None:
        super().__init__()
        self.norm = RMSNorm(channels, eps, affine)
        self.conv = nn.Conv2d(channels, channels, 2, stride=2, padding=0, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    def __init__(self, channels: int, eps: float, affine: bool, bias: bool) -> None:
        super().__init__()
        self.norm = RMSNorm(channels, eps, affine)
        self.conv = nn.ConvTranspose2d(channels, channels, 2, stride=2, padding=0, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.conv(x)


class UVitBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_res_blocks: int,
        hidden: int,
        eps: float,
        affine: bool,
        bias: bool,
        heads: int,
        downsample: bool,
        upsample: bool,
    ) -> None:
        super().__init__()
        self.downsample = Downsample(channels, eps, affine, bias) if downsample else None
        self.res_blocks = nn.ModuleList(
            ConvNextBlock(channels, eps, affine, bias, hidden) for _ in range(num_res_blocks)
        )
        self.attention_blocks = nn.ModuleList(
            SkipFFTransformerBlock(channels, heads, hidden, bias) for _ in range(num_res_blocks)
        )
        self.upsample = Upsample(channels, eps, affine, bias) if upsample else None

    def forward(self, x: Tensor, pooled: Tensor, context: Tensor) -> Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        for res_block, attention_block in zip(self.res_blocks, self.attention_blocks, strict=True):
            x = res_block(x, pooled)
            batch, channels, height, width = x.shape
            x = x.view(batch, channels, height * width).permute(0, 2, 1)
            x = attention_block(x, context)
            x = x.permute(0, 2, 1).view(batch, channels, height, width)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class ConvEmbed(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, vocab: int, affine: bool, eps: float, bias: bool
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab, in_channels)
        self.layer_norm = RMSNorm(in_channels, eps, affine)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        embeddings = self.layer_norm(self.embeddings(input_ids)).permute(0, 3, 1, 2)
        return self.conv(embeddings)


class ConvMlmLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        in_channels: int,
        bias: bool,
        affine: bool,
        eps: float,
        codebook_size: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, in_channels, 1, bias=bias)
        self.layer_norm = RMSNorm(in_channels, eps, affine)
        self.conv2 = nn.Conv2d(in_channels, codebook_size, 1, bias=bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.conv2(hidden_states)


class CondEmbed(nn.Module):
    def __init__(self, in_channels: int, dim: int, bias: bool) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, dim, bias=bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim, bias=bias)

    def forward(self, sample: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class UVit2DModel(NativeModel):
    """aMUSEd transformer with published tensor names."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in UVIT_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported UVit2D setting {unknown[0]}")
        super().__init__(UVIT_DEFAULTS | {k: v for k, v in config.items() if not k.startswith("_")})
        cfg = self.config
        hidden = cfg["hidden_size"]
        bias = cfg["use_bias"]
        affine = cfg["ln_elementwise_affine"]
        eps = cfg["layer_norm_eps"]
        channels = cfg["block_out_channels"]
        if hidden % cfg["num_attention_heads"] or channels % cfg["block_num_heads"]:
            raise ValueError("Unsupported UVit2D setting num_attention_heads")
        self.encoder_proj = nn.Linear(cfg["encoder_hidden_size"], hidden, bias=bias)
        self.encoder_proj_layer_norm = RMSNorm(hidden, eps, affine)
        self.embed = ConvEmbed(cfg["in_channels"], channels, cfg["vocab_size"], affine, eps, bias)
        self.cond_embed = CondEmbed(
            cfg["micro_cond_embed_dim"] + cfg["cond_embed_dim"], hidden, bias
        )
        self.down_block = UVitBlock(
            channels,
            cfg["num_res_blocks"],
            hidden,
            eps,
            affine,
            bias,
            cfg["block_num_heads"],
            cfg["downsample"],
            False,
        )
        self.project_to_hidden_norm = RMSNorm(channels, eps, affine)
        self.project_to_hidden = nn.Linear(channels, hidden, bias=bias)
        self.transformer_layers = nn.ModuleList(
            TransformerBlock(
                hidden, cfg["num_attention_heads"], cfg["intermediate_size"], eps, affine, bias
            )
            for _ in range(cfg["num_hidden_layers"])
        )
        self.project_from_hidden_norm = RMSNorm(hidden, eps, affine)
        self.project_from_hidden = nn.Linear(hidden, channels, bias=bias)
        self.up_block = UVitBlock(
            channels,
            cfg["num_res_blocks"],
            hidden,
            eps,
            affine,
            bias,
            cfg["block_num_heads"],
            False,
            cfg["upsample"],
        )
        self.mlm_layer = ConvMlmLayer(
            channels, cfg["in_channels"], bias, affine, eps, cfg["codebook_size"]
        )

    def forward(
        self,
        input_ids: Tensor,
        encoder_hidden_states: Tensor,
        pooled_text_emb: Tensor,
        micro_conds: Tensor,
    ) -> Tensor:
        """Logits (batch, codebook_size, height, width) for token ids (batch, height, width)."""
        encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
        encoder_hidden_states = self.encoder_proj_layer_norm(encoder_hidden_states)
        micro_cond_embeds = timestep_embedding(
            micro_conds.flatten(),
            self.config["micro_cond_encode_dim"],
            flip_sin_to_cos=True,
            shift=0,
        )
        micro_cond_embeds = micro_cond_embeds.reshape((input_ids.shape[0], -1))
        pooled = torch.cat([pooled_text_emb, micro_cond_embeds], dim=1)
        pooled = pooled.to(dtype=self.encoder_proj.weight.dtype)
        pooled = self.cond_embed(pooled).to(encoder_hidden_states.dtype)

        hidden_states = self.embed(input_ids)
        hidden_states = self.down_block(hidden_states, pooled, encoder_hidden_states)
        batch, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        hidden_states = self.project_to_hidden_norm(hidden_states)
        hidden_states = self.project_to_hidden(hidden_states)
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, pooled)
        hidden_states = self.project_from_hidden_norm(hidden_states)
        hidden_states = self.project_from_hidden(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        hidden_states = self.up_block(hidden_states, pooled, encoder_hidden_states)
        return self.mlm_layer(hidden_states)
