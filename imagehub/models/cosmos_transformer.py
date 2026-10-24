"""Native Cosmos-Predict2 diffusion transformer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

DEFAULTS: dict[str, Any] = {
    "in_channels": 16,
    "out_channels": 16,
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "num_layers": 28,
    "mlp_ratio": 4.0,
    "text_embed_dim": 1024,
    "adaln_lora_dim": 256,
    "max_size": [128, 240, 240],
    "patch_size": [1, 2, 2],
    "rope_scale": [2.0, 1.0, 1.0],
    "concat_padding_mask": True,
    "extra_pos_embed_type": "learnable",
}


class CosmosRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)) * self.weight


def _timestep_embedding(timestep: Tensor, dim: int) -> Tensor:
    half = dim // 2
    exponent = -math.log(10000) * torch.arange(half, device=timestep.device, dtype=torch.float32)
    exponent = exponent / half
    features = timestep[:, None].float() * torch.exp(exponent)[None]
    features = torch.cat((torch.sin(features), torch.cos(features)), dim=-1)
    return torch.cat((features[:, half:], features[:, :half]), dim=-1)


class CosmosPatchEmbed(nn.Module):
    def __init__(self, channels: int, dim: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(channels * math.prod(patch_size), dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, frames, height, width = x.shape
        pt, ph, pw = self.patch_size
        x = x.reshape(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
        return self.proj(x.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7))


class CosmosTimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(self.activation(self.linear_1(x)))


class CosmosEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.t_embedder = CosmosTimestepEmbedding(dim)
        self.norm = CosmosRMSNorm(dim)
        self.dim = dim

    def forward(self, timestep: Tensor) -> tuple[Tensor, Tensor]:
        projected = _timestep_embedding(timestep, self.dim).to(timestep.dtype)
        return self.t_embedder(projected), self.norm(projected)


class CosmosAdaLayerNormZero(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.activation = nn.SiLU()
        self.linear_1 = nn.Linear(dim, hidden, bias=False)
        self.linear_2 = nn.Linear(hidden, 3 * dim, bias=False)

    def forward(self, x: Tensor, embedded: Tensor, temb: Tensor) -> tuple[Tensor, Tensor]:
        modulation = self.linear_2(self.linear_1(self.activation(embedded))) + temb
        shift, scale, gate = modulation.chunk(3, dim=-1)
        if modulation.ndim == 2:
            shift, scale, gate = (value.unsqueeze(1) for value in (shift, scale, gate))
        return self.norm(x) * (1 + scale) + shift, gate


class CosmosAdaLayerNorm(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.dim = dim
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.linear_1 = nn.Linear(dim, hidden, bias=False)
        self.linear_2 = nn.Linear(hidden, 2 * dim, bias=False)

    def forward(self, x: Tensor, embedded: Tensor, temb: Tensor) -> Tensor:
        modulation = (
            self.linear_2(self.linear_1(self.activation(embedded))) + temb[..., : 2 * self.dim]
        )
        shift, scale = modulation.chunk(2, dim=-1)
        if modulation.ndim == 2:
            shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


def _rope(x: Tensor, rotary: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = rotary
    real, imaginary = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
    rotated = torch.cat((-imaginary, real), dim=-1)
    return (x.float() * cos[None, None] + rotated.float() * sin[None, None]).to(x.dtype)


class CosmosAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.norm_q = CosmosRMSNorm(head_dim, eps=1e-5)
        self.norm_k = CosmosRMSNorm(head_dim, eps=1e-5)
        self.to_out = nn.ModuleList([nn.Linear(inner, query_dim, bias=False), nn.Dropout(0.0)])

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        rotary: tuple[Tensor, Tensor] | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        context = x if context is None else context
        query = self.to_q(x).unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        key = self.to_k(context).unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        query = self.norm_q(query)
        key = self.norm_k(key)
        value = self.to_v(context).unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        if rotary is not None:
            query, key = _rope(query, rotary), _rope(key, rotary)
        query, key, value = (item.transpose(1, 2) for item in (query, key, value))
        hidden = F.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
            attn_mask=mask,
            dropout_p=0.0,
        )
        hidden = hidden.permute(0, 2, 1, 3).flatten(2)
        return self.to_out[1](self.to_out[0](hidden))


class CosmosFeedForward(nn.Module):
    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            [
                CosmosGELU(dim, int(dim * ratio)),
                nn.Dropout(0.0),
                nn.Linear(int(dim * ratio), dim, bias=False),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net[2](self.net[1](self.net[0](x)))


class CosmosGELU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.proj(x))


class CosmosTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, heads: int, head_dim: int, text_dim: int, ratio: float, adaln: int
    ) -> None:
        super().__init__()
        self.norm1 = CosmosAdaLayerNormZero(dim, adaln)
        self.attn1 = CosmosAttention(dim, dim, heads, head_dim)
        self.norm2 = CosmosAdaLayerNormZero(dim, adaln)
        self.attn2 = CosmosAttention(dim, text_dim, heads, head_dim)
        self.norm3 = CosmosAdaLayerNormZero(dim, adaln)
        self.ff = CosmosFeedForward(dim, ratio)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        embedded: Tensor,
        temb: Tensor,
        rotary: tuple[Tensor, Tensor],
        mask: Tensor | None,
    ) -> Tensor:
        normed, gate = self.norm1(x, embedded, temb)
        x = x + gate * self.attn1(normed, rotary=rotary)
        normed, gate = self.norm2(x, embedded, temb)
        x = x + gate * self.attn2(normed, context, mask=mask)
        normed, gate = self.norm3(x, embedded, temb)
        return x + gate * self.ff(normed)


class CosmosRotaryPosEmbed(nn.Module):
    def __init__(
        self, head_dim: int, max_size: list[int], patch: tuple[int, int, int], scale: list[float]
    ) -> None:
        super().__init__()
        self.max_size = [size // step for size, step in zip(max_size, patch, strict=True)]
        self.patch = patch
        self.dh = head_dim // 6 * 2
        self.dw = head_dim // 6 * 2
        self.dt = head_dim - self.dh - self.dw
        self.scale = scale

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        sizes = [value // step for value, step in zip(x.shape[2:], self.patch, strict=True)]
        seq = torch.arange(max(self.max_size), device=x.device, dtype=torch.float32)
        embeddings = []
        for size, dim, factor in zip(sizes, (self.dt, self.dh, self.dw), self.scale, strict=True):
            theta = 10000.0 * factor ** (dim / (dim - 2))
            freq = 1 / theta ** (
                torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim
            )
            embeddings.append(torch.outer(seq[:size], freq))
        et, eh, ew = embeddings
        et = et[:, None, None].expand(-1, sizes[1], sizes[2], -1)
        eh = eh[None, :, None].expand(sizes[0], -1, sizes[2], -1)
        ew = ew[None, None, :].expand(sizes[0], sizes[1], -1, -1)
        frequencies = torch.cat([et, eh, ew] * 2, dim=-1).flatten(0, 2)
        return frequencies.cos(), frequencies.sin()


class CosmosTransformer3DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(DEFAULTS | dict(config))
        cfg = self.config
        dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        patch = tuple(cfg["patch_size"])
        channels = cfg["in_channels"] + int(cfg["concat_padding_mask"])
        self.patch_embed = CosmosPatchEmbed(channels, dim, patch)
        self.rope = CosmosRotaryPosEmbed(
            cfg["attention_head_dim"], cfg["max_size"], patch, cfg["rope_scale"]
        )
        self.time_embed = CosmosEmbedding(dim)
        self.transformer_blocks = nn.ModuleList(
            [
                CosmosTransformerBlock(
                    dim,
                    cfg["num_attention_heads"],
                    cfg["attention_head_dim"],
                    cfg["text_embed_dim"],
                    cfg["mlp_ratio"],
                    cfg["adaln_lora_dim"],
                )
                for _ in range(cfg["num_layers"])
            ]
        )
        self.norm_out = CosmosAdaLayerNorm(dim, cfg["adaln_lora_dim"])
        self.proj_out = nn.Linear(dim, math.prod(patch) * cfg["out_channels"], bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch, _, frames, height, width = hidden_states.shape
        if self.config["concat_padding_mask"]:
            if padding_mask is None:
                padding_mask = hidden_states.new_zeros(batch, 1, height, width)
            resized = F.interpolate(padding_mask, (height, width), mode="nearest")
            hidden_states = torch.cat(
                (hidden_states, resized.unsqueeze(2).repeat(1, 1, frames, 1, 1)), dim=1
            )
        rotary = self.rope(hidden_states)
        pt, ph, pw = self.config["patch_size"]
        hidden = self.patch_embed(hidden_states).flatten(1, 3)
        temb, embedded = self.time_embed(timestep)
        mask = attention_mask[:, None, None].bool() if attention_mask is not None else None
        for block in self.transformer_blocks:
            hidden = block(hidden, encoder_hidden_states, embedded, temb, rotary, mask)
        hidden = self.proj_out(self.norm_out(hidden, embedded, temb))
        hidden = hidden.unflatten(2, (ph, pw, pt, -1))
        hidden = hidden.unflatten(1, (frames // pt, height // ph, width // pw))
        return hidden.permute(0, 7, 1, 6, 2, 4, 3, 5).flatten(6, 7).flatten(4, 5).flatten(2, 3)
