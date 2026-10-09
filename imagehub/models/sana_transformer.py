"""Native Sana transformer (linear attention, GLUMBConv, masked cross-attention)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

SANA_DEFAULTS: dict[str, Any] = {
    "in_channels": 32,
    "out_channels": 32,
    "num_attention_heads": 70,
    "attention_head_dim": 32,
    "num_layers": 20,
    "num_cross_attention_heads": 20,
    "cross_attention_head_dim": 112,
    "cross_attention_dim": 2240,
    "caption_channels": 2304,
    "mlp_ratio": 2.5,
    "dropout": 0.0,
    "attention_bias": False,
    "sample_size": 32,
    "patch_size": 1,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
    "interpolation_scale": None,
    "guidance_embeds": False,
    "guidance_embeds_scale": 0.1,
    "qk_norm": None,
    "timestep_scale": 1.0,
}
# Options without an effect on the inference graph for the supported settings.
_IGNORED = frozenset({"guidance_embeds_scale", "timestep_scale", "sample_size", "dropout"})
_REQUIRED = {"interpolation_scale": None, "guidance_embeds": False, "qk_norm": None}


def _timestep_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / half_dim
    embedding = torch.exp(exponent)
    embedding = timesteps[:, None].float() * embedding[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
    return torch.cat([embedding[:, half_dim:], embedding[:, :half_dim]], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, bias: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        hidden_states = hidden_states * self.weight
        if self.bias is not None:
            hidden_states = hidden_states + self.bias
        return hidden_states


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class CombinedTimestepEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)

    def forward(self, timestep: Tensor, hidden_dtype: torch.dtype) -> Tensor:
        projected = _timestep_embedding(timestep, 256)
        return self.timestep_embedder(projected.to(dtype=hidden_dtype))


class AdaLayerNormSingle(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.emb = CombinedTimestepEmbeddings(embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)

    def forward(self, timestep: Tensor, hidden_dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class TextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, caption: Tensor) -> Tensor:
        return self.linear_2(self.act_1(self.linear_1(caption)))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=True,
        )

    def forward(self, latent: Tensor) -> Tensor:
        latent = self.proj(latent)
        return latent.flatten(2).transpose(1, 2)


class GLUMBConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: float) -> None:
        super().__init__()
        hidden_channels = int(expand_ratio * in_channels)
        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
        self.conv_depth = nn.Conv2d(
            hidden_channels * 2, hidden_channels * 2, 3, 1, 1, groups=hidden_channels * 2
        )
        self.conv_point = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.nonlinearity(self.conv_inverted(hidden_states))
        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)
        return self.conv_point(hidden_states)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        qkv_bias: bool,
        out_bias: bool,
        cross_attention_dim: int | None,
        dropout: float,
    ) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        context_dim = query_dim if cross_attention_dim is None else cross_attention_dim
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, query_dim, bias=out_bias), nn.Dropout(dropout)]
        )


class LinearSelfAttention(Attention):
    def forward(self, hidden_states: Tensor) -> Tensor:
        original_dtype = hidden_states.dtype
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.transpose(1, 2).unflatten(1, (self.heads, -1))
        key = key.transpose(1, 2).unflatten(1, (self.heads, -1)).transpose(2, 3)
        value = value.transpose(1, 2).unflatten(1, (self.heads, -1))

        query = F.relu(query)
        key = F.relu(key)

        query, key, value = query.float(), key.float(), value.float()

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
        scores = torch.matmul(value, key)
        hidden_states = torch.matmul(scores, query)

        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + 1e-15)
        hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        if original_dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


class CrossAttention(Attention):
    def forward(
        self, hidden_states: Tensor, encoder_hidden_states: Tensor, attention_mask: Tensor | None
    ) -> Tensor:
        batch_size = encoder_hidden_states.shape[0]
        if attention_mask is not None:
            # The bias has shape (batch, 1, key_tokens); expand it per head as the reference does.
            attention_mask = attention_mask.repeat_interleave(
                self.heads, dim=0, output_size=attention_mask.shape[0] * self.heads
            )
            attention_mask = attention_mask.view(
                batch_size, self.heads, -1, attention_mask.shape[-1]
            )
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        head_dim = key.shape[-1] // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)


class SanaTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float,
        num_cross_attention_heads: int,
        cross_attention_head_dim: int,
        cross_attention_dim: int,
        attention_bias: bool,
        norm_elementwise_affine: bool,
        norm_eps: float,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = LinearSelfAttention(
            dim, num_attention_heads, attention_head_dim, attention_bias, True, None, dropout
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn2 = CrossAttention(
            dim,
            num_cross_attention_heads,
            cross_attention_head_dim,
            True,
            True,
            cross_attention_dim,
            dropout,
        )
        self.ff = GLUMBConv(dim, dim, mlp_ratio)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None,
        timestep: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        batch_size = hidden_states.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)
        attn_output = self.attn1(norm_hidden_states)
        hidden_states = hidden_states + gate_msa * attn_output

        attn_output = self.attn2(hidden_states, encoder_hidden_states, encoder_attention_mask)
        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
        ff_output = self.ff(norm_hidden_states)
        ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
        return hidden_states + gate_mlp * ff_output


class ModulatedNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: Tensor, temb: Tensor, scale_shift_table: Tensor) -> Tensor:
        hidden_states = self.norm(hidden_states)
        shift, scale = (scale_shift_table[None] + temb[:, None]).chunk(2, dim=1)
        return hidden_states * (1 + scale) + shift


class SanaTransformer2DModel(NativeModel):
    """Sana transformer with published tensor names."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in SANA_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported Sana transformer setting {unknown[0]}")
        cfg = SANA_DEFAULTS | {k: v for k, v in config.items() if not k.startswith("_")}
        for key, expected in _REQUIRED.items():
            if cfg[key] != expected:
                raise ValueError(f"Unsupported Sana transformer setting {key}={cfg[key]!r}")
        if cfg["cross_attention_dim"] is None or cfg["num_cross_attention_heads"] is None:
            raise ValueError("Unsupported Sana transformer setting cross_attention_dim=None")
        out_channels = cfg["out_channels"] or cfg["in_channels"]
        inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        self.patch_size = cfg["patch_size"]
        self.patch_embed = PatchEmbed(cfg["patch_size"], cfg["in_channels"], inner_dim)
        self.time_embed = AdaLayerNormSingle(inner_dim)
        self.caption_projection = TextProjection(cfg["caption_channels"], inner_dim)
        self.caption_norm = RMSNorm(inner_dim, eps=1e-5)
        self.transformer_blocks = nn.ModuleList(
            [
                SanaTransformerBlock(
                    inner_dim,
                    cfg["num_attention_heads"],
                    cfg["attention_head_dim"],
                    cfg["dropout"],
                    cfg["num_cross_attention_heads"],
                    cfg["cross_attention_head_dim"],
                    cfg["cross_attention_dim"],
                    cfg["attention_bias"],
                    cfg["norm_elementwise_affine"],
                    cfg["norm_eps"],
                    cfg["mlp_ratio"],
                )
                for _ in range(cfg["num_layers"])
            ]
        )
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.norm_out = ModulatedNorm(inner_dim)
        self.proj_out = nn.Linear(inner_dim, cfg["patch_size"] ** 2 * out_channels)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        encoder_attention_mask: Tensor | None = None,
    ) -> Tensor:
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            # (1 = keep, 0 = discard) becomes an additive bias with a singleton query dimension.
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size, _, height, width = hidden_states.shape
        p = self.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        hidden_states = self.patch_embed(hidden_states)
        timestep_emb, embedded_timestep = self.time_embed(timestep, hidden_states.dtype)

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep_emb,
                post_patch_height,
                post_patch_width,
            )

        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width, p, p, -1
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        return hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)
