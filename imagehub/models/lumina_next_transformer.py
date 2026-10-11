"""Native Lumina-Next DiT: text cross-attention with a zero-initialised tanh gate and 2D RoPE."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

LUMINA_NEXT_DEFAULTS: dict[str, Any] = {
    "sample_size": 128,
    "patch_size": 2,
    "in_channels": 4,
    "hidden_size": 2304,
    "num_layers": 32,
    "num_attention_heads": 32,
    "num_kv_heads": None,
    "multiple_of": 256,
    "ffn_dim_multiplier": None,
    "norm_eps": 1e-5,
    "learn_sigma": True,
    "qk_norm": True,
    "cross_attention_dim": 2048,
    "scaling_factor": 1.0,
}
FREQUENCY_EMBEDDING_SIZE = 256
ROPE_THETA = 10000.0
#: Side, in tokens, of the rotary table the pipeline builds before cropping it to the image grid.
ROTARY_TABLE_SIZE = 384


def _timestep_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
    """Sinusoidal embedding with ``flip_sin_to_cos`` and no frequency shift."""
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
    embedding = torch.cat([embedding[:, half_dim:], embedding[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1, 0, 0))
    return embedding


def rotary_frequencies(
    head_dim: int,
    length_h: int = ROTARY_TABLE_SIZE,
    length_w: int = ROTARY_TABLE_SIZE,
    *,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
) -> Tensor:
    """Complex 2D rotary table of shape ``(length_h, length_w, head_dim // 2)``."""
    if head_dim % 4:
        raise ValueError("2d rope needs head dim to be divisible by 4")

    def axis(length: int) -> Tensor:
        dim = head_dim // 2
        theta = ROPE_THETA * ntk_factor
        freqs = (
            1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)) / linear_factor
        )
        freqs = torch.outer(torch.arange(length), freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    emb_h = axis(length_h).view(length_h, 1, head_dim // 4, 1).repeat(1, length_w, 1, 1)
    emb_w = axis(length_w).view(1, length_w, head_dim // 4, 1).repeat(length_h, 1, 1, 1)
    return torch.cat([emb_h, emb_w], dim=-1).flatten(2)


def _apply_rotary(x: Tensor, freqs_cis: Tensor) -> Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_complex * freqs_cis.unsqueeze(2)).flatten(3).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class CombinedTimestepCaptionEmbedding(nn.Module):
    """Timestep embedding plus the mask-averaged caption feature, both of width ``hidden_size``."""

    def __init__(self, hidden_size: int, cross_attention_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(FREQUENCY_EMBEDDING_SIZE, hidden_size)
        self.caption_embedder = nn.Sequential(
            nn.LayerNorm(cross_attention_dim), nn.Linear(cross_attention_dim, hidden_size)
        )

    def forward(self, timestep: Tensor, caption_feat: Tensor, caption_mask: Tensor) -> Tensor:
        time_freq = _timestep_embedding(timestep, FREQUENCY_EMBEDDING_SIZE)
        time_embed = self.timestep_embedder(time_freq.to(dtype=caption_feat.dtype))
        mask = caption_mask.float().unsqueeze(-1)
        pooled = (caption_feat * mask).sum(dim=1) / mask.sum(dim=1)
        pooled = pooled.to(caption_feat)
        return time_embed + self.caption_embedder(pooled)


class FeedForward(nn.Module):
    def __init__(
        self, dim: int, inner_dim: int, multiple_of: int, ffn_dim_multiplier: float | None
    ) -> None:
        super().__init__()
        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)
        self.linear_1 = nn.Linear(dim, inner_dim, bias=False)
        self.linear_2 = nn.Linear(inner_dim, dim, bias=False)
        self.linear_3 = nn.Linear(dim, inner_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.linear_1(x)
        return self.linear_2(F.silu(gate.float()).to(gate.dtype) * self.linear_3(x))


class Attention(nn.Module):
    """Grouped-query attention with query/key LayerNorm across all heads.

    ``to_out`` exists only on the cross-attention, whose projection the block applies to the
    sum of both attentions.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        heads: int,
        kv_heads: int,
        *,
        qk_norm: bool,
        output_projection: bool,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(context_dim, kv_heads * self.head_dim, bias=False)
        self.norm_q: nn.LayerNorm | None = None
        self.norm_k: nn.LayerNorm | None = None
        if qk_norm:
            self.norm_q = nn.LayerNorm(dim, eps=1e-5)
            self.norm_k = nn.LayerNorm(kv_heads * self.head_dim, eps=1e-5)
        if output_projection:
            self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Tensor,
        query_rotary: Tensor,
        key_rotary: Tensor | None,
        base_sequence_length: int | None,
    ) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        dtype = query.dtype
        kv_heads = key.shape[-1] // self.head_dim

        if self.norm_q is not None and self.norm_k is not None:
            query = self.norm_q(query)
            key = self.norm_k(key)

        query = query.view(batch_size, -1, self.heads, self.head_dim)
        key = key.view(batch_size, -1, kv_heads, self.head_dim)
        value = value.view(batch_size, -1, kv_heads, self.head_dim)

        query = _apply_rotary(query, query_rotary)
        if key_rotary is not None:
            key = _apply_rotary(key, key_rotary)
        query, key = query.to(dtype), key.to(dtype)

        # Proportional attention rescales the self-attention logits with the token count.
        softmax_scale: float | None
        if key_rotary is None:
            softmax_scale = None
        elif base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * self.scale
        else:
            softmax_scale = self.scale

        n_rep = self.heads // kv_heads
        key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)
        attention_mask = attention_mask.expand(-1, self.heads, sequence_length, -1)

        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            scale=softmax_scale,
        )
        return hidden_states.transpose(1, 2).to(dtype)


class ModulatedNorm(nn.Module):
    """Adaptive RMS norm producing the attention scale and the MLP scale and gates."""

    def __init__(self, dim: int, norm_eps: float) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(min(dim, 1024), 4 * dim, bias=True)
        self.norm = RMSNorm(dim, norm_eps)

    def forward(self, x: Tensor, emb: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        return self.norm(x) * (1 + scale_msa[:, None]), gate_msa, scale_mlp, gate_mlp


class LuminaNextDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        norm_eps: float,
        qk_norm: bool,
        cross_attention_dim: int,
    ) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros([heads]))
        self.attn1 = Attention(dim, dim, heads, kv_heads, qk_norm=qk_norm, output_projection=False)
        self.attn2 = Attention(
            dim, cross_attention_dim, heads, kv_heads, qk_norm=qk_norm, output_projection=True
        )
        self.feed_forward = FeedForward(dim, int(4 * 2 * dim / 3), multiple_of, ffn_dim_multiplier)
        self.norm1 = ModulatedNorm(dim, norm_eps)
        self.ffn_norm1 = RMSNorm(dim, norm_eps)
        self.norm2 = RMSNorm(dim, norm_eps)
        self.ffn_norm2 = RMSNorm(dim, norm_eps)
        self.norm1_context = RMSNorm(cross_attention_dim, norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        image_rotary_emb: Tensor,
        encoder_hidden_states: Tensor,
        encoder_mask: Tensor,
        temb: Tensor,
        base_sequence_length: int | None,
    ) -> Tensor:
        residual = hidden_states
        norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        self_attn_output = self.attn1(
            norm_hidden_states,
            norm_hidden_states,
            attention_mask,
            image_rotary_emb,
            image_rotary_emb,
            base_sequence_length,
        )
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        cross_attn_output = self.attn2(
            norm_hidden_states,
            norm_encoder_hidden_states,
            encoder_mask,
            image_rotary_emb,
            None,
            base_sequence_length,
        )
        cross_attn_output = cross_attn_output * self.gate.tanh().view(1, 1, -1, 1)
        mixed_attn_output = (self_attn_output + cross_attn_output).flatten(-2)
        hidden_states = self.attn2.to_out[0](mixed_attn_output)

        hidden_states = residual + gate_msa.unsqueeze(1).tanh() * self.norm2(hidden_states)
        mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
        return hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)


class OutputNorm(nn.Module):
    def __init__(self, dim: int, conditioning_dim: int, out_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_dim, dim, bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.linear_2 = nn.Linear(dim, out_dim, bias=True)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        scale = self.linear_1(self.silu(conditioning).to(x.dtype))
        return self.linear_2(self.norm(x) * (1 + scale)[:, None, :])


class PatchEmbedder(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim, bias=True)


class LuminaNextDiT2DModel(NativeModel):
    """Lumina-Next DiT with published tensor names (diffusers ``LuminaNextDiT2DModel``)."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in LUMINA_NEXT_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported Lumina-Next transformer setting {unknown[0]}")
        cfg = LUMINA_NEXT_DEFAULTS | {k: v for k, v in config.items() if not k.startswith("_")}
        hidden, heads = cfg["hidden_size"], cfg["num_attention_heads"]
        kv_heads = cfg["num_kv_heads"] or heads
        if cfg["scaling_factor"] != 1.0:
            raise ValueError(
                "Unsupported Lumina-Next transformer setting "
                f"scaling_factor={cfg['scaling_factor']!r}"
            )
        if hidden % heads or heads % kv_heads:
            raise ValueError("Unsupported Lumina-Next transformer setting num_attention_heads")
        if (hidden // heads) % 4:
            raise ValueError("2d rope needs head dim to be divisible by 4")
        self.patch_size = cfg["patch_size"]
        self.in_channels = cfg["in_channels"]
        self.out_channels = cfg["in_channels"] * 2 if cfg["learn_sigma"] else cfg["in_channels"]
        self.hidden_size = hidden
        self.head_dim = hidden // heads

        self.patch_embedder = PatchEmbedder(cfg["patch_size"], cfg["in_channels"], hidden)
        self.pad_token = nn.Parameter(torch.empty(hidden))
        self.time_caption_embed = CombinedTimestepCaptionEmbedding(
            min(hidden, 1024), cfg["cross_attention_dim"]
        )
        self.layers = nn.ModuleList(
            LuminaNextDiTBlock(
                hidden,
                heads,
                kv_heads,
                cfg["multiple_of"],
                cfg["ffn_dim_multiplier"],
                cfg["norm_eps"],
                cfg["qk_norm"],
                cfg["cross_attention_dim"],
            )
            for _ in range(cfg["num_layers"])
        )
        self.norm_out = OutputNorm(
            hidden, min(hidden, 1024), cfg["patch_size"] ** 2 * self.out_channels
        )

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        encoder_mask: Tensor,
        image_rotary_emb: Tensor,
        *,
        base_sequence_length: int | None = None,
    ) -> Tensor:
        """Predict ``(N, out_channels, H, W)`` from latents ``(N, C, H, W)``.

        ``image_rotary_emb`` is the complex table from :func:`rotary_frequencies`; it is cropped
        to the token grid here.
        """
        patch = self.patch_size
        batch_size, channel, height, width = hidden_states.shape
        height_tokens, width_tokens = height // patch, width // patch
        x = hidden_states.view(batch_size, channel, height_tokens, patch, width_tokens, patch)
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(3)
        x = self.patch_embedder.proj(x).flatten(1, 2)
        mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.int32, device=x.device)
        rotary = (
            image_rotary_emb.to(x.device)[:height_tokens, :width_tokens].flatten(0, 1).unsqueeze(0)
        )

        temb = self.time_caption_embed(timestep, encoder_hidden_states, encoder_mask)

        encoder_mask = encoder_mask.bool()
        for layer in self.layers:
            x = layer(
                x,
                mask,
                rotary,
                encoder_hidden_states,
                encoder_mask,
                temb,
                base_sequence_length,
            )
        x = self.norm_out(x, temb)

        x = x[:, : height_tokens * width_tokens].view(
            batch_size, height_tokens, width_tokens, patch, patch, self.out_channels
        )
        return x.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
