"""Native Lumina-Image 2.0 transformer (grouped-query attention, three-axis RoPE, refiners)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

LUMINA2_DEFAULTS: dict[str, Any] = {
    "sample_size": 128,
    "patch_size": 2,
    "in_channels": 16,
    "out_channels": None,
    "hidden_size": 2304,
    "num_layers": 26,
    "num_refiner_layers": 2,
    "num_attention_heads": 24,
    "num_kv_heads": 8,
    "multiple_of": 256,
    "ffn_dim_multiplier": None,
    "norm_eps": 1e-5,
    "scaling_factor": 1.0,
    "axes_dim_rope": [32, 32, 32],
    "axes_lens": [300, 512, 512],
    "cap_feat_dim": 1024,
}
ROPE_THETA = 10000
FREQUENCY_EMBEDDING_SIZE = 256


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
    embedding = torch.cat([embedding[:, half_dim:], embedding[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1, 0, 0))
    return embedding


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
    def __init__(self, hidden_size: int, cap_feat_dim: int, norm_eps: float) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(FREQUENCY_EMBEDDING_SIZE, min(hidden_size, 1024))
        self.caption_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, norm_eps), nn.Linear(cap_feat_dim, hidden_size, bias=True)
        )

    def forward(
        self, hidden_states: Tensor, timestep: Tensor, encoder_hidden_states: Tensor
    ) -> tuple[Tensor, Tensor]:
        timestep_proj = _timestep_embedding(timestep, FREQUENCY_EMBEDDING_SIZE).type_as(
            hidden_states
        )
        return self.timestep_embedder(timestep_proj), self.caption_embedder(encoder_hidden_states)


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
    def __init__(self, dim: int, heads: int, kv_heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.norm_q = RMSNorm(self.head_dim, 1e-5)
        self.norm_k = RMSNorm(self.head_dim, 1e-5)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])

    @staticmethod
    def _rotary(x: Tensor, freqs_cis: Tensor) -> Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(x_complex * freqs_cis.unsqueeze(2)).flatten(3).type_as(x)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None, freqs_cis: Tensor
    ) -> Tensor:
        batch_size = hidden_states.shape[0]
        dtype = hidden_states.dtype
        head_dim = self.head_dim
        query = self.to_q(hidden_states).view(batch_size, -1, self.heads, head_dim)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        kv_heads = key.shape[-1] // head_dim
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)
        query, key = self.norm_q(query), self.norm_k(key)
        query = self._rotary(query, freqs_cis)
        key = self._rotary(key, freqs_cis)
        query, key = query.to(dtype), key.to(dtype)

        n_rep = self.heads // kv_heads
        key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        if attention_mask is not None:
            attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)

        out = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        return self.to_out[0](out.type_as(query))


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


class Lumina2TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        norm_eps: float,
        modulation: bool,
    ) -> None:
        super().__init__()
        self.modulation = modulation
        self.attn = Attention(dim, heads, kv_heads)
        self.feed_forward = FeedForward(dim, 4 * dim, multiple_of, ffn_dim_multiplier)
        self.norm1: nn.Module = (
            ModulatedNorm(dim, norm_eps) if modulation else RMSNorm(dim, norm_eps)
        )
        self.ffn_norm1 = RMSNorm(dim, norm_eps)
        self.norm2 = RMSNorm(dim, norm_eps)
        self.ffn_norm2 = RMSNorm(dim, norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        freqs_cis: Tensor,
        temb: Tensor | None = None,
    ) -> Tensor:
        if self.modulation:
            assert temb is not None
            norm_hidden, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
            attn_output = self.attn(norm_hidden, attention_mask, freqs_cis)
            hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
            mlp_output = self.feed_forward(
                self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1))
            )
            return hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
        attn_output = self.attn(self.norm1(hidden_states), attention_mask, freqs_cis)
        hidden_states = hidden_states + self.norm2(attn_output)
        return hidden_states + self.ffn_norm2(self.feed_forward(self.ffn_norm1(hidden_states)))


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


class RotaryPosEmbed:
    """Complex rotary tables per axis, built in float64 and gathered by position id."""

    def __init__(self, axes_dim: list[int], axes_lens: list[int]) -> None:
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self._tables: list[Tensor] | None = None

    def _build(self, device: torch.device) -> list[Tensor]:
        # Follows the reference: single-precision tables when MPS is available on the host.
        dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        tables = []
        for dim, length in zip(self.axes_dim, self.axes_lens, strict=True):
            pos = torch.arange(length, device=device)
            freqs = 1.0 / (
                ROPE_THETA ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim)
            )
            freqs = torch.outer(pos, freqs)
            tables.append(torch.polar(torch.ones_like(freqs), freqs))
        return tables

    def gather(self, ids: Tensor) -> Tensor:
        if self._tables is None or self._tables[0].device != ids.device:
            self._tables = self._build(ids.device)
        result = []
        for axis, table in enumerate(self._tables):
            index = ids[:, :, axis : axis + 1].repeat(1, 1, table.shape[-1]).to(torch.int64)
            result.append(
                torch.gather(table.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index)
            )
        return torch.cat(result, dim=-1)


class Lumina2Transformer2DModel(NativeModel):
    """Lumina-Image 2.0 transformer with published tensor names."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in LUMINA2_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported Lumina2 transformer setting {unknown[0]}")
        cfg = LUMINA2_DEFAULTS | {k: v for k, v in config.items() if not k.startswith("_")}
        hidden, heads, kv_heads = (
            cfg["hidden_size"],
            cfg["num_attention_heads"],
            cfg["num_kv_heads"],
        )
        axes_dim = list(cfg["axes_dim_rope"])
        axes_lens = list(cfg["axes_lens"])
        if cfg["scaling_factor"] != 1.0:
            raise ValueError(
                f"Unsupported Lumina2 transformer setting scaling_factor={cfg['scaling_factor']!r}"
            )
        if hidden % heads or heads % kv_heads:
            raise ValueError("Unsupported Lumina2 transformer setting num_attention_heads")
        if len(axes_dim) != 3 or len(axes_lens) != 3 or sum(axes_dim) != hidden // heads:
            raise ValueError("Unsupported Lumina2 transformer setting axes_dim_rope")
        self.patch_size = cfg["patch_size"]
        self.hidden_size = hidden
        self.out_channels = cfg["out_channels"] or cfg["in_channels"]
        self.rope_embedder = RotaryPosEmbed(axes_dim, axes_lens)
        self.x_embedder = nn.Linear(cfg["patch_size"] ** 2 * cfg["in_channels"], hidden)
        self.time_caption_embed = CombinedTimestepCaptionEmbedding(
            hidden, cfg["cap_feat_dim"], cfg["norm_eps"]
        )

        def blocks(count: int, modulation: bool) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    Lumina2TransformerBlock(
                        hidden,
                        heads,
                        kv_heads,
                        cfg["multiple_of"],
                        cfg["ffn_dim_multiplier"],
                        cfg["norm_eps"],
                        modulation,
                    )
                    for _ in range(count)
                ]
            )

        self.noise_refiner = blocks(cfg["num_refiner_layers"], True)
        self.context_refiner = blocks(cfg["num_refiner_layers"], False)
        self.layers = blocks(cfg["num_layers"], True)
        self.norm_out = OutputNorm(
            hidden, min(hidden, 1024), cfg["patch_size"] ** 2 * self.out_channels
        )

    def _positions(
        self, hidden_states: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, list[int], list[int]]:
        batch_size, channels, height, width = hidden_states.shape
        p = self.patch_size
        post_h, post_w = height // p, width // p
        image_seq_len = post_h * post_w
        device = hidden_states.device

        encoder_seq_len = attention_mask.shape[1]
        cap_lengths: list[int] = attention_mask.sum(dim=1).tolist()
        seq_lengths = [cap_len + image_seq_len for cap_len in cap_lengths]
        position_ids = torch.zeros(
            batch_size, max(seq_lengths), 3, dtype=torch.int32, device=device
        )
        row_ids = torch.arange(post_h, dtype=torch.int32, device=device).view(-1, 1)
        row_ids = row_ids.repeat(1, post_w).flatten()
        col_ids = torch.arange(post_w, dtype=torch.int32, device=device).view(1, -1)
        col_ids = col_ids.repeat(post_h, 1).flatten()
        for i, (cap_len, seq_len) in enumerate(zip(cap_lengths, seq_lengths, strict=True)):
            position_ids[i, :cap_len, 0] = torch.arange(cap_len, dtype=torch.int32, device=device)
            position_ids[i, cap_len:seq_len, 0] = cap_len
            position_ids[i, cap_len:seq_len, 1] = row_ids
            position_ids[i, cap_len:seq_len, 2] = col_ids

        freqs_cis = self.rope_embedder.gather(position_ids)
        cap_freqs = torch.zeros(
            batch_size, encoder_seq_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )
        img_freqs = torch.zeros(
            batch_size, image_seq_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )
        for i, (cap_len, seq_len) in enumerate(zip(cap_lengths, seq_lengths, strict=True)):
            cap_freqs[i, :cap_len] = freqs_cis[i, :cap_len]
            img_freqs[i, :image_seq_len] = freqs_cis[i, cap_len:seq_len]

        patches = (
            hidden_states.view(batch_size, channels, post_h, p, post_w, p)
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(3)
            .flatten(1, 2)
        )
        return patches, cap_freqs, img_freqs, freqs_cis, cap_lengths, seq_lengths

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
    ) -> Tensor:
        batch_size, _, height, width = hidden_states.shape
        temb, encoder_hidden_states = self.time_caption_embed(
            hidden_states, timestep, encoder_hidden_states
        )
        patches, context_rotary, noise_rotary, rotary, cap_lengths, seq_lengths = self._positions(
            hidden_states, encoder_attention_mask
        )
        hidden_states = self.x_embedder(patches)

        for layer in self.context_refiner:
            encoder_hidden_states = layer(
                encoder_hidden_states, encoder_attention_mask, context_rotary
            )
        for layer in self.noise_refiner:
            hidden_states = layer(hidden_states, None, noise_rotary, temb)

        max_seq_len = max(seq_lengths)
        use_mask = len(set(seq_lengths)) > 1
        attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        joint = hidden_states.new_zeros(batch_size, max_seq_len, self.hidden_size)
        for i, (cap_len, seq_len) in enumerate(zip(cap_lengths, seq_lengths, strict=True)):
            attention_mask[i, :seq_len] = True
            joint[i, :cap_len] = encoder_hidden_states[i, :cap_len]
            joint[i, cap_len:seq_len] = hidden_states[i]
        hidden_states = joint

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask if use_mask else None, rotary, temb)

        hidden_states = self.norm_out(hidden_states, temb)

        p = self.patch_size
        output = []
        for i, (cap_len, seq_len) in enumerate(zip(cap_lengths, seq_lengths, strict=True)):
            output.append(
                hidden_states[i][cap_len:seq_len]
                .view(height // p, width // p, p, p, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )
        return torch.stack(output, dim=0)
