"""Emu3 causal language model with the published tensor layout."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

KVCache = list[tuple[Tensor, Tensor]]


def _rotate_half(value: Tensor) -> Tensor:
    left, right = value.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class Emu3RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = eps

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * normalized.to(dtype)


class Emu3MLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden, intermediate = config["hidden_size"], config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class Emu3Attention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = hidden // self.heads
        self.groups = self.heads // self.kv_heads
        self.dropout = float(config.get("attention_dropout", 0.0))
        self.rope_theta = float(config["rope_theta"])
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self, value: Tensor, positions: Tensor, past: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, length, _ = value.shape
        query = self.q_proj(value).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(value).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        val = self.v_proj(value).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        inv = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, device=value.device).float() / self.head_dim)
        )
        freqs = torch.einsum("bi,j->bij", positions.float(), inv)
        emb = torch.cat((freqs, freqs), -1)[:, None]
        cos, sin = emb.cos().to(query.dtype), emb.sin().to(query.dtype)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        if past is not None:
            key = torch.cat((past[0], key), dim=2)
            val = torch.cat((past[1], val), dim=2)
        present = (key, val)
        key = key.repeat_interleave(self.groups, dim=1)
        val = val.repeat_interleave(self.groups, dim=1)
        causal = None
        if length > 1:
            total = key.shape[2]
            causal = torch.arange(total, device=value.device)[None, :] <= positions[:, :, None]
            causal = causal[:, None]
        output = F.scaled_dot_product_attention(
            query,
            key,
            val,
            attn_mask=causal,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.o_proj(output.transpose(1, 2).contiguous().view(batch, length, -1)), present


class Emu3DecoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.self_attn = Emu3Attention(config)
        self.mlp = Emu3MLP(config)
        self.input_layernorm = Emu3RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.post_attention_layernorm = Emu3RMSNorm(config["hidden_size"], config["rms_norm_eps"])

    def forward(
        self, value: Tensor, positions: Tensor, past: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attention, present = self.self_attn(self.input_layernorm(value), positions, past)
        value = value + attention
        return value + self.mlp(self.post_attention_layernorm(value)), present


class Emu3Transformer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            Emu3DecoderLayer(config) for _ in range(config["num_hidden_layers"])
        )
        self.norm = Emu3RMSNorm(config["hidden_size"], config["rms_norm_eps"])

    def forward(self, ids: Tensor, past: KVCache | None) -> tuple[Tensor, KVCache]:
        past_length = 0 if past is None else past[0][0].shape[2]
        positions = torch.arange(past_length, past_length + ids.shape[1], device=ids.device).expand(
            ids.shape[0], -1
        )
        value = self.embed_tokens(ids)
        present: KVCache = []
        for index, layer in enumerate(self.layers):
            value, layer_present = layer(value, positions, None if past is None else past[index])
            present.append(layer_present)
        return self.norm(value), present


class Emu3ForCausalLM(NativeModel):
    """Decoder-only Emu3 generator; ``forward`` returns float32 logits and a KV cache."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        if config.get("hidden_act", "silu") != "silu" or config.get("rope_scaling") is not None:
            raise ValueError("Unsupported Emu3 configuration")
        self.model = Emu3Transformer(self.config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(
        self, input_ids: Tensor, past_key_values: KVCache | None = None
    ) -> tuple[Tensor, KVCache]:
        hidden, present = self.model(input_ids, past_key_values)
        return self.lm_head(hidden).float(), present
