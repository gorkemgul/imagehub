"""Llama decoder stack used as the Janus-Pro language model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

LLAMA_DEFAULTS: dict[str, Any] = {
    "vocab_size": 32000,
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "head_dim": 128,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "mlp_bias": False,
    "hidden_act": "silu",
    "pad_token_id": None,
    "use_cache": True,
}


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaMLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        head_dim = config["head_dim"]
        base = config["rope_theta"]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .expand(position_ids.shape[0], -1, 1)
            .to(dtype=torch.float, device=x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: Tensor, repetitions: int) -> Tensor:
    batch, key_value_heads, sequence_length, head_dim = hidden_states.shape
    if repetitions == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, key_value_heads, repetitions, sequence_length, head_dim
    )
    return hidden_states.reshape(batch, key_value_heads * repetitions, sequence_length, head_dim)


class LlamaCache:
    """Preallocated KV storage matching the static cache used by Janus generation."""

    def __init__(self, config: Mapping[str, Any], max_cache_len: int) -> None:
        self.max_cache_len = max_cache_len
        self.num_layers = config["num_hidden_layers"]
        self.keys: list[Tensor] = []
        self.values: list[Tensor] = []
        self.sequence_length = 0

    def _initialize(self, key_states: Tensor, value_states: Tensor) -> None:
        key_shape = (*key_states.shape[:2], self.max_cache_len, key_states.shape[-1])
        value_shape = (*value_states.shape[:2], self.max_cache_len, value_states.shape[-1])
        self.keys = [key_states.new_zeros(key_shape) for _ in range(self.num_layers)]
        self.values = [value_states.new_zeros(value_shape) for _ in range(self.num_layers)]

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_index: int,
        cache_position: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self.keys:
            self._initialize(key_states, value_states)
        self.keys[layer_index].index_copy_(2, cache_position, key_states)
        self.values[layer_index].index_copy_(2, cache_position, value_states)
        return self.keys[layer_index], self.values[layer_index]


class LlamaAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any], layer_index: int) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        head_dim = config["head_dim"]
        self.layer_index = layer_index
        self.head_dim = head_dim
        self.num_key_value_groups = config["num_attention_heads"] // config["num_key_value_heads"]
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, config["num_attention_heads"] * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, config["num_key_value_heads"] * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, config["num_key_value_heads"] * head_dim, bias=False)
        self.o_proj = nn.Linear(config["num_attention_heads"] * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        cache: LlamaCache,
        cache_position: Tensor,
    ) -> Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states, value_states = cache.update(
            key_states, value_states, self.layer_index, cache_position
        )
        enable_gqa = self.num_key_value_groups > 1 and attention_mask is None
        if self.num_key_value_groups > 1 and not enable_gqa:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        is_causal = query_states.shape[2] > 1 and attention_mask is None
        if is_causal and key_states.shape[2] > query_states.shape[2]:
            key_states = key_states[:, :, : query_states.shape[2], :]
            value_states = value_states[:, :, : query_states.shape[2], :]
        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scaling,
            enable_gqa=enable_gqa,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention_output)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any], layer_index: int) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_index)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.post_attention_layernorm = LlamaRMSNorm(config["hidden_size"], config["rms_norm_eps"])

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        cache: LlamaCache,
        cache_position: Tensor,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_embeddings, attention_mask, cache, cache_position
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class LlamaModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(LLAMA_DEFAULTS | dict(config))
        unsupported = {
            "activation": self.config["hidden_act"] != "silu",
            "attention bias": self.config["attention_bias"],
            "attention dropout": self.config["attention_dropout"] != 0.0,
            "MLP bias": self.config["mlp_bias"],
            "rope scaling": self.config["rope_scaling"] is not None,
        }
        for option, is_unsupported in unsupported.items():
            if is_unsupported:
                raise ValueError(f"Unsupported Llama {option}")

        self.embed_tokens = nn.Embedding(
            self.config["vocab_size"],
            self.config["hidden_size"],
            self.config["pad_token_id"],
        )
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(self.config, layer_index)
            for layer_index in range(self.config["num_hidden_layers"])
        )
        self.norm = LlamaRMSNorm(self.config["hidden_size"], self.config["rms_norm_eps"])
        self.rotary_emb = LlamaRotaryEmbedding(self.config)

    def init_non_persistent_buffers(self) -> None:
        head_dim = self.config["head_dim"]
        base = self.config["rope_theta"]
        self.rotary_emb.inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )

    def _causal_mask(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        cache: LlamaCache,
        cache_position: Tensor,
    ) -> Tensor | None:
        key_positions = torch.arange(cache.max_cache_len, device=inputs_embeds.device)
        allowed = key_positions[None, :] <= cache_position[:, None]
        padding = torch.zeros(
            (attention_mask.shape[0], cache.max_cache_len),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        padding[:, : attention_mask.shape[1]] = attention_mask.bool()
        query_length = inputs_embeds.shape[1]
        if (query_length == 1 or cache.max_cache_len == query_length) and padding.all():
            return None
        if (
            cache_position[0] == 0
            and padding[:, :query_length].all()
            and (~padding[:, query_length:]).all()
        ):
            return None
        allowed = allowed[None, None] & padding[:, None, None, :]
        return allowed

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        cache: LlamaCache,
        cache_position: Tensor,
    ) -> Tensor:
        """Return hidden states while updating a full-length static KV cache."""
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        causal_mask = self._causal_mask(inputs_embeds, attention_mask, cache, cache_position)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, position_embeddings, causal_mask, cache, cache_position
            )
        cache.sequence_length += inputs_embeds.shape[1]
        return self.norm(hidden_states)
