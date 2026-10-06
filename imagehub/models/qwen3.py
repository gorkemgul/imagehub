"""Qwen3 decoder stack used as the Z-Image text encoder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

QWEN3_DEFAULTS: dict[str, Any] = {
    "vocab_size": 151936,
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 40960,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "use_sliding_window": False,
    "sliding_window": None,
}


class Qwen3RMSNorm(nn.Module):
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


class Qwen3MLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = config["head_dim"]
        base = config["rope_theta"]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


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


class Qwen3Attention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        head_dim = config["head_dim"]
        self.head_dim = head_dim
        self.num_key_value_groups = config["num_attention_heads"] // config["num_key_value_heads"]
        self.scaling = head_dim**-0.5
        self.attention_dropout = config["attention_dropout"]
        self.q_proj = nn.Linear(hidden_size, config["num_attention_heads"] * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, config["num_key_value_heads"] * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, config["num_key_value_heads"] * head_dim, bias=False)
        self.o_proj = nn.Linear(config["num_attention_heads"] * head_dim, hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(head_dim, config["rms_norm_eps"])
        self.k_norm = Qwen3RMSNorm(head_dim, config["rms_norm_eps"])

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        sdpa_kwargs: dict[str, bool] = {}
        if self.num_key_value_groups > 1:
            if attention_mask is None:
                sdpa_kwargs = {"enable_gqa": True}
            else:
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
            is_causal=attention_mask is None,
            **sdpa_kwargs,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention_output)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.post_attention_layernorm = Qwen3RMSNorm(config["hidden_size"], config["rms_norm_eps"])

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3Transformer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config) for _ in range(config["num_hidden_layers"])
        )
        self.norm = Qwen3RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(
            0
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask = None
        if not bool(attention_mask.bool().all()):
            sequence_length = hidden_states.shape[1]
            positions = torch.arange(sequence_length, device=hidden_states.device)
            causal_mask = positions[:, None] >= positions[None, :]
            causal_mask = causal_mask[None, None] & attention_mask[:, None, None, :].bool()

        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, causal_mask)
            all_hidden_states.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        all_hidden_states[-1] = hidden_states
        if output_hidden_states:
            return hidden_states, tuple(all_hidden_states)
        return hidden_states


class Qwen3Model(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(QWEN3_DEFAULTS | dict(config))
        unsupported = {
            "activation": self.config["hidden_act"] != "silu",
            "attention bias": self.config["attention_bias"],
            "rope scaling": self.config["rope_scaling"] is not None,
            "sliding window": self.config["use_sliding_window"]
            or self.config["sliding_window"] is not None,
        }
        for option, is_unsupported in unsupported.items():
            if is_unsupported:
                raise ValueError(f"Unsupported Qwen3 {option}")
        self.model = Qwen3Transformer(self.config)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Map Qwen3Model or Qwen3ForCausalLM tensors to the decoder stack."""
        state = {name: value for name, value in state.items() if name != "lm_head.weight"}
        if state and not next(iter(state)).startswith("model."):
            return {"model." + name: value for name, value in state.items()}
        return state

    def init_non_persistent_buffers(self) -> None:
        dim = self.config["head_dim"]
        base = self.config["rope_theta"]
        self.model.rotary_emb.inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        """Return the final hidden state and optionally all captured hidden states."""
        return self.model(input_ids, attention_mask, output_hidden_states=output_hidden_states)
