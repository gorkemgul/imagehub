"""Qwen3-VL text model used by Qwen-Image 2.1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.qwen3 import Qwen3MLP, Qwen3RMSNorm, repeat_kv, rotate_half

QWEN3_VL_TEXT_DEFAULTS: dict[str, Any] = {
    "vocab_size": 151936,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5_000_000.0,
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    },
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
}


class Qwen3VLTextRotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = config["head_dim"]
        base = config["rope_theta"]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = config["rope_scaling"]["mrope_section"]

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        cos = self._recompose(freqs.cos())
        sin = self._recompose(freqs.sin())
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _recompose(self, frequencies: Tensor) -> Tensor:
        frequencies_thw = frequencies[0]
        for dimension, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dimension] * 3
            indices = slice(offset, length, 3)
            frequencies_thw[..., indices] = frequencies[dimension, ..., indices]
        return torch.cat((frequencies_thw, frequencies_thw), dim=-1)


def apply_multimodal_rotary_pos_emb(
    query: Tensor, key: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query = (query * cos) + (rotate_half(query) * sin)
    key = (key * cos) + (rotate_half(key) * sin)
    return query, key


class Qwen3VLTextAttention(nn.Module):
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
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

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


class Qwen3VLTextDecoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(config)
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


class Qwen3VLTextModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(QWEN3_VL_TEXT_DEFAULTS | dict(config))
        rope_scaling = self.config["rope_scaling"]
        unsupported = {
            "activation": self.config["hidden_act"] != "silu",
            "attention bias": self.config["attention_bias"],
            "rope type": rope_scaling.get("rope_type") != "default",
            "mRoPE layout": not rope_scaling.get("mrope_interleaved"),
        }
        for option, is_unsupported in unsupported.items():
            if is_unsupported:
                raise ValueError(f"Unsupported Qwen3-VL text {option}")

        self.embed_tokens = nn.Embedding(self.config["vocab_size"], self.config["hidden_size"])
        self.layers = nn.ModuleList(
            Qwen3VLTextDecoderLayer(self.config) for _ in range(self.config["num_hidden_layers"])
        )
        self.norm = Qwen3RMSNorm(self.config["hidden_size"], self.config["rms_norm_eps"])
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(self.config)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Strip the published conditional-generation text-model prefix."""
        prefix = "model.language_model."
        if state and all(name.startswith(prefix) for name in state):
            return {name.removeprefix(prefix): value for name, value in state.items()}
        return state

    def init_non_persistent_buffers(self) -> None:
        dim = self.config["head_dim"]
        base = self.config["rope_theta"]
        self.rotary_emb.inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, ...]:
        """Return decoder hidden states, ending with the pre-norm state used by the pipeline."""
        hidden_states = self.embed_tokens(input_ids)
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        position_ids = positions.view(1, 1, -1).expand(3, hidden_states.shape[0], -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask = None
        if not bool(attention_mask.bool().all()):
            causal_mask = positions[:, None] >= positions[None, :]
            causal_mask = causal_mask[None, None] & attention_mask[:, None, None, :].bool()

        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, causal_mask)
            all_hidden_states.append(hidden_states)
        return tuple(all_hidden_states)
