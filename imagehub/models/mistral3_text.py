"""Ministral 3 decoder stack: the language model of Mistral3, the ERNIE-Image text encoder.

The published ``text_encoder`` snapshot is a full ``Mistral3Model`` (vision tower, projector and
language model). Only the language model takes part in text encoding, so the vision tensors are
dropped on loading.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.qwen3 import apply_rotary_pos_emb, repeat_kv

MINISTRAL3_DEFAULTS: dict[str, Any] = {
    "vocab_size": 131072,
    "hidden_size": 3072,
    "intermediate_size": 9216,
    "num_hidden_layers": 26,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "attention_dropout": 0.0,
    "sliding_window": None,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 1_000_000.0,
        "llama_4_scaling_beta": 0.1,
        "original_max_position_embeddings": 16384,
    },
}
_DROPPED_PREFIXES = ("vision_tower.", "multi_modal_projector.")


def rope_inverse_frequencies(
    head_dim: int, max_positions: int, parameters: Mapping[str, Any]
) -> tuple[Tensor, float]:
    """Return the rotary inverse frequencies and attention factor (default and YaRN types)."""
    base = parameters["rope_theta"]
    kind = parameters.get("rope_type", "default")
    if kind == "default":
        return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)), 1.0
    if kind != "yarn":
        raise ValueError(f"Unsupported Ministral3 rope type {kind!r}")
    factor = parameters["factor"]
    original = parameters["original_max_position_embeddings"]
    attention_factor = parameters.get("attention_factor")
    mscale = parameters.get("mscale")
    mscale_all_dim = parameters.get("mscale_all_dim")
    if factor is None:
        factor = max_positions / original

    def get_mscale(scale: float, mscale: float = 1) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(
                get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
            )
        else:
            attention_factor = get_mscale(factor)
    beta_fast = parameters.get("beta_fast") or 32
    beta_slow = parameters.get("beta_slow") or 1

    def correction_dim(rotations: float) -> float:
        return (head_dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low, high = correction_dim(beta_fast), correction_dim(beta_slow)
    if parameters.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, head_dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp(
        (torch.arange(head_dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1
    )
    pos_freqs = base ** (torch.arange(0, head_dim, 2).to(dtype=torch.float) / head_dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (factor * pos_freqs)
    extrapolation_factor = 1 - ramp.to(dtype=torch.float)
    inv_freq = interpolation * (1 - extrapolation_factor) + extrapolation * extrapolation_factor
    return inv_freq, attention_factor


class Ministral3RMSNorm(nn.Module):
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


class Ministral3MLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Ministral3RotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        inv_freq, self.attention_scaling = rope_inverse_frequencies(
            config["head_dim"], config["max_position_embeddings"], config["rope_parameters"]
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        inv_freq = inv_freq.to(dtype=torch.float, device=x.device)
        positions = position_ids[:, None, :].float()
        freqs = (inv_freq @ positions).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def llama_4_attention_scale(positions: Tensor, beta: float, max_positions: int) -> Tensor:
    scaling = 1 + beta * torch.log(1 + torch.floor(positions / max_positions))
    return scaling[:, None, :, None]


class Ministral3Attention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden, head_dim = config["hidden_size"], config["head_dim"]
        self.head_dim = head_dim
        self.num_key_value_groups = config["num_attention_heads"] // config["num_key_value_heads"]
        self.scaling = head_dim**-0.5
        self.beta = config["rope_parameters"].get("llama_4_scaling_beta")
        self.original_positions = config["rope_parameters"].get("original_max_position_embeddings")
        self.q_proj = nn.Linear(hidden, config["num_attention_heads"] * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, config["num_key_value_heads"] * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, config["num_key_value_heads"] * head_dim, bias=False)
        self.o_proj = nn.Linear(config["num_attention_heads"] * head_dim, hidden, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        position_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        query = query * llama_4_attention_scale(
            position_ids, self.beta, self.original_positions
        ).to(query.dtype)

        sdpa_kwargs: dict[str, bool] = {}
        if self.num_key_value_groups > 1:
            if attention_mask is None:
                sdpa_kwargs = {"enable_gqa": True}
            else:
                key = repeat_kv(key, self.num_key_value_groups)
                value = repeat_kv(value, self.num_key_value_groups)
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scaling,
            is_causal=query.shape[2] > 1 and attention_mask is None,
            **sdpa_kwargs,
        )
        out = out.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
        return self.o_proj(out)


class Ministral3DecoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.self_attn = Ministral3Attention(config)
        self.mlp = Ministral3MLP(config)
        self.input_layernorm = Ministral3RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.post_attention_layernorm = Ministral3RMSNorm(
            config["hidden_size"], config["rms_norm_eps"]
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        position_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_embeddings, position_ids, attention_mask
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class Ministral3Transformer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            Ministral3DecoderLayer(config) for _ in range(config["num_hidden_layers"])
        )
        self.norm = Ministral3RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.rotary_emb = Ministral3RotaryEmbedding(config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        output_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)[None]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        mask = None
        if attention_mask is not None and not bool(attention_mask.bool().all()):
            length = hidden_states.shape[1]
            positions = torch.arange(length, device=hidden_states.device)
            mask = (positions[:, None] >= positions[None, :])[None, None]
            mask = mask & attention_mask[:, None, None, :].bool()
        collected = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, position_ids, mask)
            collected.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        collected[-1] = hidden_states
        if output_hidden_states:
            return hidden_states, tuple(collected)
        return hidden_states


class _LanguageModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.model = Ministral3Transformer(config)


class Mistral3TextModel(NativeModel):
    """Language model of a ``Mistral3Model`` snapshot (config.json with a ``text_config``)."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        text = MINISTRAL3_DEFAULTS | dict(config.get("text_config", config))
        super().__init__(text)
        unsupported = {
            "activation": self.config["hidden_act"] != "silu",
            "sliding window": self.config["sliding_window"] is not None,
            "attention dropout": self.config["attention_dropout"] != 0.0,
        }
        for option, is_unsupported in unsupported.items():
            if is_unsupported:
                raise ValueError(f"Unsupported Ministral3 {option}")
        self.language_model = _LanguageModel(self.config)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Drop the vision tower and projector tensors; the text path never uses them."""
        return {
            name: value for name, value in state.items() if not name.startswith(_DROPPED_PREFIXES)
        }

    def init_non_persistent_buffers(self) -> None:
        rotary = self.language_model.model.rotary_emb
        rotary.inv_freq, rotary.attention_scaling = rope_inverse_frequencies(
            self.config["head_dim"],
            self.config["max_position_embeddings"],
            self.config["rope_parameters"],
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        output_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        return self.language_model.model(
            input_ids, attention_mask, output_hidden_states=output_hidden_states
        )
