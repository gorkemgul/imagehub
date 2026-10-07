"""GLM-4 text encoder used by CogView4."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

GLM_DEFAULTS: dict[str, Any] = {
    "vocab_size": 151552,
    "hidden_size": 4096,
    "intermediate_size": 13696,
    "num_hidden_layers": 40,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1.5625e-7,
    "rope_theta": 10_000.0,
    "partial_rotary_factor": 0.5,
    "attention_bias": True,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "pad_token_id": 151329,
}


class GlmRMSNorm(nn.Module):
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


class GlmMLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config["hidden_size"], 2 * config["intermediate_size"], bias=False
        )
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, up_states = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(up_states * F.silu(gate))


class GlmRotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = int(config["head_dim"] * config["partial_rotary_factor"])
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
        embeddings = torch.cat((freqs, freqs), dim=-1)
        return embeddings.cos().to(dtype=x.dtype), embeddings.sin().to(dtype=x.dtype)


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rotary_pos_emb(
    query: Tensor, key: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
    rotary_dim = cos.shape[-1]
    query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    query_embed = (query_rotary * cos) + (_rotate_half(query_rotary) * sin)
    key_embed = (key_rotary * cos) + (_rotate_half(key_rotary) * sin)
    return torch.cat((query_embed, query_pass), dim=-1), torch.cat((key_embed, key_pass), dim=-1)


def _repeat_kv(hidden_states: Tensor, repetitions: int) -> Tensor:
    batch, key_value_heads, sequence_length, head_dim = hidden_states.shape
    if repetitions == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, key_value_heads, repetitions, sequence_length, head_dim
    )
    return hidden_states.reshape(batch, key_value_heads * repetitions, sequence_length, head_dim)


def _torch_version_at_least(major: int, minor: int) -> bool:
    version = torch.__version__.split("+", maxsplit=1)[0].split(".")
    return (int(version[0]), int(version[1])) >= (major, minor)


def _use_gqa_in_sdpa(attention_mask: Tensor | None, key: Tensor, value: Tensor) -> bool:
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return _torch_version_at_least(2, 8)
    if torch.backends.mps.is_available():
        return _torch_version_at_least(2, 13)
    return attention_mask is None and key.shape[-1] == value.shape[-1] <= 256


class GlmAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        head_dim = config["head_dim"]
        self.head_dim = head_dim
        self.num_key_value_groups = config["num_attention_heads"] // config["num_key_value_heads"]
        self.scaling = head_dim**-0.5
        self.attention_dropout = config["attention_dropout"]
        bias = config["attention_bias"]
        self.q_proj = nn.Linear(hidden_size, config["num_attention_heads"] * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, config["num_key_value_heads"] * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, config["num_key_value_heads"] * head_dim, bias=bias)
        self.o_proj = nn.Linear(config["num_attention_heads"] * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = _apply_rotary_pos_emb(
            query_states, key_states, *position_embeddings
        )

        # The reference's default cache concatenates with an empty tensor, making copies.
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        sdpa_kwargs: dict[str, bool] = {}
        if self.num_key_value_groups > 1:
            if _use_gqa_in_sdpa(attention_mask, key_states, value_states):
                sdpa_kwargs = {"enable_gqa": True}
            else:
                key_states = _repeat_kv(key_states, self.num_key_value_groups)
                value_states = _repeat_kv(value_states, self.num_key_value_groups)

        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
            is_causal=attention_mask is None and query_states.shape[2] > 1,
            **sdpa_kwargs,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention_output)


class GlmDecoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.self_attn = GlmAttention(config)
        self.mlp = GlmMLP(config)
        self.input_layernorm = GlmRMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.post_attention_layernorm = GlmRMSNorm(config["hidden_size"], config["rms_norm_eps"])

    def forward(self, hidden_states: Tensor, position_embeddings: tuple[Tensor, Tensor]) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class GlmModel(NativeModel):
    """GLM decoder returning the penultimate hidden state consumed by CogView4."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(GLM_DEFAULTS | dict(config))
        if self.config["hidden_act"] != "silu":
            raise ValueError("Unsupported GLM activation")
        if not self.config["attention_bias"]:
            raise ValueError("Unsupported GLM attention bias")
        if self.config["partial_rotary_factor"] != 0.5:
            raise ValueError("Unsupported GLM partial rotary factor")
        self.embed_tokens = nn.Embedding(
            self.config["vocab_size"],
            self.config["hidden_size"],
            self.config["pad_token_id"],
        )
        self.layers = nn.ModuleList(
            GlmDecoderLayer(self.config) for _ in range(self.config["num_hidden_layers"])
        )
        self.norm = GlmRMSNorm(self.config["hidden_size"], self.config["rms_norm_eps"])
        self.rotary_emb = GlmRotaryEmbedding(self.config)

    def init_non_persistent_buffers(self) -> None:
        dim = int(self.config["head_dim"] * self.config["partial_rotary_factor"])
        base = self.config["rope_theta"]
        self.rotary_emb.inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(
            0
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        pipeline_hidden_state = hidden_states
        for index, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_embeddings)
            if index == len(self.layers) - 2:
                pipeline_hidden_state = hidden_states
        return pipeline_hidden_state
