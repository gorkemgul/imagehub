"""Gemma-2 decoder stack used as the Sana text encoder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb, repeat_kv
from imagehub.models.qwen3_vl_text import _use_gqa_in_sdpa

GEMMA2_DEFAULTS: dict[str, Any] = {
    "vocab_size": 256000,
    "hidden_size": 2304,
    "intermediate_size": 9216,
    "num_hidden_layers": 26,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_logit_softcapping": 50.0,
    "query_pre_attn_scalar": 256,
    "sliding_window": 4096,
    "layer_types": None,
    "hidden_activation": "gelu_pytorch_tanh",
}


class Gemma2RMSNorm(nn.Module):
    """RMS norm in float32 with a ``1 + weight`` gain."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        value = x.float()
        output = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Gemma2MLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Gemma2Attention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        head_dim = config["head_dim"]
        bias = config["attention_bias"]
        self.head_dim = head_dim
        self.num_key_value_groups = config["num_attention_heads"] // config["num_key_value_heads"]
        self.scaling = config["query_pre_attn_scalar"] ** -0.5
        self.attention_dropout = config["attention_dropout"]
        self.q_proj = nn.Linear(hidden_size, config["num_attention_heads"] * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, config["num_key_value_heads"] * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, config["num_key_value_heads"] * head_dim, bias=bias)
        self.o_proj = nn.Linear(config["num_attention_heads"] * head_dim, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # The pipeline calls the text encoder with transformers' default ``use_cache=True``;
        # the key-value cache stores keys and values as contiguous copies, and the SDPA kernel's
        # summation order depends on that layout.
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        sdpa_kwargs: dict[str, bool] = {}
        if self.num_key_value_groups > 1:
            if _use_gqa_in_sdpa(attention_mask, key_states, value_states):
                sdpa_kwargs = {"enable_gqa": True}
            else:
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

        # transformers' default SDPA path does not apply ``attn_logit_softcapping`` (only the
        # eager path does), so the published text encoder is evaluated without it.
        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
            is_causal=query_states.shape[2] > 1 and attention_mask is None,
            **sdpa_kwargs,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention_output)


class Gemma2DecoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        eps = config["rms_norm_eps"]
        size = config["hidden_size"]
        self.self_attn = Gemma2Attention(config)
        self.mlp = Gemma2MLP(config)
        self.input_layernorm = Gemma2RMSNorm(size, eps)
        self.post_attention_layernorm = Gemma2RMSNorm(size, eps)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(size, eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(size, eps)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class Gemma2Model(NativeModel):
    """Gemma-2 base model (no LM head); ``forward`` returns the final-norm hidden states."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(GEMMA2_DEFAULTS | dict(config))
        if self.config["hidden_activation"] != "gelu_pytorch_tanh":
            raise ValueError("Unsupported Gemma2 activation")
        count = self.config["num_hidden_layers"]
        layer_types = self.config["layer_types"] or [
            "sliding_attention" if (index + 1) % 2 else "full_attention" for index in range(count)
        ]
        if len(layer_types) != count or not set(layer_types) <= {
            "sliding_attention",
            "full_attention",
        }:
            raise ValueError("Unsupported Gemma2 layer_types")
        self.layer_types: list[str] = list(layer_types)

        self.embed_tokens = nn.Embedding(self.config["vocab_size"], self.config["hidden_size"])
        self.layers = nn.ModuleList(Gemma2DecoderLayer(self.config) for _ in range(count))
        self.norm = Gemma2RMSNorm(self.config["hidden_size"], self.config["rms_norm_eps"])
        self.rotary_emb = Qwen3RotaryEmbedding(self.config)

    def init_non_persistent_buffers(self) -> None:
        dim = self.config["head_dim"]
        base = self.config["rope_theta"]
        self.rotary_emb.inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

    def _masks(
        self, attention_mask: Tensor | None, batch: int, length: int, device: torch.device
    ) -> dict[str, Tensor | None]:
        """Boolean SDPA masks as transformers builds them; ``None`` means ``is_causal``."""
        padding = None if attention_mask is None else attention_mask.to(torch.bool)
        has_padding = padding is not None and not bool(padding.all())
        positions = torch.arange(length, device=device)
        causal = positions[None, :] <= positions[:, None]
        window = self.config["sliding_window"]
        sliding = causal & (positions[None, :] > positions[:, None] - window)

        def combine(base: Tensor) -> Tensor:
            mask = base[None, None]
            if padding is not None:
                mask = mask & padding[:, None, None, :]
            return mask.expand(batch, -1, length, length)

        return {
            "full_attention": combine(causal) if has_padding else None,
            "sliding_attention": combine(sliding) if has_padding or length >= window else None,
        }

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        hidden_state_index: int | None = None,
    ) -> Tensor:
        """Final-norm hidden states, or one entry of transformers' ``hidden_states`` tuple.

        ``hidden_state_index`` indexes that tuple (embeddings, the output of each layer, and the
        final-norm output last), so ``-2`` is the input of the last layer, without the final norm.
        """
        count = len(self.layers)
        applied = count if hidden_state_index is None else hidden_state_index % (count + 1)
        embeddings = self.embed_tokens(input_ids)
        scale = torch.tensor(self.config["hidden_size"] ** 0.5, device=embeddings.device)
        hidden_states = embeddings * scale.to(self.embed_tokens.weight.dtype)
        length = hidden_states.shape[1]
        position_ids = torch.arange(length, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        masks = self._masks(attention_mask, hidden_states.shape[0], length, hidden_states.device)
        for layer_type, layer in list(zip(self.layer_types, self.layers, strict=True))[:applied]:
            hidden_states = layer(hidden_states, position_embeddings, masks[layer_type])
        return self.norm(hidden_states) if applied == count else hidden_states
