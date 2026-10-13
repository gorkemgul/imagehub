"""T5Gemma encoder used by PRX."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from imagehub.checkpoints import NativeModel
from imagehub.models.gemma2 import Gemma2Attention, Gemma2MLP, Gemma2RMSNorm
from imagehub.models.qwen3 import Qwen3RotaryEmbedding

T5GEMMA_DEFAULTS: dict[str, Any] = {
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
    "dropout_rate": 0.0,
    "pad_token_id": 0,
}


class T5GemmaEncoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        size = config["hidden_size"]
        eps = config["rms_norm_eps"]
        self.self_attn = Gemma2Attention(config)
        self.pre_self_attn_layernorm = Gemma2RMSNorm(size, eps)
        self.post_self_attn_layernorm = Gemma2RMSNorm(size, eps)
        self.mlp = Gemma2MLP(config)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(size, eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(size, eps)
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + self.dropout(hidden_states)


class T5GemmaEncoder(NativeModel):
    """Encoder-only T5Gemma with the published flat tensor layout."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        normalized = dict(config)
        rope = normalized.get("rope_parameters")
        if isinstance(rope, Mapping):
            normalized.setdefault("rope_theta", rope.get("rope_theta", 10000.0))
        super().__init__(T5GEMMA_DEFAULTS | normalized)
        if self.config["hidden_activation"] != "gelu_pytorch_tanh":
            raise ValueError("Unsupported T5Gemma activation")
        count = self.config["num_hidden_layers"]
        layer_types = self.config["layer_types"] or [
            "sliding_attention" if not index % 2 else "full_attention" for index in range(count)
        ]
        if len(layer_types) != count or not set(layer_types) <= {
            "sliding_attention",
            "full_attention",
        }:
            raise ValueError("Unsupported T5Gemma layer_types")
        self.layer_types: list[str] = list(layer_types)
        self.embed_tokens = nn.Embedding(
            self.config["vocab_size"], self.config["hidden_size"], self.config["pad_token_id"]
        )
        self.norm = Gemma2RMSNorm(self.config["hidden_size"], self.config["rms_norm_eps"])
        self.layers = nn.ModuleList(T5GemmaEncoderLayer(self.config) for _ in range(count))
        self.dropout = nn.Dropout(self.config["dropout_rate"])
        self.rotary_emb = Qwen3RotaryEmbedding(self.config)

    def init_non_persistent_buffers(self) -> None:
        dim = self.config["head_dim"]
        base = self.config["rope_theta"]
        self.rotary_emb.inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

    def _masks(
        self, attention_mask: Tensor, batch: int, length: int, device: torch.device
    ) -> dict[str, Tensor | None]:
        padding = attention_mask.to(device=device, dtype=torch.bool)
        full = padding[:, None, None, :].expand(batch, 1, length, length)
        positions = torch.arange(length, device=device)
        window = (positions[None, :] - positions[:, None]).abs() <= self.config["sliding_window"]
        sliding = full & window[None, None]
        return {
            "full_attention": None if bool(padding.all()) else full,
            "sliding_attention": None
            if bool(padding.all()) and length <= self.config["sliding_window"] + 1
            else sliding,
        }

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config["pad_token_id"])
        hidden_states = self.embed_tokens(input_ids)
        normalizer = torch.tensor(
            self.config["hidden_size"] ** 0.5,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        hidden_states = self.dropout(hidden_states * normalizer)
        length = hidden_states.shape[1]
        position_ids = torch.arange(length, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        masks = self._masks(attention_mask, hidden_states.shape[0], length, hidden_states.device)
        for layer_type, layer in zip(self.layer_types, self.layers, strict=True):
            hidden_states = layer(hidden_states, position_embeddings, masks[layer_type])
        return self.dropout(self.norm(hidden_states))
