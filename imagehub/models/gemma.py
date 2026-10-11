"""Gemma (first generation) decoder stack used as the Lumina-Next text encoder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from imagehub.checkpoints import NativeModel
from imagehub.models.gemma2 import Gemma2Attention, Gemma2MLP, Gemma2RMSNorm
from imagehub.models.qwen3 import Qwen3RotaryEmbedding

GEMMA_DEFAULTS: dict[str, Any] = {
    "vocab_size": 256000,
    "hidden_size": 2048,
    "intermediate_size": 16384,
    "num_hidden_layers": 18,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_activation": None,
}
# The published configs carry the legacy value ``hidden_act="gelu"``; transformers maps it to the
# tanh approximation the model was trained with, and so does this implementation.
_TANH_GELU = ("gelu", "gelu_pytorch_tanh")


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        eps = config["rms_norm_eps"]
        size = config["hidden_size"]
        # Gemma scales the attention logits by ``head_dim ** -0.5`` and has no logit softcapping;
        # the shared attention takes the scale as ``query_pre_attn_scalar ** -0.5``.
        self.self_attn = Gemma2Attention({**config, "query_pre_attn_scalar": config["head_dim"]})
        self.mlp = Gemma2MLP(config)
        self.input_layernorm = Gemma2RMSNorm(size, eps)
        self.post_attention_layernorm = Gemma2RMSNorm(size, eps)

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
        return residual + self.mlp(hidden_states)


class GemmaModel(NativeModel):
    """Gemma base model (no LM head); ``forward`` returns the final-norm hidden states."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(GEMMA_DEFAULTS | dict(config))
        activation = self.config["hidden_activation"] or self.config["hidden_act"]
        if activation not in _TANH_GELU:
            raise ValueError("Unsupported Gemma activation")
        count = self.config["num_hidden_layers"]
        self.embed_tokens = nn.Embedding(self.config["vocab_size"], self.config["hidden_size"])
        self.layers = nn.ModuleList(GemmaDecoderLayer(self.config) for _ in range(count))
        self.norm = Gemma2RMSNorm(self.config["hidden_size"], self.config["rms_norm_eps"])
        self.rotary_emb = Qwen3RotaryEmbedding(self.config)

    def init_non_persistent_buffers(self) -> None:
        dim = self.config["head_dim"]
        base = self.config["rope_theta"]
        self.rotary_emb.inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

    @staticmethod
    def _mask(attention_mask: Tensor | None, length: int, device: torch.device) -> Tensor | None:
        """Boolean SDPA mask as transformers builds it; ``None`` means ``is_causal``."""
        if attention_mask is None:
            return None
        padding = attention_mask.to(torch.bool)
        if bool(padding.all()):
            return None
        positions = torch.arange(length, device=device)
        causal = positions[None, :] <= positions[:, None]
        mask = causal[None, None] & padding[:, None, None, :]
        return mask.expand(padding.shape[0], -1, length, length)

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
        mask = self._mask(attention_mask, length, hidden_states.device)
        for layer in list(self.layers)[:applied]:
            hidden_states = layer(hidden_states, position_embeddings, mask)
        return self.norm(hidden_states) if applied == count else hidden_states
