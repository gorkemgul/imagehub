"""UMT5 encoder (the AuraFlow text encoder): T5 with a relative position bias in every layer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from imagehub.checkpoints import NativeModel
from imagehub.models.t5 import T5_DEFAULTS, T5Block, T5LayerNorm


def _clamp_half(hidden_states: Tensor) -> Tensor:
    if hidden_states.dtype != torch.float16:
        return hidden_states
    max_dtype = torch.finfo(hidden_states.dtype).max
    clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype - 1000, max_dtype)
    return torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)


class UMT5Block(T5Block):
    """A T5 block whose attention owns its relative bias and never reuses an earlier one."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config, has_relative_attention_bias=True)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:  # type: ignore[override]
        hidden_states, _ = self.layer[0](hidden_states, attention_mask, None)
        hidden_states = _clamp_half(hidden_states)
        return _clamp_half(self.layer[1](hidden_states))


class UMT5Stack(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.block = nn.ModuleList(UMT5Block(config) for _ in range(config["num_layers"]))
        self.final_layer_norm = T5LayerNorm(config["d_model"], config["layer_norm_epsilon"])

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        for block in self.block:
            hidden_states = block(hidden_states, attention_mask)
        return self.final_layer_norm(hidden_states)


class UMT5EncoderModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(T5_DEFAULTS | dict(config))
        unsupported = {
            "feed-forward projection": self.config["feed_forward_proj"] != "gated-gelu",
            "activation": self.config["dense_act_fn"] != "gelu_new",
            "ungated feed-forward": not self.config["is_gated_act"],
        }
        for option, is_unsupported in unsupported.items():
            if is_unsupported:
                raise ValueError(f"Unsupported UMT5 {option}")
        self.shared = nn.Embedding(self.config["vocab_size"], self.config["d_model"])
        self.encoder = UMT5Stack(self.config)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor]:
        """Return the final encoder hidden state in Transformers-compatible tuple form."""
        hidden_states = self.shared(input_ids)
        return (self.encoder(hidden_states, attention_mask),)
