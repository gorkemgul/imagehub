"""T5 encoder used as the Stable Diffusion 3.5 third text encoder."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

T5_DEFAULTS: dict[str, Any] = {
    "vocab_size": 32128,
    "d_model": 4096,
    "d_kv": 64,
    "d_ff": 10240,
    "num_layers": 24,
    "num_heads": 64,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "dropout_rate": 0.1,
    "layer_norm_epsilon": 1e-6,
    "feed_forward_proj": "gated-gelu",
    "dense_act_fn": "gelu_new",
    "is_gated_act": True,
}


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


def gelu_new(hidden_states: Tensor) -> Tensor:
    return (
        0.5
        * hidden_states
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (hidden_states + 0.044715 * torch.pow(hidden_states, 3.0))
            )
        )
    )


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(config["d_model"], config["d_ff"], bias=False)
        self.wi_1 = nn.Linear(config["d_model"], config["d_ff"], bias=False)
        self.wo = nn.Linear(config["d_ff"], config["d_model"], bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_gelu = gelu_new(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        if hidden_states.dtype != self.wo.weight.dtype and self.wo.weight.dtype != torch.int8:
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        return self.wo(hidden_states)


class T5LayerFF(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(config)
        self.layer_norm = T5LayerNorm(config["d_model"], config["layer_norm_epsilon"])

    def forward(self, hidden_states: Tensor) -> Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        return hidden_states + forwarded_states


class T5Attention(nn.Module):
    def __init__(self, config: Mapping[str, Any], has_relative_attention_bias: bool) -> None:
        super().__init__()
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config["relative_attention_num_buckets"]
        self.relative_attention_max_distance = config["relative_attention_max_distance"]
        self.key_value_proj_dim = config["d_kv"]
        self.n_heads = config["num_heads"]
        inner_dim = self.n_heads * self.key_value_proj_dim
        self.q = nn.Linear(config["d_model"], inner_dim, bias=False)
        self.k = nn.Linear(config["d_model"], inner_dim, bias=False)
        self.v = nn.Linear(config["d_model"], inner_dim, bias=False)
        self.o = nn.Linear(inner_dim, config["d_model"], bias=False)
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: Tensor,
        *,
        num_buckets: int,
        max_distance: int,
    ) -> Tensor:
        relative_buckets = (relative_position > 0).to(torch.long) * (num_buckets // 2)
        relative_position = torch.abs(relative_position)
        num_buckets //= 2
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        return relative_buckets + torch.where(
            is_small, relative_position, relative_position_if_large
        )

    def compute_bias(self, length: int, device: torch.device) -> Tensor:
        context_position = torch.arange(length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        return values.permute(2, 0, 1).unsqueeze(0)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        position_bias: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.key_value_proj_dim)
        query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_bias is None:
            position_bias = self.compute_bias(input_shape[1], query_states.device)
        sdpa_mask = position_bias
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].bool()
            sdpa_mask = torch.where(mask, position_bias, torch.finfo(key_states.dtype).min)
        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o(attention_output), position_bias


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any], has_relative_attention_bias: bool) -> None:
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config["d_model"], config["layer_norm_epsilon"])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        position_bias: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, position_bias = self.SelfAttention(
            normed_hidden_states, attention_mask, position_bias
        )
        return hidden_states + attention_output, position_bias


class T5Block(nn.Module):
    def __init__(self, config: Mapping[str, Any], has_relative_attention_bias: bool) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [T5LayerSelfAttention(config, has_relative_attention_bias), T5LayerFF(config)]
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        position_bias: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        hidden_states, position_bias = self.layer[0](hidden_states, attention_mask, position_bias)
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, position_bias


class T5Stack(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.block = nn.ModuleList(
            T5Block(config, has_relative_attention_bias=index == 0)
            for index in range(config["num_layers"])
        )
        self.final_layer_norm = T5LayerNorm(config["d_model"], config["layer_norm_epsilon"])

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        position_bias = None
        for block in self.block:
            hidden_states, position_bias = block(hidden_states, attention_mask, position_bias)
        return self.final_layer_norm(hidden_states)


class T5EncoderModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(T5_DEFAULTS | dict(config))
        unsupported = {
            "feed-forward projection": self.config["feed_forward_proj"] != "gated-gelu",
            "activation": self.config["dense_act_fn"] != "gelu_new",
            "ungated feed-forward": not self.config["is_gated_act"],
        }
        for option, is_unsupported in unsupported.items():
            if is_unsupported:
                raise ValueError(f"Unsupported T5 {option}")
        self.shared = nn.Embedding(self.config["vocab_size"], self.config["d_model"])
        self.encoder = T5Stack(self.config)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor]:
        """Return the final encoder hidden state in Transformers-compatible tuple form."""
        hidden_states = self.shared(input_ids)
        return (self.encoder(hidden_states, attention_mask),)
