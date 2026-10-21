"""Show-o's Phi-1.5 language model with omni-attention."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

PHI_1_5_CONFIG: dict[str, Any] = {
    "attention_dropout": 0.0,
    "embd_pdrop": 0.0,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "layer_norm_eps": 1e-5,
    "max_position_embeddings": 2048,
    "num_attention_heads": 32,
    "num_hidden_layers": 24,
    "num_key_value_heads": 32,
    "partial_rotary_factor": 0.5,
    "resid_pdrop": 0.0,
    "rope_theta": 10000.0,
    "vocab_size": 58498,
}


def _rotate_half(x: Tensor) -> Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def _gelu_new(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class PhiAttention(nn.Module):
    inv_freq: Tensor

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = hidden // self.heads
        self.groups = self.heads // self.kv_heads
        self.rotary_dim = int(config["partial_rotary_factor"] * self.head_dim)
        self.dropout = config["attention_dropout"]
        self.q_proj = nn.Linear(hidden, self.heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=True)
        self.dense = nn.Linear(self.heads * self.head_dim, hidden, bias=True)
        self.q_layernorm = nn.LayerNorm(self.head_dim, eps=config["layer_norm_eps"])
        self.k_layernorm = nn.LayerNorm(self.head_dim, eps=config["layer_norm_eps"])
        self.rope_theta = config["rope_theta"]
        self.register_buffer("inv_freq", self._inv_freq(), persistent=False)

    def _inv_freq(self) -> Tensor:
        return 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, length, self.heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch, length, self.kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch, length, self.kv_heads, self.head_dim)
        query = self.q_layernorm(query.transpose(1, 2))
        key = self.k_layernorm(key.transpose(1, 2))
        value = value.transpose(1, 2)
        positions = torch.arange(length, device=hidden_states.device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq.float())
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cosine = embedding.cos().to(query.dtype)[None, None]
        sine = embedding.sin().to(query.dtype)[None, None]
        query_rot, query_pass = query[..., : self.rotary_dim], query[..., self.rotary_dim :]
        key_rot, key_pass = key[..., : self.rotary_dim], key[..., self.rotary_dim :]
        query = torch.cat((query_rot * cosine + _rotate_half(query_rot) * sine, query_pass), -1)
        key = torch.cat((key_rot * cosine + _rotate_half(key_rot) * sine, key_pass), -1)
        if self.groups != 1:
            key = (
                key[:, :, None]
                .expand(-1, -1, self.groups, -1, -1)
                .reshape(batch, self.heads, length, self.head_dim)
            )
            value = (
                value[:, :, None]
                .expand(-1, -1, self.groups, -1, -1)
                .reshape(batch, self.heads, length, self.head_dim)
            )
        weights = torch.matmul(query.float(), key.float().transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(value.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        return self.dense(output.reshape(batch, length, -1))


class PhiMLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"])

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(_gelu_new(self.fc1(hidden_states)))


class PhiLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.self_attn = PhiAttention(config)
        self.mlp = PhiMLP(config)
        self.input_layernorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.resid_dropout = config["resid_pdrop"]

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
        normalized = self.input_layernorm(hidden_states)
        attention = F.dropout(
            self.self_attn(normalized, attention_mask), self.resid_dropout, self.training
        )
        feed_forward = F.dropout(self.mlp(normalized), self.resid_dropout, self.training)
        return attention + feed_forward + hidden_states


class PhiTransformer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.embed_dropout = config["embd_pdrop"]
        self.layers = nn.ModuleList(PhiLayer(config) for _ in range(config["num_hidden_layers"]))
        self.final_layernorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        hidden_states = F.dropout(self.embed_tokens(input_ids), self.embed_dropout, self.training)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return self.final_layernorm(hidden_states)


class PhiCausalLM(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.model = PhiTransformer(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=True)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.lm_head(self.model(input_ids, attention_mask)).float()


class ShowOModel(NativeModel):
    """Show-o model retaining the published ``showo.*`` tensor layout."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        merged = PHI_1_5_CONFIG | dict(config)
        merged["num_key_value_heads"] = (
            merged.get("num_key_value_heads") or merged["num_attention_heads"]
        )
        super().__init__(merged)
        self.showo = PhiCausalLM(self.config)

    def init_non_persistent_buffers(self) -> None:
        for module in self.showo.model.layers:
            layer = cast(PhiLayer, module)
            layer.self_attn.inv_freq = layer.self_attn._inv_freq()

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.showo(input_ids, attention_mask)
