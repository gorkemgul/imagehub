"""Qwen2 decoder used as the HART text encoder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel


class Qwen2RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        return self.weight * (
            x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)
        ).to(dtype)


class Qwen2MLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden, intermediate = config["hidden_size"], config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _rotate_half(x: Tensor) -> Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class Qwen2Attention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = hidden // self.heads
        self.groups = self.heads // self.kv_heads
        self.dropout = config["attention_dropout"]
        self.q_proj = nn.Linear(hidden, self.heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor | None) -> Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = cos[:, None], sin[:, None]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        k = (
            k[:, :, None]
            .expand(-1, -1, self.groups, -1, -1)
            .reshape(batch, self.heads, length, self.head_dim)
        )
        v = (
            v[:, :, None]
            .expand(-1, -1, self.groups, -1, -1)
            .reshape(batch, self.heads, length, self.head_dim)
        )
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None,
        )
        return self.o_proj(out.transpose(1, 2).contiguous().view(batch, length, -1))


class Qwen2Layer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.post_attention_layernorm = Qwen2RMSNorm(config["hidden_size"], config["rms_norm_eps"])

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor | None) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2Transformer(nn.Module):
    inv_freq: Tensor

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(Qwen2Layer(config) for _ in range(config["num_hidden_layers"]))
        self.norm = Qwen2RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.register_buffer("inv_freq", self._inv_freq(), persistent=False)

    def _inv_freq(self) -> Tensor:
        dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        return 1.0 / (
            self.config["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        x = self.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        freqs = torch.einsum("bi,j->bij", positions.float(), self.inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        mask = None
        if not bool(attention_mask.bool().all()):
            length = input_ids.shape[1]
            causal = (
                torch.arange(length, device=x.device)[:, None]
                >= torch.arange(length, device=x.device)[None, :]
            )
            mask = causal[None, None] & attention_mask[:, None, None].bool()
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.norm(x)


class Qwen2Model(NativeModel):
    """Text-only Qwen2 model with the published `model.*` tensor layout."""

    skipped_prefixes = ("visual.",)

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        if config.get("hidden_act", "silu") != "silu" or config.get("use_sliding_window", False):
            raise ValueError("Unsupported Qwen2 configuration")
        self.model = Qwen2Transformer(self.config)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        prepared = {
            name: value
            for name, value in state.items()
            if name != "lm_head.weight" and not name.startswith("visual.")
        }
        if prepared and not next(iter(prepared)).startswith("model."):
            return {"model." + name: value for name, value in prepared.items()}
        return prepared

    def init_non_persistent_buffers(self) -> None:
        self.model.inv_freq = self.model._inv_freq()

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.model(input_ids, attention_mask)
