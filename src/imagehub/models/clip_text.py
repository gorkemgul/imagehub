"""CLIP text transformer (OpenAI ViT-L/14 text tower) with transformers ``CLIPTextModel`` names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

CLIP_TEXT_DEFAULTS: dict[str, Any] = {
    "vocab_size": 49408,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "max_position_embeddings": 77,
    "layer_norm_eps": 1e-5,
    "hidden_act": "quick_gelu",
}


class CLIPAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        q, k, v = (
            proj(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).reshape(b, n, d))


class CLIPMLP(nn.Module):
    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        return self.fc2(x * torch.sigmoid(1.702 * x))  # quick_gelu


class CLIPEncoderLayer(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        dim, eps = cfg["hidden_size"], cfg["layer_norm_eps"]
        self.self_attn = CLIPAttention(dim, cfg["num_attention_heads"])
        self.layer_norm1 = nn.LayerNorm(dim, eps=eps)
        self.mlp = CLIPMLP(dim, cfg["intermediate_size"])
        self.layer_norm2 = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class CLIPEmbeddings(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.position_embedding = nn.Embedding(cfg["max_position_embeddings"], cfg["hidden_size"])

    def forward(self, input_ids: Tensor) -> Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return self.token_embedding(input_ids) + self.position_embedding(positions)[None]


class CLIPEncoder(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(CLIPEncoderLayer(cfg) for _ in range(cfg["num_hidden_layers"]))


class CLIPTextTransformer(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.embeddings = CLIPEmbeddings(cfg)
        self.encoder = CLIPEncoder(cfg)
        self.final_layer_norm = nn.LayerNorm(cfg["hidden_size"], eps=cfg["layer_norm_eps"])

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embeddings(input_ids)
        for layer in self.encoder.layers:
            x = layer(x)
        return self.final_layer_norm(x)


class CLIPTextModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(CLIP_TEXT_DEFAULTS | dict(config))
        if self.config["hidden_act"] != "quick_gelu":
            raise ValueError(f"Unsupported CLIP activation {self.config['hidden_act']!r}")
        self.text_model = CLIPTextTransformer(self.config)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        # Published files store the constant position-id buffer; positions are recomputed.
        state = dict(state)
        state.pop("text_model.embeddings.position_ids", None)
        return state

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return ``last_hidden_state`` of shape (batch, tokens, hidden_size)."""
        return self.text_model(input_ids)
