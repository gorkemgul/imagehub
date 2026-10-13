"""LDMBert text encoder used by the original latent-diffusion text-to-image release."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

LDM_BERT_DEFAULTS: dict[str, Any] = {
    "vocab_size": 30522,
    "max_position_embeddings": 77,
    "encoder_layers": 32,
    "encoder_ffn_dim": 5120,
    "encoder_attention_heads": 8,
    "head_dim": 64,
    "dropout": 0.1,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
    "activation_function": "gelu",
    "d_model": 1280,
    "pad_token_id": 0,
}


class LDMBertAttention(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        dim, heads, head_dim = cfg["d_model"], cfg["encoder_attention_heads"], cfg["head_dim"]
        self.heads, self.head_dim = heads, head_dim
        self.scaling = head_dim**-0.5
        self.k_proj = nn.Linear(dim, heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, heads * head_dim, bias=False)
        self.q_proj = nn.Linear(dim, heads * head_dim, bias=False)
        self.out_proj = nn.Linear(heads * head_dim, dim)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        batch, tokens, _ = x.shape
        shape = (batch, tokens, self.heads, self.head_dim)
        q = (
            (self.q_proj(x) * self.scaling)
            .view(shape)
            .transpose(1, 2)
            .reshape(-1, tokens, self.head_dim)
        )
        k = self.k_proj(x).view(shape).transpose(1, 2).reshape(-1, tokens, self.head_dim)
        v = self.v_proj(x).view(shape).transpose(1, 2).reshape(-1, tokens, self.head_dim)
        weights = torch.bmm(q, k.transpose(1, 2))
        if mask is not None:
            weights = weights.view(batch, self.heads, tokens, tokens) + mask
            weights = weights.view(-1, tokens, tokens)
        output = torch.bmm(F.softmax(weights, dim=-1), v)
        output = output.view(batch, self.heads, tokens, self.head_dim).transpose(1, 2)
        return self.out_proj(output.reshape(batch, tokens, -1))


class LDMBertEncoderLayer(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        dim = cfg["d_model"]
        self.self_attn = LDMBertAttention(cfg)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, cfg["encoder_ffn_dim"])
        self.fc2 = nn.Linear(cfg["encoder_ffn_dim"], dim)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.dropout, self.activation_dropout = cfg["dropout"], cfg["activation_dropout"]

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        x = x + F.dropout(
            self.self_attn(self.self_attn_layer_norm(x), mask), self.dropout, self.training
        )
        residual = x
        x = F.gelu(self.fc1(self.final_layer_norm(x)))
        x = F.dropout(x, self.activation_dropout, self.training)
        return residual + F.dropout(self.fc2(x), self.dropout, self.training)


class LDMBertEncoder(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        dim = cfg["d_model"]
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], dim)
        self.embed_positions = nn.Embedding(cfg["max_position_embeddings"], dim)
        self.layers = nn.ModuleList(LDMBertEncoderLayer(cfg) for _ in range(cfg["encoder_layers"]))
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = cfg["dropout"]

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        tokens = input_ids.shape[1]
        positions = torch.arange(tokens, device=input_ids.device).expand(1, -1)
        x = F.dropout(
            self.embed_tokens(input_ids) + self.embed_positions(positions),
            self.dropout,
            self.training,
        )
        mask = None
        if attention_mask is not None:
            mask = (1.0 - attention_mask[:, None, None, :].to(x.dtype)).expand(-1, 1, tokens, -1)
            mask = mask.masked_fill(mask.bool(), torch.finfo(x.dtype).min)
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)


class LDMBertModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(LDM_BERT_DEFAULTS | dict(config))
        self.model = LDMBertEncoder(self.config)
        self.to_logits = nn.Linear(self.config["d_model"], self.config["vocab_size"])

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.model(input_ids, attention_mask)
