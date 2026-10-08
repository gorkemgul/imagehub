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
    "projection_dim": 512,
    "eos_token_id": 49407,
}


class CLIPAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, n, d = x.shape
        q, k, v = (
            proj(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        if mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(out.transpose(1, 2).reshape(b, n, d))


class CLIPMLP(nn.Module):
    def __init__(self, dim: int, inner: int, activation: str) -> None:
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        if self.activation == "gelu":
            x = F.gelu(x)
        else:
            x = x * torch.sigmoid(1.702 * x)
        return self.fc2(x)


class CLIPEncoderLayer(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        dim, eps = cfg["hidden_size"], cfg["layer_norm_eps"]
        self.self_attn = CLIPAttention(dim, cfg["num_attention_heads"])
        self.layer_norm1 = nn.LayerNorm(dim, eps=eps)
        self.mlp = CLIPMLP(dim, cfg["intermediate_size"], cfg["hidden_act"])
        self.layer_norm2 = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.self_attn(self.layer_norm1(x), mask)
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

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        x = self.embeddings(input_ids)
        mask = None
        # An all-ones padding mask is plain causal attention; only real padding needs a mask.
        if attention_mask is not None and not bool(attention_mask.all()):
            n = input_ids.shape[1]
            causal = torch.ones(n, n, dtype=torch.bool, device=input_ids.device).tril()
            mask = causal[None, None] & attention_mask.to(torch.bool)[:, None, None, :]
        hidden_states = [x]
        for layer in self.encoder.layers:
            x = layer(x, mask)
            hidden_states.append(x)
        x = self.final_layer_norm(x)
        if output_hidden_states:
            return x, tuple(hidden_states)
        return x


class CLIPTextModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(CLIP_TEXT_DEFAULTS | dict(config))
        if self.config["hidden_act"] not in ("quick_gelu", "gelu"):
            raise ValueError(f"Unsupported CLIP activation {self.config['hidden_act']!r}")
        self.text_model = CLIPTextTransformer(self.config)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        # Published files store the constant position-id buffer; positions are recomputed.
        state = dict(state)
        state.pop("text_model.embeddings.position_ids", None)
        return state

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        """Return ``last_hidden_state`` of shape (batch, tokens, hidden_size)."""
        return self.text_model(
            input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states
        )


class CLIPTextModelWithProjection(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(CLIP_TEXT_DEFAULTS | dict(config))
        if self.config["hidden_act"] not in ("quick_gelu", "gelu"):
            raise ValueError(f"Unsupported CLIP activation {self.config['hidden_act']!r}")
        self.text_model = CLIPTextTransformer(self.config)
        self.text_projection = nn.Linear(
            self.config["hidden_size"], self.config["projection_dim"], bias=False
        )

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        state = dict(state)
        state.pop("text_model.embeddings.position_ids", None)
        return state

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        last_hidden_state, hidden_states = self.text_model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        if self.config["eos_token_id"] == 2:
            positions = input_ids.to(dtype=torch.int).argmax(dim=-1)
        else:
            positions = (
                (input_ids.to(dtype=torch.int) == self.config["eos_token_id"]).int().argmax(dim=-1)
            )
        pooled = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), positions
        ]
        text_embeds = self.text_projection(pooled)
        if output_hidden_states:
            return text_embeds, last_hidden_state, hidden_states
        return text_embeds, last_hidden_state
