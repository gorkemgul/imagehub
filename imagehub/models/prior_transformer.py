"""Kandinsky 2.2 prior transformer with published diffusers tensor names."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

PRIOR_DEFAULTS: dict[str, Any] = {
    "num_attention_heads": 32,
    "attention_head_dim": 64,
    "num_layers": 20,
    "embedding_dim": 768,
    "num_embeddings": 77,
    "additional_embeddings": 4,
    "dropout": 0.0,
    "time_embed_act_fn": "silu",
    "norm_in_type": None,
    "embedding_proj_norm_type": None,
    "encoder_hid_proj_type": "linear",
    "added_emb_type": "prd",
    "time_embed_dim": None,
    "embedding_proj_dim": None,
    "clip_embed_dim": None,
}


def _timestep_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / half_dim
    embeddings = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    embeddings = torch.cat([torch.cos(embeddings), torch.sin(embeddings)], dim=-1)
    if embedding_dim % 2 == 1:
        embeddings = F.pad(embeddings, (0, 1, 0, 0))
    return embeddings


class TimestepEmbedding(nn.Module):
    def __init__(self, inner_dim: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(inner_dim, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, inner_dim)

    def forward(self, sample: Tensor) -> Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class PriorAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(dropout)])

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
        batch_size = hidden_states.shape[0]
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        if attention_mask is not None:
            attention_mask = attention_mask.view(
                batch_size, self.heads, -1, attention_mask.shape[-1]
            )
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)


class GELU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim * 4)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.gelu(self.proj(hidden_states), approximate="none")


class PriorFeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.ModuleList([GELU(dim), nn.Dropout(dropout), nn.Linear(dim * 4, dim)])

    def forward(self, hidden_states: Tensor) -> Tensor:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


class PriorTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = PriorAttention(dim, heads, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = PriorFeedForward(dim, dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
        hidden_states = self.attn1(self.norm1(hidden_states), attention_mask) + hidden_states
        return self.ff(self.norm3(hidden_states)) + hidden_states


def _causal_attention_mask(length: int, device: torch.device | None = None) -> Tensor:
    mask = torch.full((length, length), -10000.0, device=device)
    mask.triu_(1)
    return mask[None]


class PriorTransformer(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(PRIOR_DEFAULTS | dict(config))
        cfg = self.config
        heads = cfg["num_attention_heads"]
        inner_dim = heads * cfg["attention_head_dim"]
        time_embed_dim = cfg["time_embed_dim"] or inner_dim
        embedding_proj_dim = cfg["embedding_proj_dim"] or cfg["embedding_dim"]
        clip_embed_dim = cfg["clip_embed_dim"] or cfg["embedding_dim"]
        self.additional_embeddings = cfg["additional_embeddings"]
        self.inner_dim = inner_dim
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim)
        self.proj_in = nn.Linear(cfg["embedding_dim"], inner_dim)
        if cfg["embedding_proj_norm_type"] is None:
            self.embedding_proj_norm = None
        elif cfg["embedding_proj_norm_type"] == "layer":
            self.embedding_proj_norm = nn.LayerNorm(embedding_proj_dim)
        else:
            raise ValueError(
                f"unsupported embedding_proj_norm_type: {cfg['embedding_proj_norm_type']}"
            )
        self.embedding_proj = nn.Linear(embedding_proj_dim, inner_dim)
        if cfg["encoder_hid_proj_type"] is None:
            self.encoder_hidden_states_proj = None
        elif cfg["encoder_hid_proj_type"] == "linear":
            self.encoder_hidden_states_proj = nn.Linear(cfg["embedding_dim"], inner_dim)
        else:
            raise ValueError(f"unsupported encoder_hid_proj_type: {cfg['encoder_hid_proj_type']}")
        length = cfg["num_embeddings"] + cfg["additional_embeddings"]
        self.positional_embedding = nn.Parameter(torch.zeros(1, length, inner_dim))
        if cfg["added_emb_type"] == "prd":
            self.prd_embedding = nn.Parameter(torch.zeros(1, 1, inner_dim))
        elif cfg["added_emb_type"] is None:
            self.register_parameter("prd_embedding", None)
        else:
            raise ValueError(f"unsupported added_emb_type: {cfg['added_emb_type']}")
        self.transformer_blocks = nn.ModuleList(
            PriorTransformerBlock(inner_dim, heads, cfg["dropout"])
            for _ in range(cfg["num_layers"])
        )
        self.norm_in: nn.LayerNorm | None
        if cfg["norm_in_type"] == "layer":
            self.norm_in = nn.LayerNorm(inner_dim)
        elif cfg["norm_in_type"] is None:
            self.norm_in = None
        else:
            raise ValueError(f"unsupported norm_in_type: {cfg['norm_in_type']}")
        self.norm_out = nn.LayerNorm(inner_dim)
        self.proj_to_clip_embeddings = nn.Linear(inner_dim, clip_embed_dim)
        self.register_buffer(
            "causal_attention_mask", _causal_attention_mask(length), persistent=False
        )
        self.clip_mean = nn.Parameter(torch.zeros(1, clip_embed_dim))
        self.clip_std = nn.Parameter(torch.zeros(1, clip_embed_dim))

    def init_non_persistent_buffers(self) -> None:
        length = self.config["num_embeddings"] + self.config["additional_embeddings"]
        self.causal_attention_mask = _causal_attention_mask(length)

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor | float | int,
        proj_embedding: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size = hidden_states.shape[0]
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=hidden_states.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(hidden_states.device)
        timesteps = timesteps * torch.ones(
            batch_size, dtype=timesteps.dtype, device=timesteps.device
        )
        timesteps_projected = _timestep_embedding(timesteps, self.inner_dim)
        timesteps_projected = timesteps_projected.to(dtype=self.proj_in.weight.dtype)
        time_embeddings = self.time_embedding(timesteps_projected)
        if self.embedding_proj_norm is not None:
            proj_embedding = self.embedding_proj_norm(proj_embedding)
        proj_embeddings = self.embedding_proj(proj_embedding)
        if self.encoder_hidden_states_proj is not None and encoder_hidden_states is not None:
            encoder_hidden_states = self.encoder_hidden_states_proj(encoder_hidden_states)
        elif self.encoder_hidden_states_proj is not None:
            raise ValueError("encoder_hidden_states is required")
        hidden_states = self.proj_in(hidden_states)
        positional_embeddings = self.positional_embedding.to(hidden_states.dtype)
        additional_embeds = []
        additional_embeddings_len = 0
        if encoder_hidden_states is not None:
            additional_embeds.append(encoder_hidden_states)
            additional_embeddings_len += encoder_hidden_states.shape[1]
        if proj_embeddings.ndim == 2:
            proj_embeddings = proj_embeddings[:, None, :]
        if hidden_states.ndim == 2:
            hidden_states = hidden_states[:, None, :]
        additional_embeds.extend([proj_embeddings, time_embeddings[:, None, :], hidden_states])
        if self.prd_embedding is not None:
            additional_embeds.append(
                self.prd_embedding.to(hidden_states.dtype).expand(batch_size, -1, -1)
            )
        hidden_states = torch.cat(additional_embeds, dim=1)
        additional_embeddings_len += proj_embeddings.shape[1] + 1
        if positional_embeddings.shape[1] < hidden_states.shape[1]:
            positional_embeddings = F.pad(
                positional_embeddings,
                (
                    0,
                    0,
                    additional_embeddings_len,
                    self.prd_embedding.shape[1] if self.prd_embedding is not None else 0,
                ),
                value=0.0,
            )
        hidden_states = hidden_states + positional_embeddings
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = F.pad(attention_mask, (0, self.additional_embeddings), value=0.0)
            attention_mask = (attention_mask[:, None, :] + self.causal_attention_mask).to(
                hidden_states.dtype
            )
            attention_mask = attention_mask.repeat_interleave(
                self.config["num_attention_heads"],
                dim=0,
                output_size=attention_mask.shape[0] * self.config["num_attention_heads"],
            )
        if self.norm_in is not None:
            hidden_states = self.norm_in(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask)
        hidden_states = self.norm_out(hidden_states)
        if self.prd_embedding is not None:
            hidden_states = hidden_states[:, -1]
        else:
            hidden_states = hidden_states[:, additional_embeddings_len:]
        return self.proj_to_clip_embeddings(hidden_states)

    def post_process_latents(self, prior_latents: Tensor) -> Tensor:
        return (prior_latents * self.clip_std) + self.clip_mean
