"""Native LongCat-Image transformer for text-to-image inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from imagehub.checkpoints import NativeModel
from imagehub.models.flux_transformer import (
    AdaLayerNormContinuous,
    FluxAttention,
    FluxPosEmbed,
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
    TimestepEmbedding,
    _apply_rotary_emb,
    _attention,
    _get_timestep_embedding,
)

LONGCAT_IMAGE_TRANSFORMER_DEFAULTS: dict[str, Any] = {
    "patch_size": 1,
    "in_channels": 64,
    "num_layers": 10,
    "num_single_layers": 20,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 3584,
    "pooled_projection_dim": 3584,
    "guidance_embeds": False,
    "axes_dims_rope": (16, 56, 56),
}


class LongCatImageTimestepEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)

    def forward(self, timestep: Tensor, hidden_dtype: torch.dtype) -> Tensor:
        timestep_projection = _get_timestep_embedding(timestep, 256)
        return self.timestep_embedder(timestep_projection.to(dtype=hidden_dtype))


class LongCatImageAttention(FluxAttention):
    def __init__(self, dim: int, heads: int, head_dim: int, *, added_projection: bool) -> None:
        super().__init__(dim, heads, head_dim, added_projection=added_projection)
        self.heads = heads

    def forward(
        self,
        hidden_states: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
        encoder_hidden_states: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))
        query = self.norm_q(query)
        key = self.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_query = self.add_q_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
            encoder_key = self.add_k_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
            encoder_value = self.add_v_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        query = _apply_rotary_emb(query, rotary_emb)
        key = _apply_rotary_emb(key, rotary_emb)
        output = _attention(query, key, value).flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is None:
            return output
        encoder_output, output = output.split_with_sizes(
            [encoder_hidden_states.shape[1], output.shape[1] - encoder_hidden_states.shape[1]],
            dim=1,
        )
        output = self.to_out[1](self.to_out[0](output))
        encoder_output = self.to_add_out(encoder_output)
        return output, encoder_output


class LongCatImageTransformerBlock(FluxTransformerBlock):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__(dim, heads, head_dim)
        self.attn = LongCatImageAttention(dim, heads, head_dim, added_projection=True)


class LongCatImageSingleTransformerBlock(FluxSingleTransformerBlock):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__(dim, heads, head_dim)
        self.attn = LongCatImageAttention(dim, heads, head_dim, added_projection=False)


class LongCatImageTransformer2DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = LONGCAT_IMAGE_TRANSFORMER_DEFAULTS | {
            name: value for name, value in config.items() if not name.startswith("_")
        }
        if sum(cfg["axes_dims_rope"]) != cfg["attention_head_dim"]:
            raise ValueError("Unsupported LongCat-Image transformer configuration")
        super().__init__(cfg)
        self.out_channels = cfg["in_channels"]
        self.inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        self.pos_embed = FluxPosEmbed(10000, list(cfg["axes_dims_rope"]))
        self.time_embed = LongCatImageTimestepEmbeddings(self.inner_dim)
        self.context_embedder = nn.Linear(cfg["joint_attention_dim"], self.inner_dim)
        self.x_embedder = nn.Linear(cfg["in_channels"], self.inner_dim)
        self.transformer_blocks = nn.ModuleList(
            LongCatImageTransformerBlock(
                self.inner_dim,
                cfg["num_attention_heads"],
                cfg["attention_head_dim"],
            )
            for _ in range(cfg["num_layers"])
        )
        self.single_transformer_blocks = nn.ModuleList(
            LongCatImageSingleTransformerBlock(
                self.inner_dim,
                cfg["num_attention_heads"],
                cfg["attention_head_dim"],
            )
            for _ in range(cfg["num_single_layers"])
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim)
        self.proj_out = nn.Linear(
            self.inner_dim,
            cfg["patch_size"] * cfg["patch_size"] * self.out_channels,
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        temb = self.time_embed(timestep, hidden_states.dtype)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        rotary_emb = self.pos_embed(torch.cat((txt_ids, img_ids), dim=0))
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states, temb, rotary_emb
            )
        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states, temb, rotary_emb
            )
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)
