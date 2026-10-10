"""Native GLM-Image diffusion transformer for text-to-image inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.cogview4_transformer import (
    CogView4AdaLayerNormContinuous,
    CogView4ConditionEmbedding,
    CogView4TimestepEmbedding,
    CogView4TimestepProjection,
    CogView4TransformerBlock,
)


class GlmImageCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        condition_dim: int,
        pooled_projection_dim: int,
        timesteps_dim: int,
    ) -> None:
        super().__init__()
        self.time_proj = CogView4TimestepProjection(timesteps_dim)
        self.condition_proj = CogView4TimestepProjection(condition_dim)
        self.timestep_embedder = CogView4TimestepEmbedding(timesteps_dim, embedding_dim)
        self.condition_embedder = CogView4ConditionEmbedding(pooled_projection_dim, embedding_dim)

    def forward(
        self,
        timestep: Tensor,
        target_size: Tensor,
        crop_coords: Tensor,
        hidden_dtype: torch.dtype,
    ) -> Tensor:
        timesteps_proj = self.time_proj(timestep)
        crop_coords_proj = self.condition_proj(crop_coords.flatten()).view(crop_coords.size(0), -1)
        target_size_proj = self.condition_proj(target_size.flatten()).view(target_size.size(0), -1)
        condition_proj = torch.cat([crop_coords_proj, target_size_proj], dim=1)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        condition_emb = self.condition_embedder(condition_proj.to(dtype=hidden_dtype))
        return F.silu(timesteps_emb + condition_emb)


class GlmImageImageProjector(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, channel, height, width = hidden_states.shape
        post_patch_height = height // self.patch_size
        post_patch_width = width // self.patch_size
        hidden_states = hidden_states.reshape(
            batch_size,
            channel,
            post_patch_height,
            self.patch_size,
            post_patch_width,
            self.patch_size,
        )
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        return self.proj(hidden_states)


class GlmImageProjection(nn.Module):
    """Linear, activation, linear with the published ``net.0.proj`` / ``net.2`` names."""

    def __init__(self, dim_in: int, inner_dim: int, dim_out: int, activation: str) -> None:
        super().__init__()
        if activation not in ("gelu", "silu"):
            raise ValueError(f"Unsupported projection activation {activation!r}")
        self.activation = activation
        self.net = nn.ModuleList(
            [_ProjectIn(dim_in, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim_out)]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.net[0](hidden_states)
        hidden_states = (
            F.gelu(hidden_states) if self.activation == "gelu" else F.silu(hidden_states)
        )
        return self.net[2](hidden_states)


class _ProjectIn(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.proj(hidden_states)


class GlmImageRotaryPosEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.theta = theta

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        _, _, height, width = hidden_states.shape
        height, width = height // self.patch_size, width // self.patch_size
        dim_h, dim_w = self.dim // 2, self.dim // 2
        h_inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, dim_h, 2, dtype=torch.float32)[: dim_h // 2].float() / dim_h)
        )
        w_inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, dim_w, 2, dtype=torch.float32)[: dim_w // 2].float() / dim_w)
        )
        freqs_h = torch.outer(torch.arange(height), h_inv_freq)
        freqs_w = torch.outer(torch.arange(width), w_inv_freq)
        freqs_h = freqs_h.unsqueeze(1).expand(height, width, -1)
        freqs_w = freqs_w.unsqueeze(0).expand(height, width, -1)
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        freqs = torch.cat([freqs, freqs], dim=-1)
        freqs = freqs.reshape(height * width, -1)
        return freqs.cos(), freqs.sin()


class GlmImageTransformer2DModel(NativeModel):
    """GLM-Image decoder: denoises VAE latents given prior image tokens and glyph embeddings."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        patch_size = int(config.get("patch_size", 2))
        in_channels = int(config.get("in_channels", 16))
        out_channels = int(config.get("out_channels", 16))
        num_layers = int(config.get("num_layers", 30))
        attention_head_dim = int(config.get("attention_head_dim", 40))
        num_attention_heads = int(config.get("num_attention_heads", 64))
        text_embed_dim = int(config.get("text_embed_dim", 1472))
        time_embed_dim = int(config.get("time_embed_dim", 512))
        condition_dim = int(config.get("condition_dim", 256))
        codebook_size = int(config.get("prior_vq_quantizer_codebook_size", 16384))
        inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.rope = GlmImageRotaryPosEmbed(attention_head_dim, patch_size, theta=10000.0)
        self.image_projector = GlmImageImageProjector(in_channels, inner_dim, patch_size)
        self.glyph_projector = GlmImageProjection(text_embed_dim, inner_dim, inner_dim, "gelu")
        self.prior_token_embedding = nn.Embedding(codebook_size, inner_dim)
        self.prior_projector = GlmImageProjection(inner_dim, inner_dim, inner_dim, "silu")
        self.time_condition_embed = GlmImageCombinedTimestepSizeEmbeddings(
            embedding_dim=time_embed_dim,
            condition_dim=condition_dim,
            pooled_projection_dim=2 * 2 * condition_dim,
            timesteps_dim=time_embed_dim,
        )
        self.transformer_blocks = nn.ModuleList(
            CogView4TransformerBlock(
                inner_dim, num_attention_heads, attention_head_dim, time_embed_dim
            )
            for _ in range(num_layers)
        )
        self.norm_out = CogView4AdaLayerNormContinuous(inner_dim, time_embed_dim)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels, bias=True)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        prior_token_id: Tensor,
        prior_token_drop: Tensor,
        timestep: Tensor,
        target_size: Tensor,
        crop_coords: Tensor,
    ) -> Tensor:
        batch_size, _, height, width = hidden_states.shape
        image_rotary_emb = self.rope(hidden_states)
        post_patch_height = height // self.patch_size
        post_patch_width = width // self.patch_size
        hidden_states = self.image_projector(hidden_states)
        encoder_hidden_states = self.glyph_projector(encoder_hidden_states)
        prior_embedding = self.prior_token_embedding(prior_token_id)
        prior_embedding[prior_token_drop] *= 0.0
        hidden_states = hidden_states + self.prior_projector(prior_embedding)
        temb = self.time_condition_embed(timestep, target_size, crop_coords, hidden_states.dtype)
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, image_rotary_emb, None
            )
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width, -1, self.patch_size, self.patch_size
        )
        return hidden_states.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
