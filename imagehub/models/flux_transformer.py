"""Native FLUX.1 transformer for text-to-image inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

FLUX_TRANSFORMER_DEFAULTS: dict[str, Any] = {
    "patch_size": 1,
    "in_channels": 64,
    "out_channels": None,
    "num_layers": 19,
    "num_single_layers": 38,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "guidance_embeds": False,
    "axes_dims_rope": (16, 56, 56),
}


def _get_timestep_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / half_dim
    embedding = torch.exp(exponent)
    embedding = timesteps[:, None].float() * embedding[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
    embedding = torch.cat([embedding[:, half_dim:], embedding[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1, 0, 0))
    return embedding


def _apply_rotary_emb(hidden_states: Tensor, rotary_emb: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = rotary_emb
    cos = cos[None, :, None, :].to(hidden_states.device)
    sin = sin[None, :, None, :].to(hidden_states.device)
    real, imaginary = hidden_states.reshape(*hidden_states.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([-imaginary, real], dim=-1).flatten(3)
    return (hidden_states.float() * cos + rotated.float() * sin).to(hidden_states.dtype)


def _attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    query, key, value = (item.permute(0, 2, 1, 3) for item in (query, key, value))
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False
    )
    return hidden_states.permute(0, 2, 1, 3)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, embedding_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, sample: Tensor) -> Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


class TextProjection(nn.Module):
    def __init__(self, in_features: int, embedding_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, embedding_dim)
        self.act_1 = nn.SiLU()
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_1(hidden_states)
        return self.linear_2(hidden_states)


class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, pooled_projection_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)
        self.text_embedder = TextProjection(pooled_projection_dim, embedding_dim)

    def forward(self, timestep: Tensor, pooled_projection: Tensor) -> Tensor:
        timestep_projection = _get_timestep_embedding(timestep, 256)
        timestep_embedding = self.timestep_embedder(
            timestep_projection.to(dtype=pooled_projection.dtype)
        )
        pooled_projections = self.text_embedder(pooled_projection)
        return timestep_embedding + pooled_projections


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, pooled_projection_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)
        self.guidance_embedder = TimestepEmbedding(256, embedding_dim)
        self.text_embedder = TextProjection(pooled_projection_dim, embedding_dim)

    def forward(self, timestep: Tensor, guidance: Tensor, pooled_projection: Tensor) -> Tensor:
        timestep_projection = _get_timestep_embedding(timestep, 256)
        timestep_embedding = self.timestep_embedder(
            timestep_projection.to(dtype=pooled_projection.dtype)
        )
        guidance_projection = _get_timestep_embedding(guidance, 256)
        guidance_embedding = self.guidance_embedder(
            guidance_projection.to(dtype=pooled_projection.dtype)
        )
        time_guidance_embedding = timestep_embedding + guidance_embedding
        pooled_projections = self.text_embedder(pooled_projection)
        return time_guidance_embedding + pooled_projections


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 6 * dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, hidden_states: Tensor, embedding: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        embedding = self.linear(self.silu(embedding))
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = embedding.chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return hidden_states, gate, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 3 * dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: Tensor, embedding: Tensor) -> tuple[Tensor, Tensor]:
        embedding = self.linear(self.silu(embedding))
        shift, scale, gate = embedding.chunk(3, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return hidden_states, gate


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)

    def forward(self, hidden_states: Tensor, conditioning: Tensor) -> Tensor:
        embedding = self.linear(self.silu(conditioning).to(hidden_states.dtype))
        scale, shift = torch.chunk(embedding, 2, dim=1)
        return self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]


class GELU(nn.Module):
    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.gelu(self.proj(hidden_states), approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList([GELU(dim, dim * 4), nn.Dropout(0.0), nn.Linear(dim * 4, dim)])

    def forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class FluxAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, *, added_projection: bool) -> None:
        super().__init__()
        self.head_dim = head_dim
        inner_dim = heads * head_dim
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        if added_projection:
            self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim), nn.Dropout(0.0)])
            self.norm_added_q = nn.RMSNorm(head_dim, eps=1e-6)
            self.norm_added_k = nn.RMSNorm(head_dim, eps=1e-6)
            self.add_q_proj = nn.Linear(dim, inner_dim)
            self.add_k_proj = nn.Linear(dim, inner_dim)
            self.add_v_proj = nn.Linear(dim, inner_dim)
            self.to_add_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        hidden_states: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
        encoder_hidden_states: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        query = self.to_q(hidden_states).unflatten(-1, (-1, self.head_dim))
        key = self.to_k(hidden_states).unflatten(-1, (-1, self.head_dim))
        value = self.to_v(hidden_states).unflatten(-1, (-1, self.head_dim))
        query = self.norm_q(query)
        key = self.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_query = self.add_q_proj(encoder_hidden_states).unflatten(
                -1, (-1, self.head_dim)
            )
            encoder_key = self.add_k_proj(encoder_hidden_states).unflatten(-1, (-1, self.head_dim))
            encoder_value = self.add_v_proj(encoder_hidden_states).unflatten(
                -1, (-1, self.head_dim)
            )
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
        output = self.to_out[1](self.to_out[0](output.contiguous()))
        encoder_output = self.to_add_out(encoder_output.contiguous())
        return output, encoder_output


class FluxTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)
        self.attn = FluxAttention(dim, heads, head_dim, added_projection=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        normalized, gate, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        (
            normalized_context,
            context_gate,
            context_shift_mlp,
            context_scale_mlp,
            context_gate_mlp,
        ) = self.norm1_context(encoder_hidden_states, temb)
        attention_output, context_attention_output = self.attn(
            normalized, rotary_emb, normalized_context
        )
        hidden_states = hidden_states + gate.unsqueeze(1) * attention_output
        normalized = self.norm2(hidden_states)
        normalized = normalized * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(normalized)

        encoder_hidden_states = (
            encoder_hidden_states + context_gate.unsqueeze(1) * context_attention_output
        )
        normalized_context = self.norm2_context(encoder_hidden_states)
        normalized_context = (
            normalized_context * (1 + context_scale_mlp[:, None]) + context_shift_mlp[:, None]
        )
        encoder_hidden_states = encoder_hidden_states + context_gate_mlp.unsqueeze(
            1
        ) * self.ff_context(normalized_context)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, dim * 4)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim * 5, dim)
        self.attn = FluxAttention(dim, heads, head_dim, added_projection=False)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        text_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states
        normalized, gate = self.norm(hidden_states, temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(normalized))
        attention_output = self.attn(normalized, rotary_emb)
        hidden_states = torch.cat([attention_output, mlp_hidden_states], dim=2)
        hidden_states = residual + gate.unsqueeze(1) * self.proj_out(hidden_states)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states[:, :text_length], hidden_states[:, text_length:]


class FluxPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        cosines = []
        sines = []
        positions = ids.float()
        for index, dim in enumerate(self.axes_dim):
            frequencies = 1.0 / (
                self.theta
                ** (torch.arange(0, dim, 2, dtype=torch.float64, device=ids.device) / dim)
            )
            frequencies = torch.outer(positions[:, index], frequencies)
            cosines.append(frequencies.cos().repeat_interleave(2, dim=1).float())
            sines.append(frequencies.sin().repeat_interleave(2, dim=1).float())
        return torch.cat(cosines, dim=-1), torch.cat(sines, dim=-1)


class FluxTransformer2DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = FLUX_TRANSFORMER_DEFAULTS | {
            name: value for name, value in config.items() if not name.startswith("_")
        }
        if sum(cfg["axes_dims_rope"]) != cfg["attention_head_dim"]:
            raise ValueError("Unsupported FLUX.1 transformer configuration")
        super().__init__(cfg)
        self.out_channels = cfg["out_channels"] or cfg["in_channels"]
        self.inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        self.pos_embed = FluxPosEmbed(10000, list(cfg["axes_dims_rope"]))
        embedding_class = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if cfg["guidance_embeds"]
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = embedding_class(self.inner_dim, cfg["pooled_projection_dim"])
        self.context_embedder = nn.Linear(cfg["joint_attention_dim"], self.inner_dim)
        self.x_embedder = nn.Linear(cfg["in_channels"], self.inner_dim)
        self.transformer_blocks = nn.ModuleList(
            FluxTransformerBlock(
                self.inner_dim,
                cfg["num_attention_heads"],
                cfg["attention_head_dim"],
            )
            for _ in range(cfg["num_layers"])
        )
        self.single_transformer_blocks = nn.ModuleList(
            FluxSingleTransformerBlock(
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
        pooled_projections: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor | None = None,
        *,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        if joint_attention_kwargs:
            raise ValueError("Unsupported FLUX.1 transformer option")
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        if guidance is None:
            temb = self.time_text_embed(timestep, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, guidance, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
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
