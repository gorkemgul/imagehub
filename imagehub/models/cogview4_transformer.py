"""Native CogView4 transformer for text-to-image inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel


def _timestep_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
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


class CogView4TimestepProjection(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps: Tensor) -> Tensor:
        return _timestep_embedding(timesteps, self.num_channels)


class CogView4TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, embedding_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, sample: Tensor) -> Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


class CogView4ConditionEmbedding(nn.Module):
    def __init__(self, in_features: int, embedding_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, embedding_dim)
        self.act_1 = nn.SiLU()
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, condition: Tensor) -> Tensor:
        condition = self.linear_1(condition)
        condition = self.act_1(condition)
        return self.linear_2(condition)


class CogView4CombinedTimestepSizeEmbeddings(nn.Module):
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
        original_size: Tensor,
        target_size: Tensor,
        crop_coords: Tensor,
        hidden_dtype: torch.dtype,
    ) -> Tensor:
        timesteps_proj = self.time_proj(timestep)
        original_size_proj = self.condition_proj(original_size.flatten()).view(
            original_size.size(0), -1
        )
        crop_coords_proj = self.condition_proj(crop_coords.flatten()).view(crop_coords.size(0), -1)
        target_size_proj = self.condition_proj(target_size.flatten()).view(target_size.size(0), -1)
        condition_proj = torch.cat([original_size_proj, crop_coords_proj, target_size_proj], dim=1)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        condition_emb = self.condition_embedder(condition_proj.to(dtype=hidden_dtype))
        return timesteps_emb + condition_emb


class CogView4PatchEmbed(nn.Module):
    def __init__(
        self, in_channels: int, hidden_size: int, patch_size: int, text_hidden_size: int
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size)
        self.text_proj = nn.Linear(text_hidden_size, hidden_size)

    def forward(
        self, hidden_states: Tensor, encoder_hidden_states: Tensor
    ) -> tuple[Tensor, Tensor]:
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
        hidden_states = self.proj(hidden_states)
        encoder_hidden_states = self.text_proj(encoder_hidden_states)
        return hidden_states, encoder_hidden_states


class CogView4AdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.linear = nn.Linear(embedding_dim, 12 * dim, bias=True)

    def forward(
        self, hidden_states: Tensor, encoder_hidden_states: Tensor, temb: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        dtype = hidden_states.dtype
        norm_hidden_states = self.norm(hidden_states).to(dtype=dtype)
        norm_encoder_hidden_states = self.norm_context(encoder_hidden_states).to(dtype=dtype)
        emb = self.linear(temb)
        (
            shift_msa,
            c_shift_msa,
            scale_msa,
            c_scale_msa,
            gate_msa,
            c_gate_msa,
            shift_mlp,
            c_shift_mlp,
            scale_mlp,
            c_scale_mlp,
            gate_mlp,
            c_gate_mlp,
        ) = emb.chunk(12, dim=1)
        hidden_states = norm_hidden_states * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        encoder_hidden_states = norm_encoder_hidden_states * (
            1 + c_scale_msa.unsqueeze(1)
        ) + c_shift_msa.unsqueeze(1)
        return (
            hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        )


def _apply_rotary_emb(x: Tensor, image_rotary_emb: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = image_rotary_emb
    cos = cos[None, None, :, :].to(x.device)
    sin = sin[None, None, :, :].to(x.device)
    x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
    x_rotated = torch.cat([-x_imag, x_real], dim=-1)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class CogView4Attention(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int) -> None:
        super().__init__()
        self.heads = num_attention_heads
        inner_dim = num_attention_heads * attention_head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=True), nn.Dropout(0.0)])
        self.norm_q = nn.LayerNorm(attention_head_dim, eps=1e-5, elementwise_affine=False)
        self.norm_k = nn.LayerNorm(attention_head_dim, eps=1e-5, elementwise_affine=False)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Tensor | None,
        image_rotary_emb: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor]:
        dtype = encoder_hidden_states.dtype
        batch_size, text_seq_length, _ = encoder_hidden_states.shape
        _, image_seq_length, _ = hidden_states.shape
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        query = query.unflatten(2, (self.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (self.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (self.heads, -1)).transpose(1, 2)
        query = self.norm_q(query).to(dtype=dtype)
        key = self.norm_k(key).to(dtype=dtype)
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:, :] = _apply_rotary_emb(
                query[:, :, text_seq_length:, :], image_rotary_emb
            )
            key[:, :, text_seq_length:, :] = _apply_rotary_emb(
                key[:, :, text_seq_length:, :], image_rotary_emb
            )
        if attention_mask is not None:
            text_attn_mask = attention_mask.float().to(query.device)
            mix_attn_mask = torch.ones(
                (batch_size, text_seq_length + image_seq_length), device=query.device
            )
            mix_attn_mask[:, :text_seq_length] = text_attn_mask
            mix_attn_mask = mix_attn_mask.unsqueeze(2)
            attn_mask_matrix = mix_attn_mask @ mix_attn_mask.transpose(1, 2)
            attention_mask = (attn_mask_matrix > 0).unsqueeze(1).to(query.dtype)
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class CogView4GELU(nn.Module):
    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate="tanh")


class CogView4FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inner_dim = dim * 4
        self.net = nn.ModuleList(
            [CogView4GELU(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim)]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class CogView4TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
    ) -> None:
        super().__init__()
        self.norm1 = CogView4AdaLayerNormZero(time_embed_dim, dim)
        self.attn1 = CogView4Attention(dim, num_attention_heads, attention_head_dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.ff = CogView4FeedForward(dim)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1(hidden_states, encoder_hidden_states, temb)
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            norm_hidden_states,
            norm_encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa.unsqueeze(1)
        encoder_hidden_states = (
            encoder_hidden_states + attn_encoder_hidden_states * c_gate_msa.unsqueeze(1)
        )
        norm_hidden_states = self.norm2(hidden_states) * (
            1 + scale_mlp.unsqueeze(1)
        ) + shift_mlp.unsqueeze(1)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states) * (
            1 + c_scale_mlp.unsqueeze(1)
        ) + c_shift_mlp.unsqueeze(1)
        ff_output = self.ff(norm_hidden_states)
        ff_output_context = self.ff(norm_encoder_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + ff_output_context * c_gate_mlp.unsqueeze(1)
        return hidden_states, encoder_hidden_states


class CogView4RotaryPosEmbed(nn.Module):
    def __init__(
        self, dim: int, patch_size: int, rope_axes_dim: tuple[int, int], theta: float = 10000.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.rope_axes_dim = rope_axes_dim
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
        h_seq = torch.arange(self.rope_axes_dim[0])
        w_seq = torch.arange(self.rope_axes_dim[1])
        freqs_h = torch.outer(h_seq, h_inv_freq)
        freqs_w = torch.outer(w_seq, w_inv_freq)
        h_idx = torch.arange(height, device=freqs_h.device)
        w_idx = torch.arange(width, device=freqs_w.device)
        inner_h_idx = h_idx * self.rope_axes_dim[0] // height
        inner_w_idx = w_idx * self.rope_axes_dim[1] // width
        freqs_h = freqs_h[inner_h_idx]
        freqs_w = freqs_w[inner_w_idx]
        freqs_h = freqs_h.unsqueeze(1)
        freqs_w = freqs_w.unsqueeze(0)
        freqs_h = freqs_h.expand(height, width, -1)
        freqs_w = freqs_w.expand(height, width, -1)
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        freqs = torch.cat([freqs, freqs], dim=-1)
        freqs = freqs.reshape(height * width, -1)
        return freqs.cos(), freqs.sin()


class CogView4AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-5, elementwise_affine=False, bias=False)

    def forward(self, hidden_states: Tensor, conditioning_embedding: Tensor) -> Tensor:
        embedding = self.linear(conditioning_embedding.to(hidden_states.dtype))
        scale, shift = torch.chunk(embedding, 2, dim=1)
        return self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]


class CogView4Transformer2DModel(NativeModel):
    """CogView4 denoiser using the default sharded loading for all-bfloat16 weights."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        patch_size = int(config.get("patch_size", 2))
        in_channels = int(config.get("in_channels", 16))
        out_channels = int(config.get("out_channels", 16))
        num_layers = int(config.get("num_layers", 30))
        attention_head_dim = int(config.get("attention_head_dim", 40))
        num_attention_heads = int(config.get("num_attention_heads", 64))
        text_embed_dim = int(config.get("text_embed_dim", 4096))
        time_embed_dim = int(config.get("time_embed_dim", 512))
        condition_dim = int(config.get("condition_dim", 256))
        rope_axes_value = config.get("rope_axes_dim", (256, 256))
        if not isinstance(rope_axes_value, (list, tuple)) or len(rope_axes_value) != 2:
            raise ValueError("rope_axes_dim must contain height and width dimensions")
        rope_axes_dim = (int(rope_axes_value[0]), int(rope_axes_value[1]))
        pooled_projection_dim = 3 * 2 * condition_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.rope = CogView4RotaryPosEmbed(
            attention_head_dim, patch_size, rope_axes_dim, theta=10000.0
        )
        self.patch_embed = CogView4PatchEmbed(in_channels, inner_dim, patch_size, text_embed_dim)
        self.time_condition_embed = CogView4CombinedTimestepSizeEmbeddings(
            embedding_dim=time_embed_dim,
            condition_dim=condition_dim,
            pooled_projection_dim=pooled_projection_dim,
            timesteps_dim=inner_dim,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                CogView4TransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    time_embed_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = CogView4AdaLayerNormContinuous(inner_dim, time_embed_dim)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels, bias=True)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        original_size: Tensor,
        target_size: Tensor,
        crop_coords: Tensor,
        *,
        attention_mask: Tensor | None = None,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        batch_size, _, height, width = hidden_states.shape
        if image_rotary_emb is None:
            image_rotary_emb = self.rope(hidden_states)
        post_patch_height = height // self.patch_size
        post_patch_width = width // self.patch_size
        hidden_states, encoder_hidden_states = self.patch_embed(
            hidden_states, encoder_hidden_states
        )
        temb = self.time_condition_embed(
            timestep,
            original_size,
            target_size,
            crop_coords,
            hidden_states.dtype,
        )
        temb = F.silu(temb)
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                attention_mask,
            )
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_height,
            post_patch_width,
            -1,
            self.patch_size,
            self.patch_size,
        )
        return hidden_states.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
