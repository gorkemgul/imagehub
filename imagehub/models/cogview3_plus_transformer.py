"""Native CogView3-Plus transformer for text-to-image inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.cogview4_transformer import CogView4CombinedTimestepSizeEmbeddings

_DEFAULTS: dict[str, Any] = {
    "patch_size": 2,
    "in_channels": 16,
    "num_layers": 30,
    "attention_head_dim": 40,
    "num_attention_heads": 64,
    "out_channels": 16,
    "text_embed_dim": 4096,
    "time_embed_dim": 512,
    "condition_dim": 256,
    "pos_embed_max_size": 128,
    "sample_size": 128,
}
# Keys the published configuration carries that the model never reads.
_IGNORED_KEYS = {"pooled_projection_dim"}


def _sincos_1d(embed_dim: int, position: Tensor) -> Tensor:
    omega = torch.arange(embed_dim // 2, device=position.device, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    values = torch.outer(position.reshape(-1), omega)
    return torch.cat([torch.sin(values), torch.cos(values)], dim=1)


def _sincos_2d(embed_dim: int, grid_size: int, base_size: int) -> Tensor:
    if embed_dim % 2:
        raise ValueError("embed_dim must be divisible by 2")
    grid_h = torch.arange(grid_size, dtype=torch.float32) / (grid_size / base_size)
    grid_w = torch.arange(grid_size, dtype=torch.float32) / (grid_size / base_size)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing="xy"), dim=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    return torch.cat(
        [_sincos_1d(embed_dim // 2, grid[0]), _sincos_1d(embed_dim // 2, grid[1])], dim=1
    )


class CogView3PlusPatchEmbed(nn.Module):
    pos_embed: Tensor

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
        text_hidden_size: int,
        pos_embed_max_size: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size
        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size)
        self.text_proj = nn.Linear(text_hidden_size, hidden_size)
        self.register_buffer("pos_embed", self._position(), persistent=False)

    def _position(self) -> Tensor:
        size = self.pos_embed_max_size
        position = _sincos_2d(self.hidden_size, size, size)
        return position.reshape(size, size, self.hidden_size).float()

    def init_pos_embed(self) -> None:
        self.pos_embed = self._position()

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor) -> Tensor:
        batch_size, channel, height, width = hidden_states.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Height and width must be divisible by patch size")
        height = height // self.patch_size
        width = width // self.patch_size
        hidden_states = hidden_states.view(
            batch_size, channel, height, self.patch_size, width, self.patch_size
        )
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5).contiguous()
        hidden_states = hidden_states.view(
            batch_size, height * width, channel * self.patch_size * self.patch_size
        )
        hidden_states = self.proj(hidden_states)
        encoder_hidden_states = self.text_proj(encoder_hidden_states)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        text_length = encoder_hidden_states.shape[1]
        image_pos_embed = self.pos_embed[:height, :width].reshape(height * width, -1)
        text_pos_embed = torch.zeros(
            (text_length, self.hidden_size),
            dtype=image_pos_embed.dtype,
            device=image_pos_embed.device,
        )
        pos_embed = torch.cat([text_pos_embed, image_pos_embed], dim=0)[None, ...]
        return (hidden_states + pos_embed).to(hidden_states.dtype)


class CogView3PlusAdaLayerNormZeroTextImage(nn.Module):
    def __init__(self, embedding_dim: int, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 12 * dim, bias=True)
        self.norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm_c = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)

    def forward(
        self, x: Tensor, context: Tensor, emb: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        emb = self.linear(self.silu(emb))
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = emb.chunk(12, dim=1)
        normed_x = self.norm_x(x)
        normed_context = self.norm_c(context)
        x = normed_x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        context = normed_context * (1 + c_scale_msa[:, None]) + c_shift_msa[:, None]
        return (
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            context,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        )


class CogView3PlusAttention(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int) -> None:
        super().__init__()
        self.heads = num_attention_heads
        inner_dim = num_attention_heads * attention_head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=True), nn.Dropout(0.0)])
        self.norm_q = nn.LayerNorm(attention_head_dim, eps=1e-6, elementwise_affine=False)
        self.norm_k = nn.LayerNorm(attention_head_dim, eps=1e-6, elementwise_affine=False)

    def forward(
        self, hidden_states: Tensor, encoder_hidden_states: Tensor
    ) -> tuple[Tensor, Tensor]:
        text_seq_length = encoder_hidden_states.size(1)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size = hidden_states.shape[0]
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        head_dim = key.shape[-1] // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        query = self.norm_q(query)
        key = self.norm_k(key)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class CogView3PlusGELU(nn.Module):
    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.gelu(self.proj(hidden_states), approximate="tanh")


class CogView3PlusFeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inner_dim = dim * 4
        self.net = nn.ModuleList(
            [CogView3PlusGELU(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim)]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class CogView3PlusTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, time_embed_dim: int
    ) -> None:
        super().__init__()
        self.norm1 = CogView3PlusAdaLayerNormZeroTextImage(time_embed_dim, dim)
        self.attn1 = CogView3PlusAttention(dim, num_attention_heads, attention_head_dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.ff = CogView3PlusFeedForward(dim)

    def forward(
        self, hidden_states: Tensor, encoder_hidden_states: Tensor, emb: Tensor
    ) -> tuple[Tensor, Tensor]:
        text_seq_length = encoder_hidden_states.size(1)
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
        ) = self.norm1(hidden_states, encoder_hidden_states, emb)
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            norm_hidden_states, norm_encoder_hidden_states
        )
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_hidden_states
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_msa.unsqueeze(1) * attn_encoder_hidden_states
        )
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output[:, text_seq_length:]
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * ff_output[:, :text_seq_length]
        )
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return hidden_states, encoder_hidden_states


class CogView3PlusAdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x: Tensor, conditioning_embedding: Tensor) -> Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class CogView3PlusTransformer2DModel(NativeModel):
    """CogView3-Plus denoiser; the published weights are all bfloat16."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        unknown = sorted(
            key
            for key in config
            if not key.startswith("_") and key not in _DEFAULTS and key not in _IGNORED_KEYS
        )
        if unknown:
            raise ValueError(f"Unsupported CogView3-Plus transformer setting {unknown[0]}")
        cfg = _DEFAULTS | {key: config[key] for key in _DEFAULTS if key in config}
        patch_size = int(cfg["patch_size"])
        num_heads = int(cfg["num_attention_heads"])
        head_dim = int(cfg["attention_head_dim"])
        time_embed_dim = int(cfg["time_embed_dim"])
        condition_dim = int(cfg["condition_dim"])
        inner_dim = num_heads * head_dim
        self.model_config = cfg
        self.patch_size = patch_size
        self.out_channels = int(cfg["out_channels"])
        self.inner_dim = inner_dim
        self.patch_embed = CogView3PlusPatchEmbed(
            int(cfg["in_channels"]),
            inner_dim,
            patch_size,
            int(cfg["text_embed_dim"]),
            int(cfg["pos_embed_max_size"]),
        )
        self.time_condition_embed = CogView4CombinedTimestepSizeEmbeddings(
            embedding_dim=time_embed_dim,
            condition_dim=condition_dim,
            pooled_projection_dim=3 * 2 * condition_dim,
            timesteps_dim=inner_dim,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                CogView3PlusTransformerBlock(inner_dim, num_heads, head_dim, time_embed_dim)
                for _ in range(int(cfg["num_layers"]))
            ]
        )
        self.norm_out = CogView3PlusAdaLayerNormContinuous(inner_dim, time_embed_dim)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

    def init_non_persistent_buffers(self) -> None:
        self.patch_embed.init_pos_embed()

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        original_size: Tensor,
        target_size: Tensor,
        crop_coords: Tensor,
    ) -> Tensor:
        height, width = hidden_states.shape[-2:]
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = self.patch_embed(hidden_states, encoder_hidden_states)
        emb = self.time_condition_embed(
            timestep, original_size, target_size, crop_coords, hidden_states.dtype
        )
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, emb)
        hidden_states = self.norm_out(hidden_states, emb)
        hidden_states = self.proj_out(hidden_states)
        height = height // self.patch_size
        width = width // self.patch_size
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], height, width, self.out_channels, self.patch_size, self.patch_size
        )
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
        return hidden_states.reshape(
            hidden_states.shape[0], self.out_channels, height * self.patch_size, width * self.patch_size
        )
