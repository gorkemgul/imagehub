"""Native AuraFlow transformer (MMDiT blocks followed by single-stream DiT blocks)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

AURAFLOW_DEFAULTS: dict[str, Any] = {
    "sample_size": 64,
    "patch_size": 2,
    "in_channels": 4,
    "num_mmdit_layers": 4,
    "num_single_dit_layers": 32,
    "attention_head_dim": 256,
    "num_attention_heads": 12,
    "joint_attention_dim": 2048,
    "caption_projection_dim": 3072,
    "out_channels": 4,
    "pos_embed_max_size": 1024,
}
_EPS = 1e-5


def _find_multiple(n: int, k: int) -> int:
    return n if n % k == 0 else n + k - (n % k)


class FP32LayerNorm(nn.LayerNorm):
    """Layer norm computed in float32 without affine parameters."""

    def forward(self, inputs: Tensor) -> Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


def _timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Sinusoidal features with ``scale=1000``, ``flip_sin_to_cos`` and no frequency shift."""
    half_dim = dim // 2
    exponent = -math.log(10000) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / half_dim
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = 1000 * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: Tensor) -> Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class PatchEmbed(nn.Module):
    """Linear patch projection with a learned grid of positions, centre-cropped to the input."""

    def __init__(
        self, patch_size: int, in_channels: int, embed_dim: int, pos_embed_max_size: int
    ) -> None:
        super().__init__()
        self.pos_embed_max_size = pos_embed_max_size
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_max_size, embed_dim))

    def selection_index(self, height: int, width: int) -> Tensor:
        h_p, w_p = height // self.patch_size, width // self.patch_size
        h_max = w_max = int(self.pos_embed_max_size**0.5)
        if h_p > h_max or w_p > w_max:
            raise ValueError(
                f"The latent has {h_p}x{w_p} patches; the positional grid is {h_max}x{w_max}"
            )
        start_h = h_max // 2 - h_p // 2
        start_w = w_max // 2 - w_p // 2
        rows = torch.arange(start_h, start_h + h_p, device=self.pos_embed.device)
        cols = torch.arange(start_w, start_w + w_p, device=self.pos_embed.device)
        row_indices, col_indices = torch.meshgrid(rows, cols, indexing="ij")
        return (row_indices * w_max + col_indices).flatten()

    def forward(self, latent: Tensor) -> Tensor:
        batch_size, num_channels, height, width = latent.size()
        latent = latent.view(
            batch_size,
            num_channels,
            height // self.patch_size,
            self.patch_size,
            width // self.patch_size,
            self.patch_size,
        )
        latent = latent.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        latent = self.proj(latent)
        return latent + self.pos_embed[:, self.selection_index(height, width)]


class FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = _find_multiple(int(2 * (4 * dim) / 3), 256)
        self.linear_1 = nn.Linear(dim, hidden, bias=False)
        self.linear_2 = nn.Linear(dim, hidden, bias=False)
        self.out_projection = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_projection(F.silu(self.linear_1(x)) * self.linear_2(x))


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 6 * dim, bias=False)
        self.norm = FP32LayerNorm(dim, elementwise_affine=False, bias=False)

    def forward(self, x: Tensor, emb: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        emb = self.linear(F.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class Attention(nn.Module):
    """Self-attention, or joint attention when the block also has a text stream."""

    def __init__(self, dim: int, heads: int, head_dim: int, *, joint: bool) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        inner_dim = heads * head_dim
        self.norm_q = FP32LayerNorm(head_dim, elementwise_affine=False, bias=False, eps=_EPS)
        self.norm_k = FP32LayerNorm(head_dim, elementwise_affine=False, bias=False, eps=_EPS)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=False), nn.Dropout(0.0)])
        self.joint = joint
        if joint:
            self.add_q_proj = nn.Linear(dim, inner_dim, bias=False)
            self.add_k_proj = nn.Linear(dim, inner_dim, bias=False)
            self.add_v_proj = nn.Linear(dim, inner_dim, bias=False)
            self.norm_added_q = FP32LayerNorm(
                head_dim, elementwise_affine=False, bias=False, eps=_EPS
            )
            self.norm_added_k = FP32LayerNorm(
                head_dim, elementwise_affine=False, bias=False, eps=_EPS
            )
            self.to_add_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self, hidden_states: Tensor, encoder_hidden_states: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        batch_size = hidden_states.shape[0]
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        if encoder_hidden_states is not None:
            context_query = self.add_q_proj(encoder_hidden_states)
            context_key = self.add_k_proj(encoder_hidden_states)
            context_value = self.add_v_proj(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim)
        key = key.view(batch_size, -1, self.heads, head_dim)
        value = value.view(batch_size, -1, self.heads, head_dim)
        query = self.norm_q(query)
        key = self.norm_k(key)
        if encoder_hidden_states is not None:
            context_query = context_query.view(batch_size, -1, self.heads, head_dim)
            context_key = context_key.view(batch_size, -1, self.heads, head_dim)
            context_value = context_value.view(batch_size, -1, self.heads, head_dim)
            context_query = self.norm_added_q(context_query)
            context_key = self.norm_added_k(context_key)
            query = torch.cat([context_query, query], dim=1)
            key = torch.cat([context_key, key], dim=1)
            value = torch.cat([context_value, value], dim=1)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, scale=self.scale, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            context_length = encoder_hidden_states.shape[1]
            hidden_states, encoder_hidden_states = (
                hidden_states[:, context_length:],
                hidden_states[:, :context_length],
            )
        hidden_states = self.to_out[1](self.to_out[0](hidden_states))
        if encoder_hidden_states is not None:
            return hidden_states, self.to_add_out(encoder_hidden_states)
        return hidden_states


class SingleTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim)
        self.attn = Attention(dim, heads, head_dim, joint=False)
        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.ff = FeedForward(dim)

    def forward(self, hidden_states: Tensor, temb: Tensor) -> Tensor:
        residual = hidden_states
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        attn_output = self.attn(norm_hidden_states)
        assert isinstance(attn_output, Tensor)
        hidden_states = self.norm2(residual + gate_msa.unsqueeze(1) * attn_output)
        hidden_states = hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(hidden_states)
        hidden_states = gate_mlp.unsqueeze(1) * ff_output
        return residual + hidden_states


class JointTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)
        self.attn = Attention(dim, heads, head_dim, joint=True)
        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.ff = FeedForward(dim)
        self.norm2_context = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.ff_context = FeedForward(dim)

    def forward(
        self, hidden_states: Tensor, encoder_hidden_states: Tensor, temb: Tensor
    ) -> tuple[Tensor, Tensor]:
        residual = hidden_states
        residual_context = encoder_hidden_states
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        attn_output, context_attn_output = self.attn(
            norm_hidden_states, encoder_hidden_states=norm_encoder
        )
        hidden_states = self.norm2(residual + gate_msa.unsqueeze(1) * attn_output)
        hidden_states = hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = gate_mlp.unsqueeze(1) * self.ff(hidden_states)
        hidden_states = residual + hidden_states

        encoder_hidden_states = self.norm2_context(
            residual_context + c_gate_msa.unsqueeze(1) * context_attn_output
        )
        encoder_hidden_states = (
            encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )
        encoder_hidden_states = c_gate_mlp.unsqueeze(1) * self.ff_context(encoder_hidden_states)
        encoder_hidden_states = residual_context + encoder_hidden_states
        return encoder_hidden_states, hidden_states


class PreFinalBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x: Tensor, conditioning_embedding: Tensor) -> Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return x * (1 + scale)[:, None, :] + shift[:, None, :]


class AuraFlowTransformer2DModel(NativeModel):
    """AuraFlow transformer with the published tensor names."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in AURAFLOW_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported AuraFlow transformer setting {unknown[0]}")
        cfg = AURAFLOW_DEFAULTS | {k: v for k, v in config.items() if not k.startswith("_")}
        if cfg["out_channels"] is None:
            cfg["out_channels"] = cfg["in_channels"]
        self.config = cfg
        self.patch_size = int(cfg["patch_size"])
        self.out_channels = int(cfg["out_channels"])
        inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        self.inner_dim = inner_dim
        self.pos_embed = PatchEmbed(
            self.patch_size, cfg["in_channels"], inner_dim, cfg["pos_embed_max_size"]
        )
        self.context_embedder = nn.Linear(
            cfg["joint_attention_dim"], cfg["caption_projection_dim"], bias=False
        )
        self.time_step_proj = TimestepEmbedding(256, inner_dim)
        self.joint_transformer_blocks = nn.ModuleList(
            JointTransformerBlock(inner_dim, cfg["num_attention_heads"], cfg["attention_head_dim"])
            for _ in range(cfg["num_mmdit_layers"])
        )
        self.single_transformer_blocks = nn.ModuleList(
            SingleTransformerBlock(inner_dim, cfg["num_attention_heads"], cfg["attention_head_dim"])
            for _ in range(cfg["num_single_dit_layers"])
        )
        self.norm_out = PreFinalBlock(inner_dim)
        self.proj_out = nn.Linear(
            inner_dim, self.patch_size * self.patch_size * self.out_channels, bias=False
        )
        self.register_tokens = nn.Parameter(torch.zeros(1, 8, inner_dim))

    def forward(
        self, hidden_states: Tensor, encoder_hidden_states: Tensor, timestep: Tensor
    ) -> Tensor:
        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)
        temb = _timestep_embedding(timestep, 256).to(dtype=self.pos_embed.pos_embed.dtype)
        temb = self.time_step_proj(temb)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        register_tokens = self.register_tokens.to(
            device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype
        )
        encoder_hidden_states = torch.cat(
            [register_tokens.repeat(encoder_hidden_states.size(0), 1, 1), encoder_hidden_states],
            dim=1,
        )
        for joint_block in self.joint_transformer_blocks:
            encoder_hidden_states, hidden_states = joint_block(
                hidden_states, encoder_hidden_states, temb
            )
        if len(self.single_transformer_blocks) > 0:
            encoder_seq_len = encoder_hidden_states.size(1)
            combined = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            for single_block in self.single_transformer_blocks:
                combined = single_block(combined, temb)
            hidden_states = combined[:, encoder_seq_len:]
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        patch = self.patch_size
        height = height // patch
        width = width // patch
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], height, width, patch, patch, self.out_channels
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(
            hidden_states.shape[0], self.out_channels, height * patch, width * patch
        )
