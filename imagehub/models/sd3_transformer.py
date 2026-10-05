"""Native Stable Diffusion 3 transformer (MMDiT and MMDiT-X)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

SD3_TRANSFORMER_DEFAULTS: dict[str, Any] = {
    "sample_size": 128,
    "patch_size": 2,
    "in_channels": 16,
    "num_layers": 18,
    "attention_head_dim": 64,
    "num_attention_heads": 18,
    "joint_attention_dim": 4096,
    "caption_projection_dim": 1152,
    "pooled_projection_dim": 2048,
    "out_channels": 16,
    "pos_embed_max_size": 96,
    "dual_attention_layers": (),
    "qk_norm": None,
}


def _get_1d_sincos_pos_embed(embed_dim: int, position: Tensor) -> Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")
    omega = torch.arange(embed_dim // 2, device=position.device, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    position = position.reshape(-1)
    values = torch.outer(position, omega)
    return torch.concat([torch.sin(values), torch.cos(values)], dim=1)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, base_size: int) -> Tensor:
    grid_h = torch.arange(grid_size, dtype=torch.float32) / (grid_size / base_size)
    grid_w = torch.arange(grid_size, dtype=torch.float32) / (grid_size / base_size)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing="xy"), dim=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    embed_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    embed_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return torch.concat([embed_h, embed_w], dim=1)


class PatchEmbed(nn.Module):
    pos_embed: Tensor

    def __init__(
        self,
        height: int,
        width: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        pos_embed_max_size: int,
    ) -> None:
        super().__init__()
        self.pos_embed_max_size = pos_embed_max_size
        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=True,
        )
        pos_embed = _get_2d_sincos_pos_embed(
            embed_dim, pos_embed_max_size, base_size=height // patch_size
        )
        self.register_buffer("pos_embed", pos_embed.float().unsqueeze(0), persistent=True)

    def cropped_pos_embed(self, height: int, width: int) -> Tensor:
        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: "
                f"{self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: "
                f"{self.pos_embed_max_size}."
            )
        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial = spatial[:, top : top + height, left : left + width, :]
        return spatial.reshape(1, -1, spatial.shape[-1])

    def forward(self, latent: Tensor) -> Tensor:
        height, width = latent.shape[-2:]
        latent = self.proj(latent)
        latent = latent.flatten(2).transpose(1, 2)
        pos_embed = self.cropped_pos_embed(height, width)
        return (latent + pos_embed).to(latent.dtype)


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


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: Tensor) -> Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


class PixArtAlphaTextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        self.act_1 = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, caption: Tensor) -> Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        return self.linear_2(hidden_states)


class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, pooled_projection_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim)

    def forward(self, timestep: Tensor, pooled_projection: Tensor) -> Tensor:
        timesteps_proj = _get_timestep_embedding(timestep, 256)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight


class AdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: Tensor, emb: Tensor) -> tuple[Tensor, ...]:
        projected = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = projected.chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


class SD35AdaLayerNormZeroX(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 9 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: Tensor, emb: Tensor) -> tuple[Tensor, ...]:
        projected = self.linear(self.silu(emb))
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_msa2,
            scale_msa2,
            gate_msa2,
        ) = projected.chunk(9, dim=1)
        normalized = self.norm(hidden_states)
        hidden_states = normalized * (1 + scale_msa[:, None]) + shift_msa[:, None]
        normalized2 = normalized * (1 + scale_msa2[:, None]) + shift_msa2[:, None]
        return (
            hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            normalized2,
            gate_msa2,
        )


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: Tensor, conditioning_embedding: Tensor) -> Tensor:
        embedding = self.linear(self.silu(conditioning_embedding).to(hidden_states.dtype))
        scale, shift = torch.chunk(embedding, 2, dim=1)
        return self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        *,
        context_pre_only: bool | None = None,
        added_projection: bool = False,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.context_pre_only = context_pre_only
        inner_dim = heads * head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)
        if added_projection:
            self.add_k_proj = nn.Linear(dim, inner_dim, bias=True)
            self.add_v_proj = nn.Linear(dim, inner_dim, bias=True)
            self.add_q_proj = nn.Linear(dim, inner_dim, bias=True)
            self.norm_added_q = RMSNorm(head_dim, eps=1e-6)
            self.norm_added_k = RMSNorm(head_dim, eps=1e-6)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=True), nn.Dropout(0.0)])
        if context_pre_only is not None and not context_pre_only:
            self.to_add_out = nn.Linear(inner_dim, dim, bias=True)

    def forward(
        self, hidden_states: Tensor, encoder_hidden_states: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        query = self.norm_q(query)
        key = self.norm_k(key)

        if encoder_hidden_states is not None:
            context_query = self.add_q_proj(encoder_hidden_states)
            context_key = self.add_k_proj(encoder_hidden_states)
            context_value = self.add_v_proj(encoder_hidden_states)
            context_query = context_query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            context_key = context_key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            context_value = context_value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            context_query = self.norm_added_q(context_query)
            context_key = self.norm_added_k(context_key)
            query = torch.cat([query, context_query], dim=2)
            key = torch.cat([key, context_key], dim=2)
            value = torch.cat([value, context_value], dim=2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not self.context_pre_only:
                encoder_hidden_states = self.to_add_out(encoder_hidden_states)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList(
            [GELU(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim, bias=True)]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class JointTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool,
        use_dual_attention: bool,
    ) -> None:
        super().__init__()
        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only
        self.norm1 = SD35AdaLayerNormZeroX(dim) if use_dual_attention else AdaLayerNormZero(dim)
        self.norm1_context = (
            AdaLayerNormContinuous(dim, dim) if context_pre_only else AdaLayerNormZero(dim)
        )
        self.attn = Attention(
            dim,
            num_attention_heads,
            attention_head_dim,
            context_pre_only=context_pre_only,
            added_projection=True,
        )
        self.attn2: Attention | None = (
            Attention(dim, num_attention_heads, attention_head_dim) if use_dual_attention else None
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim)
        self.norm2_context: nn.LayerNorm | None
        self.ff_context: FeedForward | None
        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim)
        else:
            self.norm2_context = None
            self.ff_context = None

    def forward(
        self, hidden_states: Tensor, encoder_hidden_states: Tensor, temb: Tensor
    ) -> tuple[Tensor | None, Tensor]:
        norm_outputs = self.norm1(hidden_states, temb)
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_outputs[:5]
        if self.use_dual_attention:
            norm_hidden_states2, gate_msa2 = norm_outputs[5:]
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            context_outputs = self.norm1_context(encoder_hidden_states, temb)
            (
                norm_encoder_hidden_states,
                c_gate_msa,
                c_shift_mlp,
                c_scale_mlp,
                c_gate_mlp,
            ) = context_outputs

        attn_result = self.attn(norm_hidden_states, norm_encoder_hidden_states)
        attn_output, context_attn_output = attn_result
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        if self.use_dual_attention:
            assert self.attn2 is not None
            attn_output2 = self.attn2(norm_hidden_states2)
            hidden_states = hidden_states + gate_msa2.unsqueeze(1) * attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        if self.context_pre_only:
            return None, hidden_states
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
        )
        assert self.norm2_context is not None and self.ff_context is not None
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        return encoder_hidden_states, hidden_states


class SD3Transformer2DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = SD3_TRANSFORMER_DEFAULTS | dict(config)
        if cfg["qk_norm"] != "rms_norm":
            raise ValueError("Unsupported SD3 transformer option")
        super().__init__(cfg)
        self.out_channels = (
            cfg["out_channels"] if cfg["out_channels"] is not None else cfg["in_channels"]
        )
        self.inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        self.patch_size = cfg["patch_size"]
        self.pos_embed = PatchEmbed(
            cfg["sample_size"],
            cfg["sample_size"],
            cfg["patch_size"],
            cfg["in_channels"],
            self.inner_dim,
            cfg["pos_embed_max_size"],
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            self.inner_dim, cfg["pooled_projection_dim"]
        )
        self.context_embedder = nn.Linear(cfg["joint_attention_dim"], cfg["caption_projection_dim"])
        dual_attention_layers = set(cfg["dual_attention_layers"])
        self.transformer_blocks = nn.ModuleList(
            JointTransformerBlock(
                self.inner_dim,
                cfg["num_attention_heads"],
                cfg["attention_head_dim"],
                context_pre_only=index == cfg["num_layers"] - 1,
                use_dual_attention=index in dual_attention_layers,
            )
            for index in range(cfg["num_layers"])
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(
            self.inner_dim,
            cfg["patch_size"] * cfg["patch_size"] * self.out_channels,
            bias=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        pooled_projections: Tensor,
        timestep: Tensor,
        *,
        block_controlnet_hidden_states: list[Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        skip_layers: list[int] | None = None,
    ) -> Tensor:
        if (
            block_controlnet_hidden_states is not None
            or joint_attention_kwargs
            or skip_layers is not None
        ):
            raise ValueError("Unsupported SD3 transformer option")
        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)
        hidden_states = hidden_states.contiguous()
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states, temb)
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        height = height // self.patch_size
        width = width // self.patch_size
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0],
            height,
            width,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(
            hidden_states.shape[0],
            self.out_channels,
            height * self.patch_size,
            width * self.patch_size,
        )
