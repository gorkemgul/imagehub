"""Native PixArt transformer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

PIXART_DEFAULTS: dict[str, Any] = {
    "num_attention_heads": 16,
    "attention_head_dim": 72,
    "in_channels": 4,
    "out_channels": 8,
    "num_layers": 28,
    "dropout": 0.0,
    "norm_num_groups": 32,
    "cross_attention_dim": 1152,
    "attention_bias": True,
    "sample_size": 128,
    "patch_size": 2,
    "activation_fn": "gelu-approximate",
    "num_embeds_ada_norm": 1000,
    "upcast_attention": False,
    "norm_type": "ada_norm_single",
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
    "interpolation_scale": None,
    "use_additional_conditions": None,
    "caption_channels": None,
    "attention_type": "default",
}
_IGNORED = frozenset(
    {
        "double_self_attention",
        "num_vector_embeds",
        "only_cross_attention",
        "use_linear_projection",
    }
)


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


def _get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: tuple[int, int],
    base_size: int,
    interpolation_scale: float,
    device: torch.device | None = None,
) -> Tensor:
    grid_h = (
        torch.arange(grid_size[0], device=device, dtype=torch.float32)
        / (grid_size[0] / base_size)
        / interpolation_scale
    )
    grid_w = (
        torch.arange(grid_size[1], device=device, dtype=torch.float32)
        / (grid_size[1] / base_size)
        / interpolation_scale
    )
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing="xy"), dim=0)
    grid = grid.reshape(2, 1, grid_size[1], grid_size[0])

    def embed_1d(dim: int, positions: Tensor) -> Tensor:
        omega = torch.arange(dim // 2, device=device, dtype=torch.float64)
        omega /= dim / 2.0
        omega = 1.0 / 10000**omega
        values = torch.outer(positions.reshape(-1), omega)
        return torch.cat([torch.sin(values), torch.cos(values)], dim=1)

    emb_h = embed_1d(embed_dim // 2, grid[0])
    emb_w = embed_1d(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: Tensor) -> Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(
        self, embedding_dim: int, size_emb_dim: int, use_additional_conditions: bool
    ) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)
        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.resolution_embedder = TimestepEmbedding(256, size_emb_dim)
            self.aspect_ratio_embedder = TimestepEmbedding(256, size_emb_dim)

    def forward(
        self,
        timestep: Tensor,
        resolution: Tensor | None,
        aspect_ratio: Tensor | None,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> Tensor:
        projected = _get_timestep_embedding(timestep, 256)
        timesteps_emb = self.timestep_embedder(projected.to(dtype=hidden_dtype))
        if not self.use_additional_conditions:
            return timesteps_emb
        if resolution is None or aspect_ratio is None:
            raise ValueError("resolution and aspect_ratio are required")
        resolution_emb = _get_timestep_embedding(resolution.flatten(), 256).to(hidden_dtype)
        resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
        aspect_ratio_emb = _get_timestep_embedding(aspect_ratio.flatten(), 256).to(hidden_dtype)
        aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio_emb).reshape(batch_size, -1)
        return timesteps_emb + torch.cat([resolution_emb, aspect_ratio_emb], dim=1)


class AdaLayerNormSingle(nn.Module):
    def __init__(self, embedding_dim: int, use_additional_conditions: bool) -> None:
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
            use_additional_conditions=use_additional_conditions,
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)

    def forward(
        self,
        timestep: Tensor,
        added_cond_kwargs: Mapping[str, Tensor] | None,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        conditions: Mapping[str, Tensor | None] = added_cond_kwargs or {
            "resolution": None,
            "aspect_ratio": None,
        }
        embedded_timestep = self.emb(
            timestep,
            conditions.get("resolution"),
            conditions.get("aspect_ratio"),
            batch_size,
            hidden_dtype,
        )
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class PixArtAlphaTextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, caption_length: int | None) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        if caption_length is not None:
            self.y_embedding = nn.Parameter(torch.randn(caption_length, in_features))

    def forward(self, caption: Tensor) -> Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        return self.linear_2(hidden_states)


class PatchEmbed(nn.Module):
    pos_embed: Tensor

    def __init__(
        self,
        height: int,
        width: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        interpolation_scale: float,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=True,
        )
        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale
        position = _get_2d_sincos_pos_embed(
            embed_dim,
            (self.height, self.width),
            self.base_size,
            interpolation_scale,
        )
        self.register_buffer("pos_embed", position.float().unsqueeze(0), persistent=False)

    def _position(self, height: int, width: int, device: torch.device) -> Tensor:
        if self.height == height and self.width == width:
            return self.pos_embed
        return (
            _get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                (height, width),
                self.base_size,
                self.interpolation_scale,
                device,
            )
            .float()
            .unsqueeze(0)
        )

    def forward(self, latent: Tensor) -> Tensor:
        height = latent.shape[-2] // self.patch_size
        width = latent.shape[-1] // self.patch_size
        latent = self.proj(latent)
        latent = latent.flatten(2).transpose(1, 2)
        return (latent + self._position(height, width, latent.device)).to(latent.dtype)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        bias: bool,
        cross_attention_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        context_dim = cross_attention_dim or query_dim
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, query_dim, bias=True), nn.Dropout(dropout)]
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch_size = context.shape[0]
        query = self.to_q(hidden_states)
        key = self.to_k(context)
        value = self.to_v(context)
        head_dim = key.shape[-1] // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(
                self.heads,
                dim=1,
                output_size=attention_mask.shape[1] * self.heads,
            )
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)


class GELU(nn.Module):
    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.gelu(self.proj(hidden_states), approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList(
            [GELU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim)]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float,
        cross_attention_dim: int,
        attention_bias: bool,
        norm_elementwise_affine: bool,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn1 = Attention(
            dim, num_attention_heads, attention_head_dim, attention_bias, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn2 = Attention(
            dim,
            num_attention_heads,
            attention_head_dim,
            attention_bias,
            cross_attention_dim,
            dropout,
        )
        self.ff = FeedForward(dim, dropout)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None,
        timestep: Tensor,
    ) -> Tensor:
        batch_size = hidden_states.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        attention_output = self.attn1(norm_hidden_states)
        hidden_states = gate_msa * attention_output + hidden_states
        attention_output = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = attention_output + hidden_states
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        return gate_mlp * self.ff(norm_hidden_states) + hidden_states


class PixArtTransformer2DModel(NativeModel):
    """PixArt Alpha and Sigma transformer with published tensor names."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        unknown = sorted(
            key
            for key in config
            if not key.startswith("_") and key not in PIXART_DEFAULTS and key not in _IGNORED
        )
        if unknown:
            raise ValueError(f"Unsupported PixArt setting {unknown[0]}")
        cfg = PIXART_DEFAULTS | {
            key: value
            for key, value in config.items()
            if not key.startswith("_") and key not in _IGNORED
        }
        required = {
            "activation_fn": "gelu-approximate",
            "attention_type": "default",
            "dropout": 0.0,
            "norm_type": "ada_norm_single",
            "upcast_attention": False,
        }
        for key, expected in required.items():
            if cfg[key] != expected:
                raise ValueError(f"Unsupported PixArt setting {key}={cfg[key]!r}")
        use_additional_conditions = cfg["use_additional_conditions"]
        if use_additional_conditions is None:
            use_additional_conditions = cfg["sample_size"] == 128
        self.use_additional_conditions = use_additional_conditions
        self.inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        self.out_channels = (
            cfg["in_channels"] if cfg["out_channels"] is None else cfg["out_channels"]
        )
        self.patch_size = cfg["patch_size"]
        interpolation_scale = cfg["interpolation_scale"]
        if interpolation_scale is None:
            interpolation_scale = max(cfg["sample_size"] // 64, 1)
        self.pos_embed = PatchEmbed(
            cfg["sample_size"],
            cfg["sample_size"],
            cfg["patch_size"],
            cfg["in_channels"],
            self.inner_dim,
            interpolation_scale,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    cfg["num_attention_heads"],
                    cfg["attention_head_dim"],
                    cfg["dropout"],
                    cfg["cross_attention_dim"],
                    cfg["attention_bias"],
                    cfg["norm_elementwise_affine"],
                    cfg["norm_eps"],
                )
                for _ in range(cfg["num_layers"])
            ]
        )
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(
            self.inner_dim, cfg["patch_size"] * cfg["patch_size"] * self.out_channels
        )
        self.adaln_single = AdaLayerNormSingle(self.inner_dim, use_additional_conditions)
        self.caption_projection = (
            PixArtAlphaTextProjection(
                cfg["caption_channels"],
                self.inner_dim,
                # Only the original Alpha checkpoints store the unused null caption embedding.
                120
                if "use_additional_conditions" not in config and "interpolation_scale" not in config
                else None,
            )
            if cfg["caption_channels"] is not None
            else None
        )
        self.model_config = cfg

    def init_non_persistent_buffers(self) -> None:
        self.pos_embed.pos_embed = (
            _get_2d_sincos_pos_embed(
                self.inner_dim,
                (self.pos_embed.height, self.pos_embed.width),
                self.pos_embed.base_size,
                self.pos_embed.interpolation_scale,
            )
            .float()
            .unsqueeze(0)
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        encoder_attention_mask: Tensor | None,
        added_cond_kwargs: Mapping[str, Tensor] | None,
    ) -> Tensor:
        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError("added_cond_kwargs cannot be None when using additional conditions")
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        batch_size = hidden_states.shape[0]
        height = hidden_states.shape[-2] // self.patch_size
        width = hidden_states.shape[-1] // self.patch_size
        hidden_states = self.pos_embed(hidden_states)
        timestep, embedded_timestep = self.adaln_single(
            timestep,
            added_cond_kwargs,
            batch_size,
            hidden_states.dtype,
        )
        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
            )
        shift, scale = (
            self.scale_shift_table[None]
            + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(
            hidden_states.device
        )
        hidden_states = self.proj_out(hidden_states).squeeze(1)
        hidden_states = hidden_states.reshape(
            -1,
            height,
            width,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(
            -1,
            self.out_channels,
            height * self.patch_size,
            width * self.patch_size,
        )
