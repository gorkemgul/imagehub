"""Native HunyuanDiT transformer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

HUNYUAN_DIT_DEFAULTS: dict[str, Any] = {
    "num_attention_heads": 16,
    "attention_head_dim": 88,
    "in_channels": 4,
    "patch_size": 2,
    "activation_fn": "gelu-approximate",
    "sample_size": 32,
    "hidden_size": 1152,
    "num_layers": 28,
    "mlp_ratio": 4.0,
    "learn_sigma": True,
    "cross_attention_dim": 1024,
    "norm_type": "layer_norm",
    "cross_attention_dim_t5": 2048,
    "pooled_projection_dim": 1024,
    "text_len": 77,
    "text_len_t5": 256,
    "use_style_cond_and_image_meta_size": True,
}


def get_resize_crop_region_for_grid(
    src: tuple[int, int], tgt_size: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    th = tw = tgt_size
    h, w = src
    ratio = h / w
    if ratio > 1:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))
    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))
    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def _get_1d_rotary_pos_embed(dim: int, pos: Tensor) -> tuple[Tensor, Tensor]:
    frequencies = 1.0 / (
        10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim)
    )
    frequencies = torch.outer(pos, frequencies)
    cos = (
        frequencies.cos().repeat_interleave(2, dim=1, output_size=frequencies.shape[1] * 2).float()
    )
    sin = (
        frequencies.sin().repeat_interleave(2, dim=1, output_size=frequencies.shape[1] * 2).float()
    )
    return cos, sin


def get_2d_rotary_pos_embed(
    embed_dim: int,
    crops_coords: tuple[tuple[int, int], tuple[int, int]],
    grid_size: tuple[int, int],
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4")
    start, stop = crops_coords
    grid_h = torch.linspace(
        start[0],
        stop[0] * (grid_size[0] - 1) / grid_size[0],
        grid_size[0],
        device=device,
        dtype=torch.float32,
    )
    grid_w = torch.linspace(
        start[1],
        stop[1] * (grid_size[1] - 1) / grid_size[1],
        grid_size[1],
        device=device,
        dtype=torch.float32,
    )
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing="xy"), dim=0)
    grid = grid.reshape(2, 1, *grid.shape[1:])
    emb_h = _get_1d_rotary_pos_embed(embed_dim // 2, grid[0].reshape(-1))
    emb_w = _get_1d_rotary_pos_embed(embed_dim // 2, grid[1].reshape(-1))
    return torch.cat([emb_h[0], emb_w[0]], dim=1), torch.cat([emb_h[1], emb_w[1]], dim=1)


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


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: Tensor) -> Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


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


class PixArtAlphaTextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.act_1 = FP32SiLU()
        self.linear_2 = nn.Linear(hidden_size, out_features)

    def forward(self, caption: Tensor) -> Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        return self.linear_2(hidden_states)


class FP32SiLU(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return F.silu(inputs.float(), inplace=False).to(inputs.dtype)


class HunyuanDiTAttentionPool(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int | None = None
    ) -> None:
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(1, 0, 2)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            training=self.training,
            need_weights=False,
            use_separate_proj_weight=True,
        )
        return x.squeeze(0)


class HunyuanCombinedTimestepTextSizeStyleEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        seq_len: int,
        cross_attention_dim: int,
        use_style_cond_and_image_meta_size: bool,
    ) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)
        self.pooler = HunyuanDiTAttentionPool(
            seq_len, cross_attention_dim, num_heads=8, output_dim=pooled_projection_dim
        )
        self.use_style_cond_and_image_meta_size = use_style_cond_and_image_meta_size
        if use_style_cond_and_image_meta_size:
            self.style_embedder = nn.Embedding(1, embedding_dim)
            extra_in_dim = 256 * 6 + embedding_dim + pooled_projection_dim
        else:
            extra_in_dim = pooled_projection_dim
        self.extra_embedder = PixArtAlphaTextProjection(
            extra_in_dim, embedding_dim * 4, embedding_dim
        )

    def forward(
        self,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        image_meta_size: Tensor,
        style: Tensor,
        hidden_dtype: torch.dtype,
    ) -> Tensor:
        timesteps_proj = _get_timestep_embedding(timestep, 256)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        pooled_projections = self.pooler(encoder_hidden_states)
        if self.use_style_cond_and_image_meta_size:
            image_meta_size = _get_timestep_embedding(image_meta_size.view(-1), 256)
            image_meta_size = image_meta_size.to(dtype=hidden_dtype)
            image_meta_size = image_meta_size.view(-1, 6 * 256)
            style_embedding = self.style_embedder(style)
            extra_cond = torch.cat([pooled_projections, image_meta_size, style_embedding], dim=1)
        else:
            extra_cond = torch.cat([pooled_projections], dim=1)
        return timesteps_emb + self.extra_embedder(extra_cond)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=True,
        )
        self.patch_size = patch_size

    def forward(self, latent: Tensor) -> Tensor:
        latent = self.proj(latent)
        return latent.flatten(2).transpose(1, 2).to(latent.dtype)


def _apply_rotary_emb(x: Tensor, freqs_cis: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = freqs_cis
    cos = cos[None, None, :, :].to(x.device)
    sin = sin[None, None, :, :].to(x.device)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(
        self, query_dim: int, cross_attention_dim: int | None, dim_head: int, heads: int
    ) -> None:
        super().__init__()
        context_dim = cross_attention_dim or query_dim
        inner_dim = dim_head * heads
        self.heads = heads
        self.is_cross_attention = cross_attention_dim is not None
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim, bias=True), nn.Dropout(0.0)])
        self.norm_q = nn.LayerNorm(dim_head, eps=1e-6)
        self.norm_k = nn.LayerNorm(dim_head, eps=1e-6)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        batch_size = hidden_states.shape[0]
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = self.to_q(hidden_states)
        key = self.to_k(context)
        value = self.to_v(context)
        head_dim = key.shape[-1] // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        query = self.norm_q(query)
        key = self.norm_k(key)
        if image_rotary_emb is not None:
            query = _apply_rotary_emb(query, image_rotary_emb)
            if not self.is_cross_attention:
                key = _apply_rotary_emb(key, image_rotary_emb)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)


class AdaLayerNormShift(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=True, eps=1e-6)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        shift = self.linear(self.silu(emb.to(torch.float32)).to(emb.dtype))
        return self.norm(x) + shift.unsqueeze(dim=1)


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            [GELU(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim, bias=True)]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class HunyuanDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        cross_attention_dim: int,
        ff_inner_dim: int,
        skip: bool,
    ) -> None:
        super().__init__()
        head_dim = dim // num_attention_heads
        self.norm1 = AdaLayerNormShift(dim)
        self.attn1 = Attention(dim, None, head_dim, num_attention_heads)
        self.norm2 = FP32LayerNorm(dim, 1e-6, True)
        self.attn2 = Attention(dim, cross_attention_dim, head_dim, num_attention_heads)
        self.norm3 = FP32LayerNorm(dim, 1e-6, True)
        self.ff = FeedForward(dim, ff_inner_dim)
        if skip:
            self.skip_norm = FP32LayerNorm(2 * dim, 1e-6, True)
            self.skip_linear: nn.Linear | None = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor],
        skip: Tensor | None = None,
    ) -> Tensor:
        if self.skip_linear is not None:
            if skip is None:
                raise ValueError("A skip tensor is required for this block")
            cat = torch.cat([hidden_states, skip], dim=-1)
            cat = self.skip_norm(cat)
            hidden_states = self.skip_linear(cat)
        norm_hidden_states = self.norm1(hidden_states, temb)
        hidden_states = hidden_states + self.attn1(
            norm_hidden_states, image_rotary_emb=image_rotary_emb
        )
        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        mlp_inputs = self.norm3(hidden_states)
        return hidden_states + self.ff(mlp_inputs)


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6, elementwise_affine=False)

    def forward(self, x: Tensor, conditioning_embedding: Tensor) -> Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class HunyuanDiT2DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        values = HUNYUAN_DIT_DEFAULTS | {
            name: value for name, value in config.items() if not name.startswith("_")
        }
        super().__init__(values)
        num_attention_heads = int(values["num_attention_heads"])
        attention_head_dim = int(values["attention_head_dim"])
        in_channels = int(values["in_channels"])
        patch_size = int(values["patch_size"])
        hidden_size = int(values["hidden_size"])
        num_layers = int(values["num_layers"])
        cross_attention_dim = int(values["cross_attention_dim"])
        cross_attention_dim_t5 = int(values["cross_attention_dim_t5"])
        pooled_projection_dim = int(values["pooled_projection_dim"])
        text_len = int(values["text_len"])
        text_len_t5 = int(values["text_len_t5"])
        self.out_channels = in_channels * 2 if values["learn_sigma"] else in_channels
        self.num_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.text_embedder = PixArtAlphaTextProjection(
            cross_attention_dim_t5, cross_attention_dim_t5 * 4, cross_attention_dim
        )
        self.text_embedding_padding = nn.Parameter(
            torch.randn(text_len + text_len_t5, cross_attention_dim)
        )
        self.pos_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.time_extra_emb = HunyuanCombinedTimestepTextSizeStyleEmbedding(
            hidden_size,
            pooled_projection_dim,
            text_len_t5,
            cross_attention_dim_t5,
            bool(values["use_style_cond_and_image_meta_size"]),
        )
        ff_inner_dim = int(self.inner_dim * float(values["mlp_ratio"]))
        self.blocks = nn.ModuleList(
            [
                HunyuanDiTBlock(
                    self.inner_dim,
                    num_attention_heads,
                    cross_attention_dim,
                    ff_inner_dim,
                    layer > num_layers // 2,
                )
                for layer in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=True
        )

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        text_embedding_mask: Tensor,
        encoder_hidden_states_t5: Tensor,
        text_embedding_mask_t5: Tensor,
        image_meta_size: Tensor,
        style: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor],
    ) -> Tensor:
        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_extra_emb(
            timestep,
            encoder_hidden_states_t5,
            image_meta_size,
            style,
            hidden_dtype=timestep.dtype,
        )
        batch_size, sequence_length, _ = encoder_hidden_states_t5.shape
        encoder_hidden_states_t5 = self.text_embedder(
            encoder_hidden_states_t5.view(-1, encoder_hidden_states_t5.shape[-1])
        )
        encoder_hidden_states_t5 = encoder_hidden_states_t5.view(batch_size, sequence_length, -1)
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_t5], dim=1)
        text_embedding_mask = torch.cat([text_embedding_mask, text_embedding_mask_t5], dim=-1)
        text_embedding_mask = text_embedding_mask.unsqueeze(2).bool()
        encoder_hidden_states = torch.where(
            text_embedding_mask, encoder_hidden_states, self.text_embedding_padding
        )
        skips: list[Tensor] = []
        num_layers = int(self.config["num_layers"])
        for layer, block in enumerate(self.blocks):
            if layer > num_layers // 2:
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    skip=skips.pop(),
                )
            else:
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                )
            if layer < (num_layers // 2 - 1):
                skips.append(hidden_states)
        hidden_states = self.norm_out(hidden_states, temb.to(torch.float32))
        hidden_states = self.proj_out(hidden_states)
        patch_size = self.pos_embed.patch_size
        height = height // patch_size
        width = width // patch_size
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0],
            height,
            width,
            patch_size,
            patch_size,
            self.out_channels,
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(
            hidden_states.shape[0],
            self.out_channels,
            height * patch_size,
            width * patch_size,
        )
