"""Kandinsky 5 3D diffusion transformer with the published diffusers tensor names."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel


def _get_freqs(dim: int, max_period: float = 10000.0) -> Tensor:
    return torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim
    )


class _RMSNorm(nn.RMSNorm):
    def __init__(self, dim: int, eps: float | None = None) -> None:
        super().__init__(dim, eps=eps)


class _TimeEmbeddings(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        assert model_dim % 2 == 0
        self.model_dim = model_dim
        self.max_period = max_period
        self.register_buffer("freqs", _get_freqs(model_dim // 2, max_period), persistent=False)
        self.in_layer = nn.Linear(model_dim, time_dim, bias=True)
        self.out_layer = nn.Linear(time_dim, time_dim, bias=True)

    def forward(self, time: Tensor) -> Tensor:
        args = torch.outer(time.to(torch.float32), cast(Tensor, self.freqs).to(device=time.device))
        embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.out_layer(F.silu(self.in_layer(embed)))


class _TextEmbeddings(nn.Module):
    def __init__(self, text_dim: int, model_dim: int) -> None:
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, text_embed: Tensor) -> Tensor:
        return self.norm(self.in_layer(text_embed)).type_as(text_embed)


class _VisualEmbeddings(nn.Module):
    def __init__(self, visual_dim: int, model_dim: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, duration, height, width, dim = x.shape
        p0, p1, p2 = self.patch_size
        x = (
            x.view(
                batch_size,
                duration // p0,
                p0,
                height // p1,
                p1,
                width // p2,
                p2,
                dim,
            )
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .flatten(4, 7)
        )
        return self.in_layer(x)


class _RoPE1D(nn.Module):
    def __init__(self, dim: int, max_pos: int = 1024, max_period: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_pos = max_pos
        self.max_period = max_period
        self._init_buffer()

    def _init_buffer(self) -> None:
        freq = _get_freqs(self.dim // 2, self.max_period)
        pos = torch.arange(self.max_pos, dtype=freq.dtype)
        self.register_buffer("args", torch.outer(pos, freq), persistent=False)

    def forward(self, pos: Tensor) -> Tensor:
        args = cast(Tensor, self.args)[pos]
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class _RoPE3D(nn.Module):
    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_pos: tuple[int, int, int] = (128, 128, 128),
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period
        self._init_buffers()

    def _init_buffers(self) -> None:
        for index, (axes_dim, axis_max_pos) in enumerate(
            zip(self.axes_dims, self.max_pos, strict=True)
        ):
            freq = _get_freqs(axes_dim // 2, self.max_period)
            pos = torch.arange(axis_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{index}", torch.outer(pos, freq), persistent=False)

    def forward(
        self,
        shape: tuple[int, ...],
        pos: tuple[Tensor, Tensor, Tensor],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tensor:
        batch_size, duration, height, width = shape
        args_t = cast(Tensor, self.args_0)[pos[0]] / scale_factor[0]
        args_h = cast(Tensor, self.args_1)[pos[1]] / scale_factor[1]
        args_w = cast(Tensor, self.args_2)[pos[2]] / scale_factor[2]

        args = torch.cat(
            [
                args_t.view(1, duration, 1, 1, -1).expand(batch_size, duration, height, width, -1),
                args_h.view(1, 1, height, 1, -1).expand(batch_size, duration, height, width, -1),
                args_w.view(1, 1, 1, width, -1).expand(batch_size, duration, height, width, -1),
            ],
            dim=-1,
        )
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class _Modulation(nn.Module):
    def __init__(self, time_dim: int, model_dim: int, num_params: int) -> None:
        super().__init__()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(F.silu(x))


def _apply_rotary(x: Tensor, rope: Tensor) -> Tensor:
    # The reference implementation casts the rotary output to bfloat16 before attention; keep
    # that behaviour so float32 parity matches the reference numerics exactly.
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = (rope * x_).sum(dim=-1)
    return x_out.reshape(*x.shape).to(torch.bfloat16)


class _Attention(nn.Module):
    def __init__(self, num_channels: int, head_dim: int) -> None:
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim
        self.head_dim = head_dim
        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = _RMSNorm(head_dim)
        self.key_norm = _RMSNorm(head_dim)
        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        rotary_emb: Tensor | None = None,
    ) -> Tensor:
        query = self.to_query(hidden_states)

        if encoder_hidden_states is not None:
            key = self.to_key(encoder_hidden_states)
            value = self.to_value(encoder_hidden_states)
            shape, cond_shape = query.shape[:-1], key.shape[:-1]
            query = query.reshape(*shape, self.num_heads, -1)
            key = key.reshape(*cond_shape, self.num_heads, -1)
            value = value.reshape(*cond_shape, self.num_heads, -1)
        else:
            key = self.to_key(hidden_states)
            value = self.to_value(hidden_states)
            shape = query.shape[:-1]
            query = query.reshape(*shape, self.num_heads, -1)
            key = key.reshape(*shape, self.num_heads, -1)
            value = value.reshape(*shape, self.num_heads, -1)

        query = self.query_norm(query.float()).type_as(query)
        key = self.key_norm(key.float()).type_as(key)

        if rotary_emb is not None:
            query = _apply_rotary(query, rotary_emb).type_as(query)
            key = _apply_rotary(key, rotary_emb).type_as(key)

        query, key, value = (x.transpose(-3, -2) for x in (query, key, value))
        hidden_states = F.scaled_dot_product_attention(query, key, value)
        hidden_states = hidden_states.transpose(-3, -2)
        hidden_states = hidden_states.flatten(-2, -1)
        return self.out_layer(hidden_states)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int) -> None:
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(F.gelu(self.in_layer(x)))


class _OutLayer(nn.Module):
    def __init__(
        self, model_dim: int, time_dim: int, visual_dim: int, patch_size: tuple[int, int, int]
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.modulation = _Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim, bias=True)

    def forward(self, visual_embed: Tensor, text_embed: Tensor, time_embed: Tensor) -> Tensor:
        del text_embed
        shift, scale = torch.chunk(self.modulation(time_embed).unsqueeze(dim=1), 2, dim=-1)
        visual_embed = (
            self.norm(visual_embed.float()) * (scale.float()[:, None, None] + 1.0)
            + shift.float()[:, None, None]
        ).type_as(visual_embed)
        x = self.out_layer(visual_embed)
        batch_size, duration, height, width, _ = x.shape
        p0, p1, p2 = self.patch_size
        x = (
            x.view(
                batch_size,
                duration,
                height,
                width,
                -1,
                p0,
                p1,
                p2,
            )
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .flatten(1, 2)
            .flatten(2, 3)
            .flatten(3, 4)
        )
        return x


class _EncoderBlock(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int) -> None:
        super().__init__()
        self.text_modulation = _Modulation(time_dim, model_dim, 6)
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = _Attention(model_dim, head_dim)
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = _FeedForward(model_dim, ff_dim)

    def forward(self, x: Tensor, time_embed: Tensor, rope: Tensor) -> Tensor:
        self_attn_params, ff_params = torch.chunk(
            self.text_modulation(time_embed).unsqueeze(dim=1), 2, dim=-1
        )
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = (self.self_attention_norm(x.float()) * (scale.float() + 1.0) + shift.float()).type_as(
            x
        )
        out = self.self_attention(out, rotary_emb=rope)
        x = (x.float() + gate.float() * out.float()).type_as(x)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = (self.feed_forward_norm(x.float()) * (scale.float() + 1.0) + shift.float()).type_as(x)
        out = self.feed_forward(out)
        x = (x.float() + gate.float() * out.float()).type_as(x)
        return x


class _DecoderBlock(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int) -> None:
        super().__init__()
        self.visual_modulation = _Modulation(time_dim, model_dim, 9)
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = _Attention(model_dim, head_dim)
        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = _Attention(model_dim, head_dim)
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = _FeedForward(model_dim, ff_dim)

    def forward(
        self, visual_embed: Tensor, text_embed: Tensor, time_embed: Tensor, rope: Tensor
    ) -> Tensor:
        self_attn_params, cross_attn_params, ff_params = torch.chunk(
            self.visual_modulation(time_embed).unsqueeze(dim=1), 3, dim=-1
        )

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = (
            self.self_attention_norm(visual_embed.float()) * (scale.float() + 1.0) + shift.float()
        ).type_as(visual_embed)
        out = self.self_attention(out, rotary_emb=rope)
        visual_embed = (visual_embed.float() + gate.float() * out.float()).type_as(visual_embed)

        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        out = (
            self.cross_attention_norm(visual_embed.float()) * (scale.float() + 1.0) + shift.float()
        ).type_as(visual_embed)
        out = self.cross_attention(out, encoder_hidden_states=text_embed)
        visual_embed = (visual_embed.float() + gate.float() * out.float()).type_as(visual_embed)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = (
            self.feed_forward_norm(visual_embed.float()) * (scale.float() + 1.0) + shift.float()
        ).type_as(visual_embed)
        out = self.feed_forward(out)
        visual_embed = (visual_embed.float() + gate.float() * out.float()).type_as(visual_embed)
        return visual_embed


def _fractal_flatten(x: Tensor, rope: Tensor, shape: tuple[int, ...]) -> tuple[Tensor, Tensor]:
    del shape
    x = x.flatten(1, 3)
    rope = rope.flatten(1, 3)
    return x, rope


def _fractal_unflatten(x: Tensor, shape: tuple[int, ...]) -> Tensor:
    return x.reshape(*shape, *x.shape[2:])


class Kandinsky5Transformer3DModel(NativeModel):
    """Native Kandinsky 5 3D transformer matching the published diffusers tensor names."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        in_visual_dim = int(config["in_visual_dim"])
        in_text_dim = int(config["in_text_dim"])
        in_text_dim2 = int(config["in_text_dim2"])
        time_dim = int(config["time_dim"])
        out_visual_dim = int(config["out_visual_dim"])
        patch_size = cast(tuple[int, int, int], tuple(int(x) for x in config["patch_size"]))
        model_dim = int(config["model_dim"])
        ff_dim = int(config["ff_dim"])
        num_text_blocks = int(config["num_text_blocks"])
        num_visual_blocks = int(config["num_visual_blocks"])
        axes_dims = cast(tuple[int, int, int], tuple(int(x) for x in config["axes_dims"]))
        visual_cond = bool(config.get("visual_cond", False))

        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_cond = visual_cond
        visual_embed_dim = 2 * in_visual_dim + 1 if visual_cond else in_visual_dim

        self.time_embeddings = _TimeEmbeddings(model_dim, time_dim)
        self.text_embeddings = _TextEmbeddings(in_text_dim, model_dim)
        self.pooled_text_embeddings = _TextEmbeddings(in_text_dim2, time_dim)
        self.visual_embeddings = _VisualEmbeddings(visual_embed_dim, model_dim, patch_size)
        self.text_rope_embeddings = _RoPE1D(head_dim)
        self.visual_rope_embeddings = _RoPE3D(axes_dims)
        self.text_transformer_blocks = nn.ModuleList(
            _EncoderBlock(model_dim, time_dim, ff_dim, head_dim) for _ in range(num_text_blocks)
        )
        self.visual_transformer_blocks = nn.ModuleList(
            _DecoderBlock(model_dim, time_dim, ff_dim, head_dim) for _ in range(num_visual_blocks)
        )
        self.out_layer = _OutLayer(model_dim, time_dim, out_visual_dim, patch_size)

    def init_non_persistent_buffers(self) -> None:
        for module in self.modules():
            if isinstance(module, _TimeEmbeddings):
                module.freqs = _get_freqs(module.model_dim // 2, module.max_period)
            elif isinstance(module, _RoPE1D):
                module._init_buffer()
            elif isinstance(module, _RoPE3D):
                module._init_buffers()

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        pooled_projections: Tensor,
        visual_rope_pos: tuple[Tensor, Tensor, Tensor],
        text_rope_pos: Tensor,
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tensor:
        text_embed = self.text_embeddings(encoder_hidden_states)
        time_embed = self.time_embeddings(timestep)
        time_embed = time_embed + self.pooled_text_embeddings(pooled_projections)
        visual_embed = self.visual_embeddings(hidden_states)
        text_rope = self.text_rope_embeddings(text_rope_pos).unsqueeze(dim=0)

        for block in self.text_transformer_blocks:
            text_embed = block(text_embed, time_embed, text_rope)

        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        visual_embed, visual_rope = _fractal_flatten(visual_embed, visual_rope, visual_shape)

        for block in self.visual_transformer_blocks:
            visual_embed = block(visual_embed, text_embed, time_embed, visual_rope)

        visual_embed = _fractal_unflatten(visual_embed, visual_shape)
        return self.out_layer(visual_embed, text_embed, time_embed)
