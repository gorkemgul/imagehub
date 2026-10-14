"""Ovis-Image joint transformer with published diffusers tensor names."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.flux_transformer import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)

DEFAULTS: dict[str, Any] = {
    "patch_size": 1,
    "in_channels": 64,
    "out_channels": None,
    "num_layers": 6,
    "num_single_layers": 27,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 2048,
    "axes_dims_rope": (16, 56, 56),
}


def _rotary(hidden: Tensor, pair: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = pair
    cos, sin = cos[None, :, None].to(hidden.device), sin[None, :, None].to(hidden.device)
    real, imaginary = hidden.reshape(*hidden.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack((-imaginary, real), dim=-1).flatten(3)
    return (hidden.float() * cos + rotated.float() * sin).to(hidden.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList([SwiGLU(dim), nn.Dropout(0.0), nn.Linear(dim * 4, dim)])

    def forward(self, hidden: Tensor) -> Tensor:
        return self.net[2](self.net[0](hidden))


class SwiGLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim * 8)

    def forward(self, hidden: Tensor) -> Tensor:
        value, gate = self.proj(hidden).chunk(2, dim=-1)
        return F.silu(gate) * value


class OvisImageAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, *, joint: bool) -> None:
        super().__init__()
        inner = heads * head_dim
        self.head_dim, self.heads = head_dim, heads
        self.norm_q, self.norm_k = nn.RMSNorm(head_dim, eps=1e-6), nn.RMSNorm(head_dim, eps=1e-6)
        self.to_q, self.to_k, self.to_v = (nn.Linear(dim, inner, bias=True) for _ in range(3))
        self.joint = joint
        if joint:
            self.to_out = nn.ModuleList([nn.Linear(inner, dim), nn.Dropout(0.0)])
            self.norm_added_q, self.norm_added_k = (
                nn.RMSNorm(head_dim, eps=1e-6),
                nn.RMSNorm(head_dim, eps=1e-6),
            )
            self.add_q_proj, self.add_k_proj, self.add_v_proj = (
                nn.Linear(dim, inner, bias=True) for _ in range(3)
            )
            self.to_add_out = nn.Linear(inner, dim)

    def forward(
        self, hidden: Tensor, rotary: tuple[Tensor, Tensor], context: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        q, k, v = (
            layer(hidden).unflatten(-1, (self.heads, self.head_dim))
            for layer in (self.to_q, self.to_k, self.to_v)
        )
        q, k = self.norm_q(q), self.norm_k(k)
        if context is not None:
            cq, ck, cv = (
                layer(context).unflatten(-1, (self.heads, self.head_dim))
                for layer in (self.add_q_proj, self.add_k_proj, self.add_v_proj)
            )
            cq, ck = self.norm_added_q(cq), self.norm_added_k(ck)
            q, k, v = torch.cat((cq, q), 1), torch.cat((ck, k), 1), torch.cat((cv, v), 1)
        q, k = _rotary(q, rotary), _rotary(k, rotary)
        result = (
            F.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), dropout_p=0.0
            )
            .permute(0, 2, 1, 3)
            .flatten(2)
        )
        if context is None:
            return result
        context_result, result = result.split((context.shape[1], hidden.shape[1]), dim=1)
        return self.to_out[1](self.to_out[0](result)), self.to_add_out(context_result)


class OvisImageTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.norm1, self.norm1_context = AdaLayerNormZero(dim), AdaLayerNormZero(dim)
        self.attn = OvisImageAttention(dim, heads, head_dim, joint=True)
        self.norm2, self.norm2_context = (
            nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6),
            nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6),
        )
        self.ff, self.ff_context = FeedForward(dim), FeedForward(dim)

    def forward(
        self, hidden: Tensor, context: Tensor, temb: Tensor, rotary: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        h, gate, shift, scale, gate_mlp = self.norm1(hidden, temb)
        c, cgate, cshift, cscale, cgate_mlp = self.norm1_context(context, temb)
        attn, cattn = self.attn(h, rotary, c)
        hidden = hidden + gate[:, None] * attn
        hidden = hidden + gate_mlp[:, None] * self.ff(
            self.norm2(hidden) * (1 + scale[:, None]) + shift[:, None]
        )
        context = context + cgate[:, None] * cattn
        context = context + cgate_mlp[:, None] * self.ff_context(
            self.norm2_context(context) * (1 + cscale[:, None]) + cshift[:, None]
        )
        return context, hidden


class OvisImageSingleTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, dim * 8)
        self.act_mlp = nn.SiLU()
        self.proj_out = nn.Linear(dim * 5, dim)
        self.attn = OvisImageAttention(dim, heads, head_dim, joint=False)

    def forward(
        self, hidden: Tensor, context: Tensor, temb: Tensor, rotary: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        count, all_hidden = context.shape[1], torch.cat((context, hidden), dim=1)
        residual = all_hidden
        all_hidden, gate = self.norm(all_hidden, temb)
        value, mlp_gate = self.proj_mlp(all_hidden).chunk(2, dim=-1)
        output = torch.cat((self.attn(all_hidden, rotary), self.act_mlp(mlp_gate) * value), dim=-1)
        all_hidden = residual + gate[:, None] * self.proj_out(output)
        return all_hidden[:, :count], all_hidden[:, count:]


class OvisImagePosEmbed(nn.Module):
    def __init__(self, axes: tuple[int, ...]) -> None:
        super().__init__()
        self.axes = axes

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        cos, sin = [], []
        for index, dim in enumerate(self.axes):
            freq = 1.0 / (
                10000 ** (torch.arange(0, dim, 2, dtype=torch.float64, device=ids.device) / dim)
            )
            phase = torch.outer(ids[:, index].float(), freq)
            cos.append(phase.cos().repeat_interleave(2, 1).float())
            sin.append(phase.sin().repeat_interleave(2, 1).float())
        return torch.cat(cos, -1), torch.cat(sin, -1)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear_1, self.act, self.linear_2 = nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim)

    def forward(self, timestep: Tensor) -> Tensor:
        half = 128
        exponent = -math.log(10000) * torch.arange(half, device=timestep.device) / half
        value = timestep.float()[:, None] * exponent.exp()[None]
        value = torch.cat((value.cos(), value.sin()), -1)
        return self.linear_2(self.act(self.linear_1(value)))


class OvisImageTransformer2DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = DEFAULTS | {key: value for key, value in config.items() if not key.startswith("_")}
        super().__init__(cfg)
        if sum(cfg["axes_dims_rope"]) != cfg["attention_head_dim"]:
            raise ValueError("Unsupported Ovis-Image configuration")
        dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        self.out_channels = cfg["out_channels"] or cfg["in_channels"]
        self.pos_embed, self.timestep_embedder = (
            OvisImagePosEmbed(tuple(cfg["axes_dims_rope"])),
            TimestepEmbedding(dim),
        )
        self.context_embedder_norm, self.context_embedder, self.x_embedder = (
            nn.RMSNorm(cfg["joint_attention_dim"], eps=1e-6),
            nn.Linear(cfg["joint_attention_dim"], dim),
            nn.Linear(cfg["in_channels"], dim),
        )
        self.transformer_blocks = nn.ModuleList(
            OvisImageTransformerBlock(dim, cfg["num_attention_heads"], cfg["attention_head_dim"])
            for _ in range(cfg["num_layers"])
        )
        self.single_transformer_blocks = nn.ModuleList(
            OvisImageSingleTransformerBlock(
                dim, cfg["num_attention_heads"], cfg["attention_head_dim"]
            )
            for _ in range(cfg["num_single_layers"])
        )
        self.norm_out, self.proj_out = (
            AdaLayerNormContinuous(dim),
            nn.Linear(dim, cfg["patch_size"] ** 2 * self.out_channels),
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
    ) -> Tensor:
        hidden_states = self.x_embedder(hidden_states)
        temb = self.timestep_embedder(timestep.to(hidden_states.dtype) * 1000)
        encoder_hidden_states = self.context_embedder(
            self.context_embedder_norm(encoder_hidden_states)
        )
        rotary = self.pos_embed(
            torch.cat(
                (
                    txt_ids[0] if txt_ids.ndim == 3 else txt_ids,
                    img_ids[0] if img_ids.ndim == 3 else img_ids,
                )
            )
        )
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states, temb, rotary
            )
        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states, temb, rotary
            )
        return self.proj_out(self.norm_out(hidden_states, temb))
