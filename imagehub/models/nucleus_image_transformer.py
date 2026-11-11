"""Native Nucleus-Image mixture-of-experts diffusion transformer."""

# mypy: ignore-errors

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel


def _rms(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(
        x.dtype
    ) * weight


def _rope(x: Tensor, freqs: Tensor) -> Tensor:
    return (
        torch.view_as_real(
            torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) * freqs.unsqueeze(1)
        )
        .flatten(3)
        .to(x.dtype)
    )


class NucleusRMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return _rms(x, self.weight)


class NucleusRoPE(nn.Module):
    def __init__(self, axes: list[int]) -> None:
        super().__init__()
        self.axes = axes
        index = torch.arange(4096)
        pieces = [
            torch.polar(
                torch.ones(4096, dim // 2),
                torch.outer(index, 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))),
            )
            for dim in axes
        ]
        self.register_buffer("freqs", torch.cat(pieces, dim=1), persistent=False)
        neg = torch.arange(4096).flip(0) * -1 - 1
        self.register_buffer(
            "neg_freqs",
            torch.cat(
                [
                    torch.polar(
                        torch.ones(4096, dim // 2),
                        torch.outer(neg, 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))),
                    )
                    for dim in axes
                ],
                dim=1,
            ),
            persistent=False,
        )

    def forward(
        self, shape: tuple[int, int, int], text_len: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        frame, height, width = shape
        pos, neg = self.freqs.to(device), self.neg_freqs.to(device)
        parts = pos.split([a // 2 for a in self.axes], -1)
        n_parts = neg.split([a // 2 for a in self.axes], -1)
        f = parts[0][:frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        h = (
            torch.cat([n_parts[1][-(height - height // 2) :], parts[1][: height // 2]])
            .view(1, height, 1, -1)
            .expand(frame, height, width, -1)
        )
        w = (
            torch.cat([n_parts[2][-(width - width // 2) :], parts[2][: width // 2]])
            .view(1, 1, width, -1)
            .expand(frame, height, width, -1)
        )
        return torch.cat([f, h, w], -1).reshape(-1, sum(a // 2 for a in self.axes)), pos[
            max(height // 2, width // 2) + torch.arange(text_len, device=device)
        ]


class NucleusAttention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int) -> None:
        super().__init__()
        self.heads, self.kv_heads = heads, kv_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, kv_heads * (dim // heads), bias=False)
        self.to_v = nn.Linear(dim, kv_heads * (dim // heads), bias=False)
        self.add_k_proj = nn.Linear(dim, kv_heads * (dim // heads), bias=False)
        self.add_v_proj = nn.Linear(dim, kv_heads * (dim // heads), bias=False)
        self.norm_q = NucleusRMSNorm(dim // heads)
        self.norm_k = NucleusRMSNorm(dim // heads)
        self.norm_added_k = NucleusRMSNorm(dim // heads)
        self.norm_added_q = NucleusRMSNorm(dim // heads)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Dropout(0.0)])

    def forward(
        self, x: Tensor, context: Tensor, freqs: tuple[Tensor, Tensor], mask: Tensor | None
    ) -> Tensor:
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, -1)))
        k = self.norm_k(self.to_k(x).unflatten(-1, (self.kv_heads, -1)))
        v = self.to_v(x).unflatten(-1, (self.kv_heads, -1))
        q, k = _rope(q, freqs[0]), _rope(k, freqs[0])
        ck = _rope(
            self.norm_added_k(self.add_k_proj(context).unflatten(-1, (self.kv_heads, -1))), freqs[1]
        )
        cv = self.add_v_proj(context).unflatten(-1, (self.kv_heads, -1))
        k, v = torch.cat([k, ck], 1), torch.cat([v, cv], 1)
        groups = self.heads // self.kv_heads
        k, v = k.repeat_interleave(groups, 2), v.repeat_interleave(groups, 2)
        out = (
            F.scaled_dot_product_attention(
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
                attn_mask=None if mask is None else mask[:, None, None],
                dropout_p=0.0,
            )
            .permute(0, 2, 1, 3)
            .flatten(2)
        )
        return self.to_out[1](self.to_out[0](out))


class NucleusFeedForward(nn.Module):
    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.net = nn.ModuleList([nn.Module(), nn.Module(), nn.Linear(inner, dim, bias=False)])
        self.net[0].proj = nn.Linear(dim, inner * 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        up, gate = self.net[0].proj(x).chunk(2, -1)
        return self.net[2](up * F.silu(gate))


class NucleusMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        inner: int,
        experts: int,
        capacity: float,
        scale: float,
        grouped_mm: bool = False,
    ) -> None:
        super().__init__()
        self.experts_count, self.capacity, self.scale = experts, capacity, scale
        self.grouped_mm = grouped_mm
        self.gate = nn.Linear(dim * 2, experts, bias=False)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.empty(experts, dim, inner * 2))
        self.experts.down_proj = nn.Parameter(torch.empty(experts, inner, dim))
        self.shared_expert = NucleusFeedForward(dim, inner)

    def forward(self, x: Tensor, unmodulated: Tensor, temb: Tensor) -> Tensor:
        batch, length, dim = x.shape
        scores = (
            F.softmax(
                self.gate(
                    torch.cat([temb[:, None].expand(-1, length, -1), unmodulated], -1)
                ).float(),
                -1,
            )
            .to(x.dtype)
            .transpose(1, 2)
        )
        capacity = max(1, math.ceil(self.capacity * length / self.experts_count))
        choices = scores.topk(capacity, -1).indices
        gates = scores.gather(-1, choices)
        offsets = (torch.arange(batch, device=x.device) * length).view(batch, 1, 1)
        indices = (offsets + choices).transpose(0, 1).reshape(-1)
        gates = gates.transpose(0, 1).reshape(-1)
        sums = torch.zeros(batch * length, device=x.device, dtype=x.dtype).scatter_add_(
            0, indices, gates
        )
        gates = gates / (sums[indices] + 1e-12) * self.scale
        routed = x.reshape(-1, dim)[indices]
        if self.grouped_mm and x.device.type in ("cpu", "cuda"):
            # The published config asks for grouped matrix products. On CPU with several threads
            # they differ in the last bit from one product per expert at some token counts.
            offsets = torch.cumsum(
                torch.full((self.experts_count,), batch * capacity, device=x.device),
                0,
                dtype=torch.int32,
            )
            a, b = F.grouped_mm(routed, self.experts.gate_up_proj, offs=offsets).chunk(2, -1)
            expert_out = F.grouped_mm(F.silu(a) * b, self.experts.down_proj, offs=offsets)
            expert_out = expert_out.type_as(routed)
        else:
            outputs = []
            for expert in range(self.experts_count):
                a, b = (
                    routed[expert * batch * capacity : (expert + 1) * batch * capacity]
                    @ self.experts.gate_up_proj[expert]
                ).chunk(2, -1)
                outputs.append((F.silu(a) * b) @ self.experts.down_proj[expert])
            expert_out = torch.cat(outputs)
        out = (
            self.shared_expert(x)
            .reshape(-1, dim)
            .scatter_add(
                0,
                indices[:, None].expand(-1, dim),
                expert_out.float().to(x.dtype) * gates[:, None],
            )
        )
        return out.reshape(batch, length, dim)


class NucleusBlock(nn.Module):
    def __init__(self, config: Mapping[str, Any], moe: bool, capacity: float) -> None:
        super().__init__()
        dim = config["attention_head_dim"] * config["num_attention_heads"]
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 4))
        self.encoder_proj = nn.Linear(config["joint_attention_dim"], dim)
        self.pre_attn_norm = nn.LayerNorm(dim, 1e-6, elementwise_affine=False, bias=False)
        self.attn = NucleusAttention(
            dim, config["num_attention_heads"], config["num_key_value_heads"]
        )
        self.pre_mlp_norm = nn.LayerNorm(dim, 1e-6, elementwise_affine=False, bias=False)
        inner = (
            config["moe_intermediate_dim"]
            if moe
            else int(dim * config["mlp_ratio"] * 2 / 3) // 128 * 128
        )
        self.img_mlp = (
            NucleusMoE(
                dim,
                inner,
                config["num_experts"],
                capacity,
                config["route_scale"],
                bool(config.get("use_grouped_mm", False)),
            )
            if moe
            else NucleusFeedForward(dim, inner)
        )
        self.moe = moe

    def forward(
        self,
        x: Tensor,
        text: Tensor,
        temb: Tensor,
        freqs: tuple[Tensor, Tensor],
        mask: Tensor | None,
    ) -> Tensor:
        s1, g1, s2, g2 = self.img_mod(temb)[:, None].chunk(4, -1)
        attn = self.attn(self.pre_attn_norm(x) * (1 + s1), self.encoder_proj(text), freqs, mask)
        x = x + g1.clamp(-2, 2).tanh() * attn
        norm = self.pre_mlp_norm(x) * (1 + s2)
        out = self.img_mlp(norm, self.pre_mlp_norm(x), temb) if self.moe else self.img_mlp(norm)
        return x + g2.clamp(-2, 2).tanh() * out


class NucleusImageTransformer2DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(dict(config))
        c = self.config
        dim = c["attention_head_dim"] * c["num_attention_heads"]
        self.pos_embed = NucleusRoPE(list(c["axes_dims_rope"]))
        self.time_text_embed = nn.Module()
        self.time_text_embed.timestep_embedder = nn.Module()
        self.time_text_embed.timestep_embedder.linear_1 = nn.Linear(dim, dim * 4)
        self.time_text_embed.timestep_embedder.linear_2 = nn.Linear(dim * 4, dim)
        self.time_text_embed.norm = NucleusRMSNorm(dim)
        self.txt_norm = NucleusRMSNorm(c["joint_attention_dim"])
        self.img_in = nn.Linear(c["in_channels"], dim)
        factors = c["capacity_factors"]
        self.transformer_blocks = nn.ModuleList(
            NucleusBlock(c, index >= 3, factors[index]) for index in range(c["num_layers"])
        )
        self.norm_out = nn.Module()
        self.norm_out.linear = nn.Linear(dim, dim * 2)
        self.norm_out.norm = nn.LayerNorm(dim, 1e-6, elementwise_affine=False, bias=False)
        self.proj_out = nn.Linear(dim, c["patch_size"] ** 2 * c["out_channels"], bias=False)

    def init_non_persistent_buffers(self) -> None:
        # ``from_pretrained`` can itself be called under a meta-device context by header tests.
        # These tables are computed constants rather than published checkpoint tensors, so force
        # their reconstruction onto CPU before the base loader performs its meta-tensor check.
        with torch.device("cpu"):
            self.pos_embed = NucleusRoPE(list(self.config["axes_dims_rope"]))

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> NucleusImageTransformer2DModel:
        model = super().from_pretrained(*args, **kwargs)
        # Rebuild after ``NativeModel.from_pretrained`` applies the requested floating dtype;
        # converting the module wholesale would otherwise cast the complex RoPE tables to real.
        model.init_non_persistent_buffers()
        model.pos_embed.to(device=model.img_in.weight.device)
        return model

    def _time(self, timestep: Tensor, dtype: torch.dtype) -> Tensor:
        half = self.img_in.out_features // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timestep.device) / half)
        value = timestep.float()[:, None] * freqs * 1000
        emb = torch.cat([torch.cos(value), torch.sin(value)], -1).to(dtype)
        return self.time_text_embed.norm(
            self.time_text_embed.timestep_embedder.linear_2(
                F.silu(self.time_text_embed.timestep_embedder.linear_1(emb))
            )
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_shapes: tuple[int, int, int] | list[tuple[int, int, int]],
        encoder_hidden_states_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.img_in(hidden_states)
        text = self.txt_norm(encoder_hidden_states)
        shape = img_shapes[0] if isinstance(img_shapes, list) else img_shapes
        freqs = self.pos_embed(shape, text.shape[1], x.device)
        temb = self._time(timestep, x.dtype)
        mask = (
            None
            if encoder_hidden_states_mask is None
            else torch.cat(
                [
                    torch.ones(x.shape[:2], dtype=torch.bool, device=x.device),
                    encoder_hidden_states_mask.bool(),
                ],
                1,
            )
        )
        for block in self.transformer_blocks:
            x = block(x, text, temb, freqs, mask)
        scale, shift = self.norm_out.linear(F.silu(temb)).chunk(2, -1)
        x = self.norm_out.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out(x)
