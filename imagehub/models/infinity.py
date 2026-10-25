"""Infinity bitwise next-scale autoregressive transformer."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

ScaleSchedule = Sequence[tuple[int, int, int]]
TextCondition = tuple[Tensor, list[int], Tensor, int]

INFINITY_2B_CONFIG: dict[str, Any] = {
    "depth": 32,
    "embed_dim": 2048,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "text_channels": 2048,
    "text_max_length": 512,
    "bit_channels": 32,
    "block_chunks": 8,
    "norm_eps": 1e-6,
    "level_count": 15,
    "rope_size": 64,
}


def _rms_norm(value: Tensor, weight: Tensor, eps: float) -> Tensor:
    source = value.float()
    normalized = source * torch.rsqrt(source.pow(2).mean(-1, keepdim=True).add_(eps))
    return (normalized * weight).to(value.dtype)


class InfinityRMSNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        return _rms_norm(value, self.weight, self.eps)


class InfinityFFN(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_channels, channels)
        self.drop = nn.Identity()
        self.heuristic = -1

    def forward(self, value: Tensor) -> Tensor:
        return self.drop(self.fc2(self.act(self.fc1(value))))


def _rope_cache(
    schedule: ScaleSchedule,
    head_dim: int,
    scale_index: int,
    device: torch.device,
    rope_size: int,
) -> tuple[Tensor, Tensor]:
    _, height, width = schedule[scale_index]
    half = head_dim // 2
    inverse = 1.0 / (
        10000.0 ** (torch.arange(0, half, 2, device=device, dtype=torch.float32) / half)
    )
    y = (torch.arange(height, device=device) * (rope_size / height)).round()
    x = (torch.arange(width, device=device) * (rope_size / width)).round()
    positions = torch.cat(
        [
            y[:, None, None].expand(height, width, inverse.numel()),
            x[None, :, None].expand(height, width, inverse.numel()),
        ],
        dim=-1,
    ).reshape(height * width, half)
    angles = positions * torch.cat([inverse, inverse])
    return angles.cos(), angles.sin()


def _apply_rope(
    query: Tensor, key: Tensor, schedule: ScaleSchedule, scale_index: int, rope_size: int
) -> tuple[Tensor, Tensor]:
    cosine, sine = _rope_cache(schedule, query.shape[-1], scale_index, query.device, rope_size)
    pairs = torch.stack((query, key)).float().reshape(2, *query.shape[:-1], -1, 2)
    cosine = cosine[None, None, None]
    sine = sine[None, None, None]
    rotated = torch.stack(
        [
            cosine * pairs[..., 0] - sine * pairs[..., 1],
            sine * pairs[..., 0] + cosine * pairs[..., 1],
        ],
        dim=-1,
    ).reshape(2, *query.shape)
    rotated_query, rotated_key = rotated.to(query.dtype).unbind(0)
    return rotated_query, rotated_key


class InfinitySelfAttention(nn.Module):
    zero_k_bias: Tensor

    def __init__(self, channels: int, heads: int, rope_size: int = 64) -> None:
        super().__init__()
        self.num_heads = heads
        self.head_dim = channels // heads
        self.rope_size = rope_size
        self.mat_qkv = nn.Linear(channels, channels * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(channels))
        self.v_bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("zero_k_bias", torch.zeros(channels))
        self.scale_mul_1H11 = nn.Parameter(torch.full((1, heads, 1, 1), 4.0).log())
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Identity()
        self.caching = False
        self.cached_k: Tensor | None = None
        self.cached_v: Tensor | None = None

    def kv_caching(self, enabled: bool) -> None:
        self.caching = enabled
        self.cached_k = None
        self.cached_v = None

    def forward(
        self,
        value: Tensor,
        attention_mask: Tensor | None = None,
        schedule: ScaleSchedule | None = None,
        scale_index: int = 0,
    ) -> Tensor:
        batch, length, channels = value.shape
        bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
        qkv = F.linear(value, self.mat_qkv.weight, bias).view(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, val = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scale = self.scale_mul_1H11.clamp_max(math.log(100.0)).exp()
        query = (F.normalize(query, dim=-1, eps=1e-12) * scale).contiguous()
        key = F.normalize(key, dim=-1, eps=1e-12).contiguous()
        val = val.contiguous()
        if schedule is not None:
            query, key = _apply_rope(query, key, schedule, scale_index, self.rope_size)
        if self.caching:
            if self.cached_k is not None:
                key = torch.cat((self.cached_k, key), dim=2)
                val = torch.cat((self.cached_v, val), dim=2)  # type: ignore[arg-type]
            self.cached_k, self.cached_v = key, val
        hidden = F.scaled_dot_product_attention(
            query, key, val, attn_mask=attention_mask, dropout_p=0.0, scale=1.0
        )
        hidden = hidden.transpose(1, 2).reshape(batch, length, channels)
        return self.proj_drop(self.proj(hidden))


class InfinityCrossAttention(nn.Module):
    zero_k_bias: Tensor

    def __init__(
        self, embed_dim: int, kv_dim: int, heads: int, *, for_attn_pool: bool = False
    ) -> None:
        super().__init__()
        self.for_attn_pool = for_attn_pool
        self.num_heads = heads
        self.head_dim = embed_dim // heads
        self.mat_q: nn.Parameter | nn.Linear
        if for_attn_pool:
            query = torch.empty(1, heads, self.head_dim)
            nn.init.trunc_normal_(query, std=math.sqrt(1.0 / embed_dim / 3.0))
            self.mat_q = nn.Parameter(query)
        else:
            self.mat_q = nn.Linear(embed_dim, embed_dim)
        self.mat_kv = nn.Linear(kv_dim, embed_dim * 2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer("zero_k_bias", torch.zeros(embed_dim))
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Identity()

    def forward(self, query: Tensor | None, condition: TextCondition) -> Tensor:
        compact, _lengths, cumulative, _max_length = condition
        bias = torch.cat((self.zero_k_bias, self.v_bias))
        kv = F.linear(compact, self.mat_kv.weight, bias).view(
            compact.shape[0], 2, self.num_heads, self.head_dim
        )
        batch = cumulative.numel() - 1
        if self.for_attn_pool:
            queries = cast(Tensor, self.mat_q).expand(batch, -1, -1).unsqueeze(2)
            query_length = 1
        else:
            assert query is not None and isinstance(self.mat_q, nn.Linear)
            query_length = query.shape[1]
            queries = (
                self.mat_q(query)
                .view(batch, query_length, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
        outputs = []
        for index in range(batch):
            start, end = int(cumulative[index]), int(cumulative[index + 1])
            key, value = kv[start:end].unbind(1)
            query_row = queries[index : index + 1]
            output_dtype = query_row.dtype
            if output_dtype == torch.float32:
                query_row = query_row.to(torch.bfloat16)
                key = key.to(torch.bfloat16)
                value = value.to(torch.bfloat16)
            output = F.scaled_dot_product_attention(
                query_row,
                key.transpose(0, 1).unsqueeze(0),
                value.transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
                scale=1.0 / math.sqrt(self.head_dim),
            )
            outputs.append(output.transpose(1, 2).reshape(1, query_length, -1).to(output_dtype))
        return self.proj_drop(self.proj(torch.cat(outputs)))


class InfinityTextPool(nn.Module):
    def __init__(self, text_channels: int, channels: int) -> None:
        super().__init__()
        head_dim = 64 if channels > 4096 else min(128, text_channels)
        self.ca = InfinityCrossAttention(
            channels, text_channels, text_channels // head_dim, for_attn_pool=True
        )

    def forward(self, condition: TextCondition) -> Tensor:
        return self.ca(None, condition).squeeze(1)


class InfinitySharedAdaLin(nn.Linear):
    def forward(self, value: Tensor) -> Tensor:
        channels = self.weight.shape[0] // 6
        return super().forward(value).reshape(-1, 1, 6, channels)


class InfinityBlock(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        channels = config["embed_dim"]
        self.sa = InfinitySelfAttention(channels, config["num_heads"], config["rope_size"])
        self.ca = InfinityCrossAttention(channels, channels, config["num_heads"])
        hidden = round(channels * config["mlp_ratio"] / 256) * 256
        self.ffn = InfinityFFN(channels, hidden)
        self.ln_wo_grad = nn.LayerNorm(channels, eps=config["norm_eps"], elementwise_affine=False)
        self.ca_norm = nn.LayerNorm(channels, eps=config["norm_eps"])
        self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, channels) / math.sqrt(channels))
        self.ca_gamma = 1

    def forward(
        self,
        value: Tensor,
        adaptive: Tensor,
        condition: TextCondition,
        schedule: ScaleSchedule,
        scale_index: int,
    ) -> Tensor:
        gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + adaptive).unbind(2)
        normalized = self.ln_wo_grad(value.float()) * (scale1 + 1) + shift1
        value = value + self.sa(normalized, schedule=schedule, scale_index=scale_index) * gamma1
        value = value + self.ca(self.ca_norm(value), condition).float() * self.ca_gamma
        normalized = self.ln_wo_grad(value.float()) * (scale2 + 1) + shift2
        return value + self.ffn(normalized) * gamma2


class InfinityBlockChunk(nn.Module):
    def __init__(self, blocks: list[InfinityBlock]) -> None:
        super().__init__()
        self.module = nn.ModuleList(blocks)


class InfinityHeadNorm(nn.Module):
    def __init__(self, channels: int, eps: float) -> None:
        super().__init__()
        self.ln_wo_grad = nn.LayerNorm(channels, eps=eps, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(), nn.Linear(channels, channels * 2))

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.ada_lin(condition).view(-1, 1, 2, value.shape[-1]).unbind(2)
        return self.ln_wo_grad(value) * (scale + 1) + shift


class InfinityTransformer(NativeModel):
    """Published Infinity-2B tensor layout and one-scale cached inference."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(INFINITY_2B_CONFIG | dict(config or {}))
        cfg = self.config
        channels = cfg["embed_dim"]
        text_channels = cfg["text_channels"]
        self.cfg_uncond = nn.Parameter(torch.empty(cfg["text_max_length"], text_channels))
        self.text_norm = InfinityRMSNorm(text_channels, cfg["norm_eps"])
        self.text_proj_for_sos = InfinityTextPool(text_channels, channels)
        self.text_proj_for_ca = nn.Sequential(
            nn.Linear(text_channels, channels),
            nn.GELU(approximate="tanh"),
            nn.Linear(channels, channels),
        )
        self.pos_start = nn.Parameter(torch.empty(1, 1, channels))
        self.lvl_embed = nn.Embedding(cfg["level_count"], channels)
        self.norm0_ve = nn.Identity()
        self.word_embed = nn.Linear(cfg["bit_channels"], channels)
        self.shared_ada_lin = nn.Sequential(nn.SiLU(), InfinitySharedAdaLin(channels, channels * 6))
        blocks = [InfinityBlock(cfg) for _ in range(cfg["depth"])]
        chunks = cfg["block_chunks"]
        if chunks == 1:
            self.blocks = nn.ModuleList(blocks)
        else:
            per_chunk = cfg["depth"] // chunks
            self.block_chunks = nn.ModuleList(
                InfinityBlockChunk(blocks[index : index + per_chunk])
                for index in range(0, cfg["depth"], per_chunk)
            )
        self.head_nm = InfinityHeadNorm(channels, cfg["norm_eps"])
        self.head = nn.Linear(channels, cfg["bit_channels"] * 2)

    def blocks_in_order(self) -> list[InfinityBlock]:
        if hasattr(self, "blocks"):
            return [cast(InfinityBlock, block) for block in self.blocks]
        return [
            cast(InfinityBlock, block)
            for chunk in self.block_chunks
            for block in cast(InfinityBlockChunk, chunk).module
        ]

    def prepare_condition(
        self, condition: TextCondition, guidance: bool
    ) -> tuple[TextCondition, Tensor]:
        compact, lengths, cumulative, max_length = condition
        if guidance:
            unconditional = torch.cat([self.cfg_uncond[:length] for length in lengths], dim=0).to(
                compact
            )
            compact = torch.cat((compact, unconditional))
            offset = cumulative[-1]
            cumulative = torch.cat((cumulative, cumulative[1:] + offset))
            lengths = lengths + lengths
        compact = self.text_norm(compact)
        normalized: TextCondition = (compact, lengths, cumulative, max_length)
        pooled = self.text_proj_for_sos(normalized)
        projected = self.text_proj_for_ca(compact)
        return (projected, lengths, cumulative, max_length), pooled

    def begin_generation(
        self, condition: TextCondition, schedule: ScaleSchedule, guidance: bool
    ) -> tuple[Tensor, Tensor, Tensor, TextCondition]:
        projected, pooled = self.prepare_condition(condition, guidance)
        adaptive = self.shared_ada_lin(pooled.float()).float()
        batch = pooled.shape[0]
        stage = pooled[:, None] + self.pos_start.expand(batch, 1, -1)
        for block in self.blocks_in_order():
            block.sa.kv_caching(True)
        return stage, adaptive, pooled, projected

    def stage_logits(
        self,
        stage: Tensor,
        adaptive: Tensor,
        pooled: Tensor,
        condition: TextCondition,
        schedule: ScaleSchedule,
        scale_index: int,
    ) -> tuple[Tensor, Tensor]:
        length = math.prod(schedule[scale_index])
        level = torch.full((stage.shape[0], length), scale_index, device=stage.device)
        stage = stage + self.lvl_embed(level)
        for block in self.blocks_in_order():
            stage = block(stage, adaptive, condition, schedule, scale_index)
        logits = self.head(self.head_nm(stage.float(), pooled.float()))
        return stage, logits

    def next_stage(self, latent: Tensor, guidance_batch: int) -> Tensor:
        stage = self.word_embed(latent)
        return stage.repeat(guidance_batch, 1, 1)

    def end_generation(self) -> None:
        for block in self.blocks_in_order():
            block.sa.kv_caching(False)
