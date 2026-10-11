"""Würstchen v2 prior and decoder networks with published diffusers tensor names."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.stable_cascade_unet import (
    CascadeAttention,
    LayerNorm2d,
    ResBlock,
)

PRIOR_DEFAULTS: dict[str, Any] = {
    "c_in": 16,
    "c": 1280,
    "c_cond": 1024,
    "c_r": 64,
    "depth": 16,
    "nhead": 16,
    "dropout": 0.1,
}
DECODER_DEFAULTS: dict[str, Any] = {
    "c_in": 4,
    "c_out": 4,
    "c_r": 64,
    "patch_size": 2,
    "c_cond": 1024,
    "c_hidden": [320, 640, 1280, 1280],
    "nhead": [-1, 10, 20, 20],
    "blocks": [4, 4, 14, 4],
    "level_config": ["CT", "CTA", "CTA", "CTA"],
    "inject_effnet": [False, True, True, True],
    "effnet_embd": 16,
    "clip_embd": 1024,
    "kernel_size": 3,
    "dropout": 0.1,
}


def _norm2d(channels: int) -> LayerNorm2d:
    return LayerNorm2d(channels, elementwise_affine=False, eps=1e-6)


def _config(defaults: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    values = {name: value for name, value in config.items() if not name.startswith("_")}
    unknown = sorted(set(values) - set(defaults))
    if unknown:
        raise ValueError(f"unsupported options: {', '.join(unknown)}")
    return {**defaults, **values}


def gen_r_embedding(r: Tensor, c_r: int, max_positions: int = 10000) -> Tensor:
    """Sinusoidal embedding of the timestep ratio."""
    r = r * max_positions
    half_dim = c_r // 2
    step = math.log(max_positions) / (half_dim - 1)
    emb = torch.arange(half_dim, device=r.device).float().mul(-step).exp()
    emb = r[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=1)
    if c_r % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    return emb.to(dtype=r.dtype)


class TimestepBlock(nn.Module):
    def __init__(self, channels: int, timestep_dim: int) -> None:
        super().__init__()
        self.mapper = nn.Linear(timestep_dim, channels * 2)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        a, b = self.mapper(t)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + a) + b


class BatchedAttention(nn.Module):
    """Attention over an NCHW query and a token sequence, computed with batched matmuls.

    This is the arithmetic of the default (non-fused) attention processor, which the prior
    uses: scaled scores through ``baddbmm``, softmax in float32, then ``bmm``.
    """

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        inner_dim = head_dim * heads
        self.scale = head_dim**-0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=True), nn.Dropout(dropout)])

    def _split(self, tensor: Tensor) -> Tensor:
        batch, length, dim = tensor.shape
        tensor = tensor.reshape(batch, length, self.heads, dim // self.heads).permute(0, 2, 1, 3)
        return tensor.reshape(batch * self.heads, length, dim // self.heads)

    def forward(self, hidden_states: Tensor, context: Tensor) -> Tensor:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        query = self._split(self.to_q(hidden_states))
        key = self._split(self.to_k(context))
        value = self._split(self.to_v(context))
        dtype = query.dtype
        scores = torch.baddbmm(
            torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=dtype, device=query.device
            ),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        probs = scores.float().softmax(dim=-1).to(dtype)
        out = torch.bmm(probs, value)
        batch, length, dim = out.shape
        out = out.reshape(batch // self.heads, self.heads, length, dim)
        out = out.permute(0, 2, 1, 3).reshape(batch // self.heads, length, dim * self.heads)
        out = self.to_out[1](self.to_out[0](out))
        return out.transpose(-1, -2).reshape(batch_size, channel, height, width)


class AttnBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_channels: int,
        heads: int,
        *,
        fused: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = _norm2d(channels)
        self.attention: nn.Module = (
            CascadeAttention(channels, heads, dropout)
            if fused
            else BatchedAttention(channels, heads, dropout)
        )
        self.kv_mapper = nn.Sequential(nn.SiLU(), nn.Linear(cond_channels, channels))

    def forward(self, x: Tensor, kv: Tensor) -> Tensor:
        kv = self.kv_mapper(kv)
        norm_x = self.norm(x)
        batch_size, channel, _, _ = x.shape
        kv = torch.cat([norm_x.view(batch_size, channel, -1).transpose(1, 2), kv], dim=1)
        return x + self.attention(norm_x, kv)


class WuerstchenPrior(NativeModel):
    """Predicts image embeddings (the semantic compressed latent) from CLIP text states."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = _config(PRIOR_DEFAULTS, config)
        super().__init__(cfg)
        c, c_cond, depth = cfg["c"], cfg["c_cond"], cfg["depth"]
        self.c_r = cfg["c_r"]
        self.projection = nn.Conv2d(cfg["c_in"], c, kernel_size=1)
        self.cond_mapper = nn.Sequential(nn.Linear(c_cond, c), nn.LeakyReLU(0.2), nn.Linear(c, c))
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(ResBlock(c, 0, 3, cfg["dropout"]))
            self.blocks.append(TimestepBlock(c, self.c_r))
            self.blocks.append(AttnBlock(c, c, cfg["nhead"], fused=False, dropout=cfg["dropout"]))
        self.out = nn.Sequential(_norm2d(c), nn.Conv2d(c, cfg["c_in"] * 2, kernel_size=1))

    def gen_r_embedding(self, r: Tensor, max_positions: int = 10000) -> Tensor:
        return gen_r_embedding(r, self.c_r, max_positions)

    def forward(self, x: Tensor, r: Tensor, c: Tensor) -> Tensor:
        x_in = x
        x = self.projection(x)
        c_embed = self.cond_mapper(c)
        r_embed = self.gen_r_embedding(r)
        for block in self.blocks:
            if isinstance(block, AttnBlock):
                x = block(x, c_embed)
            elif isinstance(block, TimestepBlock):
                x = block(x, r_embed)
            else:
                x = block(x)
        a, b = self.out(x).chunk(2, dim=1)
        return (x_in - a) / ((1 - b).abs() + 1e-5)


class WuerstchenDiffNeXt(NativeModel):
    """Decoder that maps the image embeddings and text states to the VQGAN latent."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = _config(DECODER_DEFAULTS, config)
        super().__init__(cfg)
        c_hidden: list[int] = list(cfg["c_hidden"])
        levels = len(c_hidden)
        for name in ("nhead", "blocks", "level_config", "inject_effnet"):
            if len(cfg[name]) != levels:
                raise ValueError(f"{name} must have one entry per level ({levels})")
        if any(kind not in "CTA" for level in cfg["level_config"] for kind in level):
            raise ValueError("level_config may only contain the block types C, T and A")
        for level, kinds in enumerate(cfg["level_config"]):
            if "A" in kinds and cfg["nhead"][level] <= 0:
                raise ValueError(f"level {level} has attention blocks but no attention heads")
        dropout = cfg["dropout"]
        if not isinstance(dropout, list):
            dropout = [dropout] * levels
        c_cond, c_r, kernel_size = cfg["c_cond"], cfg["c_r"], cfg["kernel_size"]
        patch_size = cfg["patch_size"]
        inject: list[bool] = list(cfg["inject_effnet"])
        self.c_r = c_r
        self.c_cond = c_cond

        self.clip_mapper = nn.Linear(cfg["clip_embd"], c_cond)
        mappers: list[nn.Module | None] = [
            nn.Conv2d(cfg["effnet_embd"], c_cond, kernel_size=1) if enabled else None
            for enabled in inject + list(reversed(inject))
        ]
        self.effnet_mappers = nn.ModuleList(cast(Any, mappers))
        self.seq_norm = nn.LayerNorm(c_cond, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(cfg["c_in"] * patch_size**2, c_hidden[0], kernel_size=1),
            _norm2d(c_hidden[0]),
        )

        def get_block(kind: str, level: int, c_skip: int) -> nn.Module:
            if kind == "C":
                return ResBlock(c_hidden[level], c_skip, kernel_size, dropout[level])
            if kind == "A":
                return AttnBlock(
                    c_hidden[level],
                    c_cond,
                    cfg["nhead"][level],
                    fused=True,
                    dropout=dropout[level],
                )
            return TimestepBlock(c_hidden[level], c_r)

        self.down_blocks = nn.ModuleList()
        for i in range(levels):
            down_block = nn.ModuleList()
            if i > 0:
                down_block.append(
                    nn.Sequential(
                        _norm2d(c_hidden[i - 1]),
                        nn.Conv2d(c_hidden[i - 1], c_hidden[i], kernel_size=2, stride=2),
                    )
                )
            for _ in range(cfg["blocks"][i]):
                for kind in cfg["level_config"][i]:
                    down_block.append(get_block(kind, i, c_cond if inject[i] else 0))
            self.down_blocks.append(down_block)

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(levels)):
            up_block = nn.ModuleList()
            for j in range(cfg["blocks"][i]):
                for k, kind in enumerate(cfg["level_config"][i]):
                    c_skip = c_hidden[i] if i < levels - 1 and j == k == 0 else 0
                    c_skip += c_cond if inject[i] else 0
                    up_block.append(get_block(kind, i, c_skip))
            if i > 0:
                up_block.append(
                    nn.Sequential(
                        _norm2d(c_hidden[i]),
                        nn.ConvTranspose2d(c_hidden[i], c_hidden[i - 1], kernel_size=2, stride=2),
                    )
                )
            self.up_blocks.append(up_block)

        self.clf = nn.Sequential(
            _norm2d(c_hidden[0]),
            nn.Conv2d(c_hidden[0], 2 * cfg["c_out"] * patch_size**2, kernel_size=1),
            nn.PixelShuffle(patch_size),
        )

    def gen_r_embedding(self, r: Tensor, max_positions: int = 10000) -> Tensor:
        return gen_r_embedding(r, self.c_r, max_positions)

    def gen_c_embeddings(self, clip: Tensor) -> Tensor:
        return self.seq_norm(self.clip_mapper(clip))

    def _effnet_features(self, mapper_index: int, effnet: Tensor, x: Tensor) -> Tensor:
        mapper = cast(nn.Conv2d, self.effnet_mappers[mapper_index])
        dtype = effnet.dtype
        resized = F.interpolate(
            effnet.float(), size=x.shape[-2:], mode="bicubic", antialias=True, align_corners=True
        )
        return mapper(resized.to(dtype))

    def _down_encode(
        self, x: Tensor, r_embed: Tensor, effnet: Tensor, clip: Tensor | None
    ) -> list[Tensor]:
        level_outputs: list[Tensor] = []
        for i, down_block in enumerate(self.down_blocks):
            effnet_c: Tensor | None = None
            has_mapper = self.effnet_mappers[i] is not None
            for block in cast(nn.ModuleList, down_block):
                if isinstance(block, ResBlock):
                    if effnet_c is None and has_mapper:
                        effnet_c = self._effnet_features(i, effnet, x)
                    x = block(x, effnet_c if has_mapper else None)
                elif isinstance(block, AttnBlock):
                    x = block(x, cast(Tensor, clip))
                elif isinstance(block, TimestepBlock):
                    x = block(x, r_embed)
                else:
                    x = block(x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(
        self, level_outputs: list[Tensor], r_embed: Tensor, effnet: Tensor, clip: Tensor | None
    ) -> Tensor:
        x = level_outputs[0]
        for i, up_block in enumerate(self.up_blocks):
            effnet_c: Tensor | None = None
            mapper_index = len(self.down_blocks) + i
            for j, block in enumerate(cast(nn.ModuleList, up_block)):
                if isinstance(block, ResBlock):
                    if effnet_c is None and self.effnet_mappers[mapper_index] is not None:
                        effnet_c = self._effnet_features(mapper_index, effnet, x)
                    skip = level_outputs[i] if j == 0 and i > 0 else None
                    if effnet_c is not None:
                        skip = effnet_c if skip is None else torch.cat([skip, effnet_c], dim=1)
                    x = block(x, skip)
                elif isinstance(block, AttnBlock):
                    x = block(x, cast(Tensor, clip))
                elif isinstance(block, TimestepBlock):
                    x = block(x, r_embed)
                else:
                    x = block(x)
        return x

    def forward(
        self,
        x: Tensor,
        r: Tensor,
        effnet: Tensor,
        clip: Tensor | None = None,
        x_cat: Tensor | None = None,
        eps: float = 1e-3,
        return_noise: bool = True,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if x_cat is not None:
            x = torch.cat([x, x_cat], dim=1)
        r_embed = self.gen_r_embedding(r)
        if clip is not None:
            clip = self.gen_c_embeddings(clip)
        x_in = x
        x = self.embedding(x)
        level_outputs = self._down_encode(x, r_embed, effnet, clip)
        x = self._up_decode(level_outputs, r_embed, effnet, clip)
        a, b = self.clf(x).chunk(2, dim=1)
        b = b.sigmoid() * (1 - eps * 2) + eps
        if return_noise:
            return (x_in - a) / b
        return a, b
