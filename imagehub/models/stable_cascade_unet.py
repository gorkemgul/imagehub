"""Stable Cascade U-Net (prior and decoder stages) with published diffusers tensor names."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

UNET_DEFAULTS: dict[str, Any] = {
    "in_channels": 16,
    "out_channels": 16,
    "timestep_ratio_embedding_dim": 64,
    "patch_size": 1,
    "conditioning_dim": 2048,
    "block_out_channels": [2048, 2048],
    "num_attention_heads": [32, 32],
    "down_num_layers_per_block": [8, 24],
    "up_num_layers_per_block": [24, 8],
    "down_blocks_repeat_mappers": [1, 1],
    "up_blocks_repeat_mappers": [1, 1],
    "block_types_per_layer": [
        ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
        ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
    ],
    "clip_text_in_channels": None,
    "clip_text_pooled_in_channels": 1280,
    "clip_image_in_channels": None,
    "clip_seq": 4,
    "effnet_in_channels": None,
    "pixel_mapper_in_channels": None,
    "kernel_size": 3,
    "dropout": [0.1, 0.1],
    "self_attn": True,
    "timestep_conditioning_type": ["sca", "crp"],
    "switch_level": None,
}
_BLOCK_TYPES = ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock")
_CONDITIONS = ("sca", "crp")


class LayerNorm2d(nn.LayerNorm):
    """Layer norm over the channel dimension of an NCHW tensor."""

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


def _norm2d(channels: int) -> LayerNorm2d:
    return LayerNorm2d(channels, elementwise_affine=False, eps=1e-6)


class TimestepBlock(nn.Module):
    def __init__(self, channels: int, timestep_dim: int, conds: Sequence[str]) -> None:
        super().__init__()
        self.mapper = nn.Linear(timestep_dim, channels * 2)
        self.conds = list(conds)
        for name in self.conds:
            setattr(self, f"mapper_{name}", nn.Linear(timestep_dim, channels * 2))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        chunks = t.chunk(len(self.conds) + 1, dim=1)
        a, b = self.mapper(chunks[0])[:, :, None, None].chunk(2, dim=1)
        for index, name in enumerate(self.conds):
            mapper = getattr(self, f"mapper_{name}")
            a_cond, b_cond = mapper(chunks[index + 1])[:, :, None, None].chunk(2, dim=1)
            a, b = a + a_cond, b + b_cond
        return x * (1 + a) + b


class GlobalResponseNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        agg_norm = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        stand_div_norm = agg_norm / (agg_norm.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * stand_div_norm) + self.beta + x


class ResBlock(nn.Module):
    def __init__(self, channels: int, skip_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels
        )
        self.norm = _norm2d(channels)
        self.channelwise = nn.Sequential(
            nn.Linear(channels + skip_channels, channels * 4),
            nn.GELU(),
            GlobalResponseNorm(channels * 4),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: Tensor, x_skip: Tensor | None = None) -> Tensor:
        x_res = x
        x = self.norm(self.depthwise(x))
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + x_res


class CascadeAttention(nn.Module):
    """Multi-head attention over an NCHW query and a token sequence as key and value."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        inner_dim = (dim // heads) * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=True), nn.Dropout(dropout)])

    def forward(self, hidden_states: Tensor, context: Tensor) -> Tensor:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        query = self.to_q(hidden_states)
        key = self.to_k(context)
        value = self.to_v(context)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)


class AttnBlock(nn.Module):
    def __init__(
        self, channels: int, cond_channels: int, heads: int, self_attn: bool, dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.norm = _norm2d(channels)
        self.attention = CascadeAttention(channels, heads, dropout)
        self.kv_mapper = nn.Sequential(nn.SiLU(), nn.Linear(cond_channels, channels))

    def forward(self, x: Tensor, kv: Tensor) -> Tensor:
        kv = self.kv_mapper(kv)
        norm_x = self.norm(x)
        if self.self_attn:
            batch_size, channel, _, _ = x.shape
            kv = torch.cat([norm_x.view(batch_size, channel, -1).transpose(1, 2), kv], dim=1)
        return x + self.attention(norm_x, kv)


class UpDownBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str, enabled: bool) -> None:
        super().__init__()
        interpolation: nn.Module
        if enabled:
            interpolation = nn.Upsample(
                scale_factor=2 if mode == "up" else 0.5, mode="bilinear", align_corners=True
            )
        else:
            interpolation = nn.Identity()
        mapping = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [interpolation, mapping] if mode == "up" else [mapping, interpolation]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _per_level(value: Any, levels: int, name: str, kinds: tuple[type, ...]) -> list[Any]:
    if isinstance(value, kinds):
        return [value] * levels
    if isinstance(value, (list, tuple)) and len(value) == levels:
        return list(value)
    raise ValueError(f"{name} must be a single value or have one entry per level ({levels})")


class StableCascadeUNet(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        values = {name: value for name, value in config.items() if not name.startswith("_")}
        unknown = sorted(set(values) - set(UNET_DEFAULTS))
        if unknown:
            raise ValueError(f"unsupported options: {', '.join(unknown)}")
        cfg = UNET_DEFAULTS | values
        super().__init__(cfg)
        channels: list[int] = list(cfg["block_out_channels"])
        levels = len(channels)
        for name in (
            "num_attention_heads",
            "down_num_layers_per_block",
            "up_num_layers_per_block",
            "down_blocks_repeat_mappers",
            "up_blocks_repeat_mappers",
            "block_types_per_layer",
        ):
            if cfg[name] is None or len(cfg[name]) != levels:
                raise ValueError(f"{name} must have one entry per level ({levels})")
        heads: list[int] = list(cfg["num_attention_heads"])
        block_types: list[list[str]] = [list(types) for types in cfg["block_types_per_layer"]]
        for level, types in enumerate(block_types):
            unsupported = [kind for kind in types if kind not in _BLOCK_TYPES]
            if unsupported:
                raise ValueError(f"Block type {unsupported[0]} not supported")
            if not types or types[0] != "SDCascadeResBlock":
                raise ValueError("the first block of every layer must be SDCascadeResBlock")
            if "SDCascadeAttnBlock" in types and heads[level] <= 0:
                raise ValueError(f"level {level} has attention blocks but no attention heads")
        conds: list[str] = list(cfg["timestep_conditioning_type"])
        if any(name not in _CONDITIONS for name in conds):
            raise ValueError(f"unsupported timestep_conditioning_type: {conds}")
        switch_level = cfg["switch_level"]
        if switch_level is not None and len(switch_level) != levels - 1:
            raise ValueError(f"switch_level must have {levels - 1} entries")
        dropout = _per_level(cfg["dropout"], levels, "dropout", (int, float))
        self_attn = _per_level(cfg["self_attn"], levels, "self_attn", (bool,))
        kernel_size: int = cfg["kernel_size"]
        patch_size: int = cfg["patch_size"]
        conditioning_dim: int = cfg["conditioning_dim"]
        clip_seq: int = cfg["clip_seq"]
        timestep_dim: int = cfg["timestep_ratio_embedding_dim"]
        first = channels[0]

        self.effnet_mapper: nn.Sequential | None = None
        self.pixels_mapper: nn.Sequential | None = None
        if cfg["effnet_in_channels"] is not None:
            self.effnet_mapper = self._conditioning_mapper(cfg["effnet_in_channels"], first)
        if cfg["pixel_mapper_in_channels"] is not None:
            self.pixels_mapper = self._conditioning_mapper(cfg["pixel_mapper_in_channels"], first)

        self.clip_txt_pooled_mapper = nn.Linear(
            cfg["clip_text_pooled_in_channels"], conditioning_dim * clip_seq
        )
        self.clip_txt_mapper: nn.Linear | None = None
        self.clip_img_mapper: nn.Linear | None = None
        if cfg["clip_text_in_channels"] is not None:
            self.clip_txt_mapper = nn.Linear(cfg["clip_text_in_channels"], conditioning_dim)
        if cfg["clip_image_in_channels"] is not None:
            self.clip_img_mapper = nn.Linear(
                cfg["clip_image_in_channels"], conditioning_dim * clip_seq
            )
        self.clip_norm = nn.LayerNorm(conditioning_dim, elementwise_affine=False, eps=1e-6)

        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(cfg["in_channels"] * patch_size**2, first, kernel_size=1),
            _norm2d(first),
        )

        def get_block(kind: str, level: int, skip_channels: int = 0) -> nn.Module:
            if kind == "SDCascadeResBlock":
                return ResBlock(channels[level], skip_channels, kernel_size, dropout[level])
            if kind == "SDCascadeAttnBlock":
                return AttnBlock(
                    channels[level],
                    conditioning_dim,
                    heads[level],
                    self_attn[level],
                    dropout[level],
                )
            return TimestepBlock(channels[level], timestep_dim, conds)

        self.down_blocks = nn.ModuleList()
        self.down_downscalers = nn.ModuleList()
        self.down_repeat_mappers = nn.ModuleList()
        for i in range(levels):
            if i > 0:
                scaler: nn.Module
                if switch_level is not None:
                    scaler = UpDownBlock2d(
                        channels[i - 1], channels[i], "down", switch_level[i - 1]
                    )
                else:
                    scaler = nn.Conv2d(channels[i - 1], channels[i], kernel_size=2, stride=2)
                self.down_downscalers.append(nn.Sequential(_norm2d(channels[i - 1]), scaler))
            else:
                self.down_downscalers.append(nn.Identity())
            down_block = nn.ModuleList()
            for _ in range(cfg["down_num_layers_per_block"][i]):
                for kind in block_types[i]:
                    down_block.append(get_block(kind, i))
            self.down_blocks.append(down_block)
            self.down_repeat_mappers.append(
                nn.ModuleList(
                    nn.Conv2d(channels[i], channels[i], kernel_size=1)
                    for _ in range(cfg["down_blocks_repeat_mappers"][i] - 1)
                )
            )

        self.up_blocks = nn.ModuleList()
        self.up_upscalers = nn.ModuleList()
        self.up_repeat_mappers = nn.ModuleList()
        for i in reversed(range(levels)):
            if i > 0:
                if switch_level is not None:
                    scaler = UpDownBlock2d(channels[i], channels[i - 1], "up", switch_level[i - 1])
                else:
                    scaler = nn.ConvTranspose2d(
                        channels[i], channels[i - 1], kernel_size=2, stride=2
                    )
                self.up_upscalers.append(nn.Sequential(_norm2d(channels[i]), scaler))
            else:
                self.up_upscalers.append(nn.Identity())
            up_block = nn.ModuleList()
            for j in range(cfg["up_num_layers_per_block"][::-1][i]):
                for k, kind in enumerate(block_types[i]):
                    has_skip = i < levels - 1 and j == k == 0
                    up_block.append(get_block(kind, i, channels[i] if has_skip else 0))
            self.up_blocks.append(up_block)
            self.up_repeat_mappers.append(
                nn.ModuleList(
                    nn.Conv2d(channels[i], channels[i], kernel_size=1)
                    for _ in range(cfg["up_blocks_repeat_mappers"][::-1][i] - 1)
                )
            )

        self.clf = nn.Sequential(
            _norm2d(first),
            nn.Conv2d(first, cfg["out_channels"] * patch_size**2, kernel_size=1),
            nn.PixelShuffle(patch_size),
        )

    @staticmethod
    def _conditioning_mapper(in_channels: int, channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            _norm2d(channels),
        )

    def get_timestep_ratio_embedding(
        self, timestep_ratio: Tensor, max_positions: int = 10000
    ) -> Tensor:
        r = timestep_ratio * max_positions
        dim = self.config["timestep_ratio_embedding_dim"]
        half_dim = dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant")
        return emb.to(dtype=r.dtype)

    def get_clip_embeddings(
        self, clip_txt_pooled: Tensor, clip_txt: Tensor | None, clip_img: Tensor | None
    ) -> Tensor:
        clip_seq = self.config["clip_seq"]
        clip_txt_pool = self.clip_txt_pooled_mapper(clip_txt_pooled).view(
            clip_txt_pooled.size(0), clip_txt_pooled.size(1) * clip_seq, -1
        )
        if clip_txt is None or clip_img is None:
            return self.clip_norm(clip_txt_pool)
        if self.clip_txt_mapper is None or self.clip_img_mapper is None:
            raise ValueError("this configuration has no clip text or clip image mapper")
        clip_txt = self.clip_txt_mapper(clip_txt)
        if clip_img.ndim == 2:
            clip_img = clip_img.unsqueeze(1)
        clip_img = self.clip_img_mapper(clip_img).view(
            clip_img.size(0), clip_img.size(1) * clip_seq, -1
        )
        return self.clip_norm(torch.cat([clip_txt, clip_txt_pool, clip_img], dim=1))

    def _down_encode(self, x: Tensor, r_embed: Tensor, clip: Tensor) -> list[Tensor]:
        level_outputs: list[Tensor] = []
        for down_block, downscaler, repmap in zip(
            self.down_blocks, self.down_downscalers, self.down_repeat_mappers, strict=True
        ):
            blocks, mappers = cast(nn.ModuleList, down_block), cast(nn.ModuleList, repmap)
            x = downscaler(x)
            for i in range(len(mappers) + 1):
                for block in blocks:
                    if isinstance(block, AttnBlock):
                        x = block(x, clip)
                    elif isinstance(block, TimestepBlock):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if i < len(mappers):
                    x = mappers[i](x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs: list[Tensor], r_embed: Tensor, clip: Tensor) -> Tensor:
        x = level_outputs[0]
        for i, (up_block, upscaler, repmap) in enumerate(
            zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers, strict=True)
        ):
            blocks, mappers = cast(nn.ModuleList, up_block), cast(nn.ModuleList, repmap)
            for j in range(len(mappers) + 1):
                for k, block in enumerate(blocks):
                    if isinstance(block, ResBlock):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (
                            x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)
                        ):
                            orig_type = x.dtype
                            x = F.interpolate(
                                x.float(), skip.shape[-2:], mode="bilinear", align_corners=True
                            )
                            x = x.to(orig_type)
                        x = block(x, skip)
                    elif isinstance(block, AttnBlock):
                        x = block(x, clip)
                    else:
                        x = block(x, r_embed)
                if j < len(mappers):
                    x = mappers[j](x)
            x = upscaler(x)
        return x

    def forward(
        self,
        sample: Tensor,
        timestep_ratio: Tensor,
        clip_text_pooled: Tensor,
        clip_text: Tensor | None = None,
        clip_img: Tensor | None = None,
        effnet: Tensor | None = None,
        pixels: Tensor | None = None,
        sca: Tensor | None = None,
        crp: Tensor | None = None,
    ) -> Tensor:
        if pixels is None:
            pixels = sample.new_zeros(sample.size(0), 3, 8, 8)
        timestep_ratio_embed = self.get_timestep_ratio_embedding(timestep_ratio)
        for name in self.config["timestep_conditioning_type"]:
            cond = sca if name == "sca" else crp
            t_cond = cond if cond is not None else torch.zeros_like(timestep_ratio)
            timestep_ratio_embed = torch.cat(
                [timestep_ratio_embed, self.get_timestep_ratio_embedding(t_cond)], dim=1
            )
        clip = self.get_clip_embeddings(clip_text_pooled, clip_text, clip_img)

        x = self.embedding(sample)
        if self.effnet_mapper is not None and effnet is not None:
            x = x + self.effnet_mapper(
                F.interpolate(effnet, size=x.shape[-2:], mode="bilinear", align_corners=True)
            )
        if self.pixels_mapper is not None:
            x = x + F.interpolate(
                self.pixels_mapper(pixels), size=x.shape[-2:], mode="bilinear", align_corners=True
            )
        level_outputs = self._down_encode(x, timestep_ratio_embed, clip)
        x = self._up_decode(level_outputs, timestep_ratio_embed, clip)
        return self.clf(x)
