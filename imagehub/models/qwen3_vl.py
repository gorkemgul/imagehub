"""Native Qwen3-VL vision and language conditioner."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.qwen3_vl_text import Qwen3VLTextModel


class _VisionMLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.linear_fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"])

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, -1)
    return torch.cat((-second, first), -1)


class _VisionAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.heads = config["num_heads"]
        dim = config["hidden_size"]
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, lengths: list[int]) -> Tensor:
        q, k, v = self.qkv(x).reshape(len(x), 3, self.heads, -1).unbind(1)
        q, k = q.float(), k.float()
        q = (q * cos[:, None] + _rotate_half(q) * sin[:, None]).type_as(v)
        k = (k * cos[:, None] + _rotate_half(k) * sin[:, None]).type_as(v)
        outputs = []
        for qs, ks, vs in zip(
            torch.split(q, lengths), torch.split(k, lengths), torch.split(v, lengths), strict=True
        ):
            out = F.scaled_dot_product_attention(
                qs.transpose(0, 1)[None], ks.transpose(0, 1)[None], vs.transpose(0, 1)[None]
            )
            outputs.append(out.squeeze(0).transpose(0, 1))
        return self.proj(torch.cat(outputs).flatten(1))


class _VisionBlock(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = config["hidden_size"]
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _VisionAttention(config)
        self.mlp = _VisionMLP(config)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, lengths: list[int]) -> Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, lengths)
        return x + self.mlp(self.norm2(x))


class _PatchMerger(nn.Module):
    def __init__(self, config: Mapping[str, Any], postshuffle: bool = False) -> None:
        super().__init__()
        dim = config["hidden_size"]
        merged = dim * config["spatial_merge_size"] ** 2
        self.norm = nn.LayerNorm(merged if postshuffle else dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(merged, merged)
        self.linear_fc2 = nn.Linear(merged, config["out_hidden_size"])
        self.merged = merged
        self.postshuffle = postshuffle

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x.reshape(-1, self.merged) if self.postshuffle else x)
        return self.linear_fc2(F.gelu(self.linear_fc1(x.reshape(-1, self.merged))))


class _PatchEmbed(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        kernel = (
            config["temporal_patch_size"],
            config["patch_size"],
            config["patch_size"],
        )
        self.proj = nn.Conv3d(config["in_channels"], config["hidden_size"], kernel, stride=kernel)


class Qwen3VLVisionModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        c = self.config
        self.patch_embed = _PatchEmbed(c)
        self.pos_embed = nn.Embedding(c["num_position_embeddings"], c["hidden_size"])
        self.blocks = nn.ModuleList(_VisionBlock(c) for _ in range(c["depth"]))
        self.merger = _PatchMerger(c)
        self.deepstack_merger_list = nn.ModuleList(
            _PatchMerger(c, True) for _ in c["deepstack_visual_indexes"]
        )

    @staticmethod
    def _positions(grid: Tensor, merge: int) -> Tensor:
        parts = []
        for t, h, w in grid.tolist():
            row = torch.arange(h).reshape(h // merge, merge)[:, None, :, None]
            row = row.expand(-1, w // merge, -1, merge)
            col = torch.arange(w).reshape(w // merge, merge)[None, :, None, :]
            col = col.expand(h // merge, -1, merge, -1)
            parts.append(torch.stack((row, col), -1).reshape(-1, 2).repeat(t, 1))
        return torch.cat(parts).to(grid.device)

    def _positional(self, grid: Tensor, positions: Tensor) -> Tensor:
        side = int(math.sqrt(self.config["num_position_embeddings"]))
        weights, indices = [], []
        offset = 0
        for t, h, w in grid.tolist():
            pos = positions[offset : offset + t * h * w]
            offset += t * h * w
            axes = []
            axis_weights = []
            for axis, size in ((pos[:, 0], h), (pos[:, 1], w)):
                source = axis.float() * (side - 1) / max(size - 1, 1)
                low = source.floor().long()
                axes.append(torch.stack((low, (low + 1).clamp(max=side - 1)), 1))
                frac = source - low
                axis_weights.append(torch.stack((1 - frac, frac), 1))
            indices.append((axes[0][:, :, None] * side + axes[1][:, None]).reshape(-1, 4))
            weights.append((axis_weights[0][:, :, None] * axis_weights[1][:, None]).reshape(-1, 4))
        index, weight = torch.cat(indices), torch.cat(weights)
        return (self.pos_embed(index) * weight[:, :, None]).sum(1)

    def forward(self, pixels: Tensor, grid: Tensor) -> tuple[Tensor, list[Tensor]]:
        c = self.config
        x = self.patch_embed.proj(
            pixels.reshape(
                -1, c["in_channels"], c["temporal_patch_size"], c["patch_size"], c["patch_size"]
            )
        ).reshape(-1, c["hidden_size"])
        positions = self._positions(grid, c["spatial_merge_size"])
        x = x + self._positional(grid, positions).to(x.dtype)
        head = c["hidden_size"] // c["num_heads"]
        inv = 1 / 10000 ** (torch.arange(0, head // 2, 2, device=x.device).float() / (head // 2))
        angles = positions[..., None].float() * inv
        cos = torch.cat((angles[:, 0], angles[:, 1]), -1).repeat(1, 2).cos()
        sin = torch.cat((angles[:, 0], angles[:, 1]), -1).repeat(1, 2).sin()
        lengths = grid.prod(1).tolist()
        deep = []
        for index, block in enumerate(self.blocks):
            x = block(x, cos, sin, lengths)
            if index in c["deepstack_visual_indexes"]:
                merger = self.deepstack_merger_list[c["deepstack_visual_indexes"].index(index)]
                deep.append(merger(x))
        return self.merger(x), deep


class _Qwen3VLCore(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.visual = Qwen3VLVisionModel(config["vision_config"])
        text = dict(config["text_config"])
        if "rope_parameters" in text:
            text["rope_scaling"] = text.pop("rope_parameters")
        if "rope_theta" in text["rope_scaling"]:
            text["rope_theta"] = text["rope_scaling"]["rope_theta"]
        self.language_model = Qwen3VLTextModel(text)


class Qwen3VLForConditionalGeneration(NativeModel):
    """Published Qwen3-VL container, including all vision, text, and LM-head tensors."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(dict(config))
        self.model = _Qwen3VLCore(self.config)
        self.lm_head = nn.Linear(
            self.config["text_config"]["hidden_size"],
            self.config["text_config"]["vocab_size"],
            bias=False,
        )

    def init_non_persistent_buffers(self) -> None:
        self.model.language_model.init_non_persistent_buffers()

    @staticmethod
    def _rope_positions(ids: Tensor, types: Tensor, grid: Tensor, merge: int) -> Tensor:
        result = torch.zeros(3, *ids.shape, dtype=torch.long, device=ids.device)
        grid_index = 0
        for batch in range(len(ids)):
            current = 0
            start = 0
            row = types[batch].tolist()
            for end in range(1, len(row) + 1):
                if end < len(row) and row[end] == row[start]:
                    continue
                if row[start] == 0:
                    result[:, batch, start:end] = (
                        torch.arange(end - start, device=ids.device) + current
                    )
                    current += end - start
                else:
                    _, h, w = grid[grid_index].tolist()
                    grid_index += 1
                    gh, gw = h // merge, w // merge
                    yy, xx = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
                    pos = torch.stack((torch.zeros_like(yy), yy, xx)).reshape(3, -1) + current
                    result[:, batch, start:end] = pos.to(ids.device)
                    current += max(gh, gw)
                start = end
        return result

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        mm_token_type_ids: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        embeds = self.model.language_model.embed_tokens(input_ids)
        deep = None
        visual_mask = None
        positions = None
        if pixel_values is not None:
            assert image_grid_thw is not None and mm_token_type_ids is not None
            visual, deep = self.model.visual(pixel_values, image_grid_thw)
            visual_mask = mm_token_type_ids.bool()
            embedding_mask = visual_mask.unsqueeze(-1).expand_as(embeds)
            embeds = embeds.masked_scatter(embedding_mask, visual.to(embeds.dtype))
            positions = self._rope_positions(
                input_ids,
                mm_token_type_ids,
                image_grid_thw,
                self.model.visual.config["spatial_merge_size"],
            )
        return self.model.language_model(
            None,
            attention_mask,
            inputs_embeds=embeds,
            position_ids=positions,
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deep,
        )
