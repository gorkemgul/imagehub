"""Native PixelDiT text-to-image denoiser."""

# Derivative of NVlabs PixelDiT under NSCLv1; non-commercial. See licenses/pixeldit.

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

PIXELDIT_DEFAULTS: dict[str, Any] = {
    "in_channels": 3,
    "patch_size": 16,
    "num_groups": 24,
    "hidden_size": 1536,
    "pixel_hidden_size": 16,
    "pixel_attn_hidden_size": 1152,
    "pixel_num_groups": 16,
    "patch_depth": 14,
    "pixel_depth": 2,
    "num_text_blocks": 4,
    "txt_embed_dim": 2304,
    "txt_max_length": 300,
    "use_text_rope": True,
    "text_rope_theta": 10000.0,
    "repa_encoder_index": 6,
    "use_pixel_abs_pos": True,
    "pit_adaln_post_modulation": False,
}
_METADATA = frozenset(
    {
        "architectures",
        "model_type",
        "image_size",
        "num_parameters",
        "scheduler",
        "sampling",
        "text_encoder",
    }
)


def _apply_adaln(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(dtype)


def _rope(head_dim: int, height: int, width: int, device: torch.device) -> Tensor:
    x = torch.linspace(0, 16.0, width, device=device)
    y = torch.linspace(0, 16.0, height, device=device)
    y, x = torch.meshgrid(y, x, indexing="ij")
    frequencies = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 4, device=device).float() / head_dim)
    )
    x_angles = torch.outer(x.reshape(-1), frequencies)
    y_angles = torch.outer(y.reshape(-1), frequencies)
    return torch.cat(
        [
            torch.polar(torch.ones_like(x_angles), x_angles).unsqueeze(-1),
            torch.polar(torch.ones_like(y_angles), y_angles).unsqueeze(-1),
        ],
        dim=-1,
    ).reshape(height * width, -1)


def _apply_rope(query: Tensor, key: Tensor, frequencies: Tensor) -> tuple[Tensor, Tensor]:
    frequency = frequencies[None, :, None, :]

    def rotate(value: Tensor) -> Tensor:
        complex_value = torch.view_as_complex(value.float().reshape(*value.shape[:-1], -1, 2))
        return torch.view_as_real(complex_value * frequency).flatten(3).to(value.dtype)

    return rotate(query), rotate(key)


class FeedForward(nn.Module):
    def __init__(self, size: int, hidden_size: int) -> None:
        super().__init__()
        hidden_size = int(2 * hidden_size / 3)
        self.w1 = nn.Linear(size, hidden_size, bias=False)
        self.w3 = nn.Linear(size, hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, size, bias=False)
        self.ffn_dropout = nn.Dropout(0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.ffn_dropout(F.silu(self.w1(x)) * self.w3(x)))


class MLP(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(size, size * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(size * 4, size)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TimestepConditioner(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, timestep: Tensor) -> Tensor:
        half = 128
        frequencies = torch.exp(
            -math.log(10) * torch.arange(half, device=timestep.device, dtype=torch.float32) / half
        )
        arguments = timestep[..., None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
        return self.mlp(embedding.to(next(self.mlp.parameters()).dtype))


class PatchTokenEmbedder(nn.Module):
    def __init__(self, input_size: int, output_size: int, *, norm: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(input_size, output_size)
        self.norm = RMSNorm(output_size) if norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


class RotaryAttention(nn.Module):
    def __init__(self, size: int, heads: int) -> None:
        super().__init__()
        if size % heads:
            raise ValueError("Attention size must be divisible by heads")
        self.num_heads = heads
        self.head_dim = size // heads
        self.qkv = nn.Linear(size, size * 3, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(size, size)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: Tensor, position: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, length, size = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 1, 3, 4).unbind(0)
        query, key = _apply_rope(self.q_norm(query), self.k_norm(key), position)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0)
        return self.proj_drop(self.proj(output.transpose(1, 2).reshape(batch, length, size)))


class JointAttention(nn.Module):
    def __init__(self, size: int, heads: int) -> None:
        super().__init__()
        self.num_heads = heads
        self.head_dim = size // heads
        self.qkv_x = nn.Linear(size, size * 3, bias=False)
        self.qkv_y = nn.Linear(size, size * 3, bias=False)
        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)
        self.proj_x = nn.Linear(size, size)
        self.proj_y = nn.Linear(size, size)
        self.attn_drop = nn.Dropout(0.0)
        self.proj_drop_x = nn.Dropout(0.0)
        self.proj_drop_y = nn.Dropout(0.0)

    def _qkv(self, layer: nn.Linear, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, size = x.shape
        values = layer(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        return values.permute(2, 0, 1, 3, 4).unbind(0)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        image_position: Tensor,
        text_position: Tensor | None,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        query_x, key_x, value_x = self._qkv(self.qkv_x, x)
        query_y, key_y, value_y = self._qkv(self.qkv_y, y)
        query_x, key_x = _apply_rope(self.q_norm_x(query_x), self.k_norm_x(key_x), image_position)
        query_y, key_y = self.q_norm_y(query_y), self.k_norm_y(key_y)
        if text_position is not None:
            query_y, key_y = _apply_rope(query_y, key_y, text_position)
        query = torch.cat([query_y.transpose(1, 2), query_x.transpose(1, 2)], dim=2)
        key = torch.cat([key_y.transpose(1, 2), key_x.transpose(1, 2)], dim=2)
        value = torch.cat([value_y.transpose(1, 2), value_x.transpose(1, 2)], dim=2)
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0)
        text_length = y.shape[1]
        output_y = output[:, :, :text_length].transpose(1, 2).reshape_as(y)
        output_x = output[:, :, text_length:].transpose(1, 2).reshape_as(x)
        return (
            self.proj_drop_x(self.proj_x(output_x)),
            self.proj_drop_y(self.proj_y(output_y)),
        )


class MMDiTBlock(nn.Module):
    def __init__(self, size: int, heads: int) -> None:
        super().__init__()
        self.norm_x1 = RMSNorm(size)
        self.norm_y1 = RMSNorm(size)
        self.attn = JointAttention(size, heads)
        self.norm_x2 = RMSNorm(size)
        self.norm_y2 = RMSNorm(size)
        self.mlp_x = FeedForward(size, size * 4)
        self.mlp_y = FeedForward(size, size * 4)
        self.adaLN_modulation_img = nn.Sequential(nn.Linear(size, size * 6))
        self.adaLN_modulation_txt = nn.Sequential(nn.Linear(size, size * 6))

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        condition: Tensor,
        image_position: Tensor,
        text_position: Tensor | None,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        img = self.adaLN_modulation_img(condition).chunk(6, dim=-1)
        txt = self.adaLN_modulation_txt(condition).chunk(6, dim=-1)
        attention_x, attention_y = self.attn(
            _apply_adaln(self.norm_x1(x), img[0], img[1]),
            _apply_adaln(self.norm_y1(y), txt[0], txt[1]),
            image_position,
            text_position,
            mask,
        )
        x = x + img[2] * attention_x
        y = y + txt[2] * attention_y
        x = x + img[5] * self.mlp_x(_apply_adaln(self.norm_x2(x), img[3], img[4]))
        y = y + txt[5] * self.mlp_y(_apply_adaln(self.norm_y2(y), txt[3], txt[4]))
        return x, y


def _sincos_position(size: int, height: int, width: int, device: torch.device) -> Tensor:
    grid_h = torch.arange(height, device=device, dtype=torch.float32)
    grid_w = torch.arange(width, device=device, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="xy")

    def embed(dimension: int, positions: Tensor) -> Tensor:
        omega = torch.arange(dimension // 2, device=device, dtype=torch.float64)
        omega /= dimension / 2.0
        values = torch.outer(positions.reshape(-1), 1.0 / 10000**omega)
        return torch.cat([torch.sin(values), torch.cos(values)], dim=1)

    return torch.cat([embed(size // 2, grid_w), embed(size // 2, grid_h)], dim=1)


class PixelTokenEmbedder(nn.Module):
    def __init__(self, channels: int, size: int, use_position: bool) -> None:
        super().__init__()
        self.proj = nn.Linear(channels, size)
        self.size = size
        self.use_position = use_position

    def forward(self, x: Tensor, patch_size: int) -> Tensor:
        batch, _, height, width = x.shape
        pixels = self.proj(x.permute(0, 2, 3, 1).contiguous())
        if self.use_position:
            position = _sincos_position(self.size, height, width, x.device).to(x.dtype)
            pixels = pixels + position.view(1, height, width, self.size)
        patch_h, patch_w = height // patch_size, width // patch_size
        pixels = pixels.view(batch, patch_h, patch_size, patch_w, patch_size, self.size)
        return pixels.permute(0, 1, 3, 2, 4, 5).reshape(
            batch * patch_h * patch_w, patch_size * patch_size, self.size
        )


class PixelBlock(nn.Module):
    def __init__(
        self,
        pixel_size: int,
        context_size: int,
        patch_size: int,
        attention_size: int,
        attention_heads: int,
        post_modulation: bool,
    ) -> None:
        super().__init__()
        self.pixel_dim = pixel_size
        self.patch_size = patch_size
        self.attn_dim = attention_size
        self.adaln_post_modulation = post_modulation
        pixels = patch_size * patch_size
        self.compress_to_attn = nn.Linear(pixels * pixel_size, attention_size)
        self.expand_from_attn = nn.Linear(attention_size, pixels * pixel_size)
        self.norm1 = RMSNorm(pixel_size)
        self.attn = RotaryAttention(attention_size, attention_heads)
        self.norm2 = RMSNorm(pixel_size)
        self.mlp = MLP(pixel_size)
        count = 4 if post_modulation else 6
        self.adaLN_modulation = nn.Sequential(nn.Linear(context_size, count * pixel_size * pixels))

    def forward(self, x: Tensor, condition: Tensor, height: int, width: int) -> Tensor:
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        length = patch_h * patch_w
        batch_length, pixels, _ = x.shape
        batch = batch_length // length
        count = 4 if self.adaln_post_modulation else 6
        modulation = self.adaLN_modulation(condition).view(
            batch_length, pixels, count * self.pixel_dim
        )
        chunks = modulation.chunk(count, dim=-1)
        normalized = self.norm1(x)
        if not self.adaln_post_modulation:
            normalized = _apply_adaln(normalized, chunks[0], chunks[1])
        compressed = self.compress_to_attn(normalized.reshape(batch_length, -1))
        compressed = compressed.view(batch, length, self.attn_dim)
        position = _rope(self.attn.head_dim, patch_h, patch_w, x.device)
        attention = self.attn(compressed, position)
        attention = self.expand_from_attn(attention.reshape(batch_length, self.attn_dim))
        attention = attention.view(batch_length, pixels, self.pixel_dim)
        if self.adaln_post_modulation:
            x = x + attention * (1 + chunks[0]) + chunks[1]
            return x + self.mlp(self.norm2(x)) * (1 + chunks[2]) + chunks[3]
        x = x + chunks[2] * attention
        return x + chunks[5] * self.mlp(_apply_adaln(self.norm2(x), chunks[3], chunks[4]))


class FinalLayer(nn.Module):
    def __init__(self, size: int, channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(size)
        self.linear = nn.Linear(size, channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.norm(x))


class PixelDiTCore(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = config
        channels = int(config["in_channels"])
        patch = int(config["patch_size"])
        hidden = int(config["hidden_size"])
        pixel = int(config["pixel_hidden_size"])
        self.patch_size = patch
        self.hidden_size = hidden
        self.num_groups = int(config["num_groups"])
        self.txt_max_length = int(config["txt_max_length"])
        self.use_text_rope = bool(config["use_text_rope"])
        self.text_rope_theta = float(config["text_rope_theta"])
        self.pixel_embedder = PixelTokenEmbedder(channels, pixel, bool(config["use_pixel_abs_pos"]))
        self.s_embedder = PatchTokenEmbedder(channels * patch * patch, hidden)
        self.t_embedder = TimestepConditioner(hidden)
        self.y_embedder = PatchTokenEmbedder(int(config["txt_embed_dim"]), hidden, norm=True)
        self.y_pos_embedding = nn.Parameter(torch.randn(1, self.txt_max_length, hidden))
        self.patch_blocks = nn.ModuleList(
            MMDiTBlock(hidden, self.num_groups) for _ in range(int(config["patch_depth"]))
        )
        self.pixel_blocks = nn.ModuleList(
            PixelBlock(
                pixel,
                hidden,
                patch,
                int(config["pixel_attn_hidden_size"]),
                int(config["pixel_num_groups"]),
                bool(config["pit_adaln_post_modulation"]),
            )
            for _ in range(int(config["pixel_depth"]))
        )
        self.final_layer = FinalLayer(pixel, channels)

    def _text_position(self, length: int, device: torch.device) -> Tensor:
        head_dim = self.hidden_size // self.num_groups
        frequencies = 1.0 / (
            self.text_rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        angles = torch.arange(length, device=device).float().unsqueeze(1) * frequencies
        return torch.polar(torch.ones_like(angles), angles)

    def forward(
        self, x: Tensor, timestep: Tensor, text: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        batch, _, height, width = x.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("PixelDiT height and width must be divisible by patch_size")
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        length = patch_h * patch_w
        image_position = _rope(self.hidden_size // self.num_groups, patch_h, patch_w, x.device)
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        time = self.t_embedder(timestep.reshape(-1)).view(batch, -1, self.hidden_size)
        text_length = min(text.shape[1], self.txt_max_length)
        text = self.y_embedder(text[:, :text_length])
        text = text + self.y_pos_embedding[:, :text_length].to(text.dtype)
        condition = F.silu(time)
        semantic = self.s_embedder(patches)
        text_position = self._text_position(text_length, x.device) if self.use_text_rope else None
        joint_mask = None
        if mask is not None:
            padding = mask[:, :text_length] == 0
            image_padding = torch.zeros((batch, length), dtype=torch.bool, device=x.device)
            joint_mask = torch.cat([padding, image_padding], dim=1).view(
                batch, 1, 1, text_length + length
            )
        for block in self.patch_blocks:
            semantic, text = block(
                semantic, text, condition, image_position, text_position, joint_mask
            )
        semantic = F.silu(time + semantic).reshape(batch * length, self.hidden_size)
        pixels = self.pixel_embedder(x, self.patch_size)
        for block in self.pixel_blocks:
            pixels = block(pixels, semantic, height, width)
        pixels = self.final_layer(pixels)
        channels = int(self.config["in_channels"])
        pixels = pixels.view(batch, length, self.patch_size**2, channels)
        pixels = pixels.permute(0, 3, 2, 1).reshape(batch, channels * self.patch_size**2, length)
        return F.fold(
            pixels,
            (height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )


class PixelDiT(NativeModel):
    """Published PixelDiT trainer layout, including the unused REPA projector."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(set(config) - set(PIXELDIT_DEFAULTS) - _METADATA)
        if unknown:
            raise ValueError(f"Unsupported PixelDiT setting {unknown[0]}")
        cfg = PIXELDIT_DEFAULTS | {
            key: value for key, value in config.items() if key not in _METADATA
        }
        super().__init__(cfg)
        if int(cfg["pixel_depth"]) <= 0:
            raise ValueError("PixelDiT requires a pixel pathway")
        self.core = PixelDiTCore(self.config)
        hidden = int(cfg["hidden_size"])
        self._repa_projector = nn.Sequential(
            nn.Linear(hidden, 2048),
            nn.SiLU(),
            nn.Linear(2048, 2048),
            nn.SiLU(),
            nn.Linear(2048, 768),
        )

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        # The published loader copies the pickle tensors into freshly allocated
        # parameters. Preserve that layout: CPU matrix multiplication can otherwise
        # choose a different summation path for tensors assigned from the mmap.
        return {name: tensor.clone() for name, tensor in state.items()}

    def forward(
        self, x: Tensor, timestep: Tensor, text: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        return self.core(x, timestep, text, mask)
