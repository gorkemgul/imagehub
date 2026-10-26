"""Native JoyAI-Image-Edit dual-stream 3D transformer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.layers import timestep_embedding


class _GELU(nn.Module):
    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.proj(x), approximate="tanh")


class _FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        inner = int(dim * ratio)
        self.net = nn.ModuleList([_GELU(dim, inner), nn.Dropout(0.0), nn.Linear(inner, dim)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.net:
            x = layer(x)
        return x


class _Modulate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.modulate_table = nn.Parameter(torch.zeros(1, 6, dim))

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        if x.ndim != 3:
            x = x[:, None]
        return tuple(part.squeeze(1) for part in (self.modulate_table + x).chunk(6, 1))


def _rope(query: Tensor, key: Tensor, freqs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    cos, sin = (
        item.to(query.device).view(
            (1, -1, 1, query.shape[-1]) if item.ndim == 2 else (*item.shape[:2], 1, -1)
        )
        for item in freqs
    )

    def rotate(x: Tensor) -> Tensor:
        real, imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
        return torch.stack((-imag, real), -1).flatten(3)

    return (
        (query.float() * cos + rotate(query) * sin).type_as(query),
        (key.float() * cos + rotate(key) * sin).type_as(key),
    )


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        head = dim // heads
        self.heads = heads
        self.img_attn_qkv = nn.Linear(dim, 3 * dim)
        self.img_attn_q_norm = nn.RMSNorm(head, eps=1e-6)
        self.img_attn_k_norm = nn.RMSNorm(head, eps=1e-6)
        self.img_attn_proj = nn.Linear(dim, dim)
        self.txt_attn_qkv = nn.Linear(dim, 3 * dim)
        self.txt_attn_q_norm = nn.RMSNorm(head, eps=1e-6)
        self.txt_attn_k_norm = nn.RMSNorm(head, eps=1e-6)
        self.txt_attn_proj = nn.Linear(dim, dim)

    def forward(
        self,
        image: Tensor,
        text: Tensor,
        image_freqs: tuple[Tensor, Tensor],
        text_freqs: tuple[Tensor, Tensor] | None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        def qkv(
            x: Tensor, linear: nn.Linear, qnorm: nn.Module, knorm: nn.Module
        ) -> tuple[Tensor, Tensor, Tensor]:
            q, k, v = linear(x).chunk(3, -1)
            q, k, v = (a.unflatten(-1, (self.heads, -1)) for a in (q, k, v))
            return qnorm(q), knorm(k), v

        iq, ik, iv = qkv(image, self.img_attn_qkv, self.img_attn_q_norm, self.img_attn_k_norm)
        tq, tk, tv = qkv(text, self.txt_attn_qkv, self.txt_attn_q_norm, self.txt_attn_k_norm)
        iq, ik = _rope(iq, ik, image_freqs)
        if text_freqs is not None:
            tq, tk = _rope(tq, tk, text_freqs)
        q, k, v = (
            torch.cat(items, 1).permute(0, 2, 1, 3) for items in ((iq, tq), (ik, tk), (iv, tv))
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
        out = out.permute(0, 2, 1, 3).flatten(2)
        return self.img_attn_proj(out[:, : image.shape[1]]), self.txt_attn_proj(
            out[:, image.shape[1] :]
        )


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: float) -> None:
        super().__init__()
        self.img_mod, self.txt_mod = _Modulate(dim), _Modulate(dim)
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp, self.txt_mlp = _FeedForward(dim, ratio), _FeedForward(dim, ratio)
        self.attn = _Attention(dim, heads)

    @staticmethod
    def _modulate(x: Tensor, norm: nn.Module, shift: Tensor, scale: Tensor) -> Tensor:
        return norm(x) * (1 + scale[:, None]) + shift[:, None]

    def forward(
        self,
        image: Tensor,
        text: Tensor,
        temb: Tensor,
        image_freqs: tuple[Tensor, Tensor],
        text_freqs: tuple[Tensor, Tensor] | None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        im = self.img_mod(temb)
        tx = self.txt_mod(temb)
        ia, ta = self.attn(
            self._modulate(image, self.img_norm1, im[0], im[1]),
            self._modulate(text, self.txt_norm1, tx[0], tx[1]),
            image_freqs,
            text_freqs,
            attention_mask,
        )
        image, text = image + ia * im[2][:, None], text + ta * tx[2][:, None]
        return (
            image
            + self.img_mlp(self._modulate(image, self.img_norm2, im[3], im[4])) * im[5][:, None],
            text
            + self.txt_mlp(self._modulate(text, self.txt_norm2, tx[3], tx[4])) * tx[5][:, None],
        )


class _Condition(nn.Module):
    def __init__(self, dim: int, text_dim: int) -> None:
        super().__init__()
        self.time_embedder = _TimeEmbedder(dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, dim * 6)
        self.text_embedder = _TextEmbedder(text_dim, dim)

    def forward(self, timestep: Tensor, text: Tensor) -> tuple[Tensor, Tensor]:
        temb = self.time_embedder(
            timestep_embedding(timestep, 256, flip_sin_to_cos=True, shift=0)
        ).type_as(text)
        return self.time_proj(self.act_fn(temb)), self.text_embedder(text)


class _TimeEmbedder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(256, dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


class _TextEmbedder(nn.Module):
    def __init__(self, source: int, dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(source, dim)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(self.act_1(self.linear_1(x)))


class JoyImageEditTransformer3DModel(NativeModel):
    """JoyAI Edit transformer; input is one or more `(C, T, H, W)` latent items."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(dict(config))
        c = self.config
        dim = c["hidden_size"]
        self.patch_size = c["patch_size"]
        self.out_channels = c.get("out_channels") or c["in_channels"]
        self.img_in = nn.Conv3d(c["in_channels"], dim, c["patch_size"], stride=c["patch_size"])
        self.condition_embedder = _Condition(dim, c["text_dim"])
        self.double_blocks = nn.ModuleList(
            _Block(dim, c["num_attention_heads"], c["mlp_width_ratio"])
            for _ in range(c["num_layers"])
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(dim, self.out_channels * math.prod(c["patch_size"]))
        self.plus = c.get("_class_name") == "JoyImageEditPlusTransformer3DModel" or c.get(
            "plus", False
        )

    def _freqs(
        self, sizes: tuple[int, int, int], text: int | None
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor] | None]:
        dims = self.config["rope_dim_list"]
        theta = self.config["theta"]
        grid = torch.stack(
            torch.meshgrid(*(torch.arange(n, dtype=torch.float32) for n in sizes), indexing="ij"), 0
        )

        def make(pos: Tensor) -> tuple[Tensor, Tensor]:
            cos = []
            sin = []
            for i, dim in enumerate(dims):
                f = torch.outer(
                    pos[i].reshape(-1),
                    1 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim),
                )
                cos.append(f.cos().repeat_interleave(2, -1))
                sin.append(f.sin().repeat_interleave(2, -1))
            return torch.cat(cos, -1), torch.cat(sin, -1)

        vis = make(grid)
        if text is None:
            return vis, None
        offset = grid.max() + 1
        pos = torch.arange(text, dtype=torch.float32) + offset
        return vis, make(pos.repeat(3, 1))

    def _plus_freqs(
        self, shape_list: list[list[tuple[int, int, int]]], length: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        dims, theta = self.config["rope_dim_list"], self.config["theta"]
        all_cos, all_sin = [], []
        for shapes in shape_list:
            cos_parts, sin_parts, offset = [], [], 0
            for sizes in shapes:
                grid = torch.stack(
                    torch.meshgrid(
                        torch.arange(offset, offset + sizes[0], dtype=torch.float32),
                        torch.arange(sizes[1], dtype=torch.float32),
                        torch.arange(sizes[2], dtype=torch.float32),
                        indexing="ij",
                    ),
                    0,
                )
                axis_cos, axis_sin = [], []
                for axis, dim in enumerate(dims):
                    freq = torch.outer(
                        grid[axis].reshape(-1),
                        1 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim),
                    )
                    axis_cos.append(freq.cos().repeat_interleave(2, -1))
                    axis_sin.append(freq.sin().repeat_interleave(2, -1))
                cos_parts.append(torch.cat(axis_cos, -1))
                sin_parts.append(torch.cat(axis_sin, -1))
                offset += sizes[0]
            sample_cos, sample_sin = torch.cat(cos_parts), torch.cat(sin_parts)
            all_cos.append(F.pad(sample_cos, (0, 0, 0, length - len(sample_cos)), value=1))
            all_sin.append(F.pad(sample_sin, (0, 0, 0, length - len(sample_sin))))
        return torch.stack(all_cos).to(device), torch.stack(all_sin).to(device)

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        encoder_hidden_states_mask: Tensor | None = None,
        shape_list: list[list[tuple[int, int, int]]] | None = None,
    ) -> Tensor:
        if self.plus:
            if shape_list is None:
                raise ValueError("shape_list is required for the Plus transformer")
            b, patches, channels, pt, ph, pw = hidden_states.shape
            image = self.img_in(hidden_states.reshape(b * patches, channels, pt, ph, pw))
            image = image.reshape(b, patches, -1)
            temb, text = self.condition_embedder(timestep, encoder_hidden_states)
            temb = temb.unflatten(1, (6, -1))
            freqs = self._plus_freqs(shape_list, patches, hidden_states.device)
            mask = None
            if encoder_hidden_states_mask is not None:
                image_mask = torch.zeros_like(hidden_states[:, :, 0, 0, 0, 0], dtype=torch.bool)
                for index, shapes in enumerate(shape_list):
                    image_mask[index, : sum(math.prod(shape) for shape in shapes)] = True
                mask = torch.cat((image_mask, encoder_hidden_states_mask.bool()), 1)[:, None, None]
            for block in self.double_blocks:
                image, text = block(image, text, temb, freqs, None, mask)
            return (
                self.proj_out(self.norm_out(image))
                .reshape(b, patches, pt, ph, pw, self.out_channels)
                .permute(0, 1, 5, 2, 3, 4)
            )
        multi = hidden_states.ndim == 6
        if multi:
            b, n, c, t, h, w = hidden_states.shape
            if n > 1:
                hidden_states = torch.cat((hidden_states[:, -1:], hidden_states[:, :-1]), 1)
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4, 5).reshape(b, c, n * t, h, w)
        b, _, t, h, w = hidden_states.shape
        pt, ph, pw = self.patch_size
        sizes = (t // pt, h // ph, w // pw)
        image = self.img_in(hidden_states).flatten(2).transpose(1, 2)
        temb, text = self.condition_embedder(timestep, encoder_hidden_states)
        temb = temb.unflatten(1, (6, -1))
        vf, tf = self._freqs(
            sizes, text.shape[1] if self.config.get("rope_type") == "mrope" else None
        )
        for block in self.double_blocks:
            image, text = block(image, text, temb, vf, tf)
        x = (
            self.proj_out(self.norm_out(image))
            .reshape(b, *sizes, pt, ph, pw, self.out_channels)
            .permute(0, 7, 1, 4, 2, 5, 3, 6)
            .reshape(b, self.out_channels, t, h, w)
        )
        if multi:
            x = x.reshape(b, self.out_channels, n, -1, h, w).permute(0, 2, 1, 3, 4, 5)
            if n > 1:
                x = torch.cat((x[:, 1:], x[:, :1]), 1)
        return x
