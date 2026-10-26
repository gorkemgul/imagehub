"""LlamaGen VQ-16 image tokenizer: codebook lookup and decoder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

# VQ-16 as used by the text-to-image checkpoints (``vq_ds16_t2i.pt``).
LLAMAGEN_VQ16: dict[str, Any] = {
    "codebook_size": 16384,
    "codebook_embed_dim": 8,
    "codebook_l2_norm": True,
    "ch": 128,
    "ch_mult": [1, 1, 2, 2, 4],
    "num_res_blocks": 2,
    "z_channels": 256,
    "out_channels": 3,
}

# Published tensors that decoding does not use.
EXCLUDED_PREFIXES = ("encoder.", "quant_conv.", "quantize.codebook_used")


def _nonlinearity(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def _normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class LlamaGenVQResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = _nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = _nonlinearity(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class LlamaGenVQAttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = _normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


class LlamaGenVQUpsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class LlamaGenVQConvBlock(nn.Module):
    def __init__(self, res: nn.ModuleList, attn: nn.ModuleList, upsample: nn.Module | None) -> None:
        super().__init__()
        self.res = res
        self.attn = attn
        if upsample is not None:
            self.upsample = upsample


class LlamaGenVQDecoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        ch_mult = config["ch_mult"]
        ch = config["ch"]
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = config["num_res_blocks"]
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(config["z_channels"], block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.ModuleList(
            [
                LlamaGenVQResnetBlock(block_in, block_in),
                LlamaGenVQAttnBlock(block_in),
                LlamaGenVQResnetBlock(block_in, block_in),
            ]
        )
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(LlamaGenVQResnetBlock(block_in, block_out))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(LlamaGenVQAttnBlock(block_in))
            upsample = LlamaGenVQUpsample(block_in) if i_level != 0 else None
            self.conv_blocks.append(LlamaGenVQConvBlock(res_block, attn_block, upsample))
        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in, config["out_channels"], kernel_size=3, stride=1, padding=1
        )

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        for mid_block in self.mid:
            h = mid_block(h)
        for i_level, module in enumerate(self.conv_blocks):
            block = cast(LlamaGenVQConvBlock, module)
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)
        h = self.norm_out(h)
        h = _nonlinearity(h)
        return self.conv_out(h)


class LlamaGenVectorQuantizer(nn.Module):
    def __init__(self, n_e: int, e_dim: int, l2_norm: bool) -> None:
        super().__init__()
        self.l2_norm = l2_norm
        self.embedding = nn.Embedding(n_e, e_dim)

    def get_codebook_entry(
        self, indices: Tensor, shape: tuple[int, ...] | None = None, channel_first: bool = True
    ) -> Tensor:
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]
        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


class LlamaGenVQ(NativeModel):
    """Decoder half of VQ-16; tensor names are the published ones (see ``prepare_state_dict``)."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(LLAMAGEN_VQ16 | dict(config))
        cfg = self.config
        self.decoder = LlamaGenVQDecoder(cfg)
        self.quantize = LlamaGenVectorQuantizer(
            cfg["codebook_size"], cfg["codebook_embed_dim"], cfg["codebook_l2_norm"]
        )
        self.post_quant_conv = nn.Conv2d(cfg["codebook_embed_dim"], cfg["z_channels"], 1)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Drop the encoder, ``quant_conv`` and the training-only ``codebook_used`` counter."""
        return {
            name: value for name, value in state.items() if not name.startswith(EXCLUDED_PREFIXES)
        }

    def decode(self, quant: Tensor) -> Tensor:
        return self.decoder(self.post_quant_conv(quant))

    def decode_code(
        self, code_b: Tensor, shape: tuple[int, ...] | None = None, channel_first: bool = True
    ) -> Tensor:
        """Token ids to pixels in [-1, 1]; ``shape`` is ``(batch, embed_dim, height, width)``."""
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        return self.decode(quant_b)
