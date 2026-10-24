"""Native single-image inference path for the published WAN causal VAE."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

WAN_VAE_DEFAULTS: dict[str, Any] = {
    "in_channels": 3,
    "out_channels": 3,
    "base_dim": 96,
    "decoder_base_dim": None,
    "z_dim": 16,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "is_residual": False,
    "latents_mean": [0.0] * 16,
    "latents_std": [1.0] * 16,
}


class _CausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self._causal_padding = (
            int(self.padding[2]),
            int(self.padding[2]),
            int(self.padding[1]),
            int(self.padding[1]),
            2 * int(self.padding[0]),
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.pad(x, self._causal_padding))


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, *, images: bool) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim, *((1, 1) if images else (1, 1, 1))))
        self.scale = dim**0.5

    def forward(self, x: Tensor) -> Tensor:
        source = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return F.normalize(source, dim=1).to(x.dtype) * self.scale * self.gamma


class _Upsample(nn.Upsample):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type_as(x)


class _Resample(nn.Module):
    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        if mode.startswith("upsample"):
            self.resample = nn.Sequential(
                _Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            if mode == "upsample3d":
                self.time_conv = _CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        else:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=2)
            )
            if mode == "downsample3d":
                self.time_conv = _CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        x = self.resample(x)
        return x.view(batch, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)


class _ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(in_dim, images=False)
        self.conv1 = _CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = _RMSNorm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = _CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            _CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv_shortcut(x)
        x = self.conv1(F.silu(self.norm1(x)))
        return self.conv2(self.dropout(F.silu(self.norm2(x)))) + residual


class _AttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = _RMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        batch, channels, frames, height, width = x.shape
        flat = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        qkv = self.to_qkv(self.norm(flat)).reshape(batch * frames, 1, channels * 3, -1)
        q, k, v = qkv.permute(0, 1, 3, 2).contiguous().chunk(3, -1)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.squeeze(1).permute(0, 2, 1).reshape(batch * frames, channels, height, width)
        out = self.proj(out).view(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        return out + residual


class _MidBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.attentions = nn.ModuleList([_AttentionBlock(dim)])
        self.resnets = nn.ModuleList([_ResidualBlock(dim, dim, dropout) for _ in range(2)])

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnets[0](x)
        return self.resnets[1](self.attentions[0](x))


class _Encoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = config["base_dim"]
        dims = [dim * value for value in [1] + config["dim_mult"]]
        self.conv_in = _CausalConv3d(config["in_channels"], dims[0], 3, padding=1)
        blocks: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            for _ in range(config["num_res_blocks"]):
                blocks.append(_ResidualBlock(in_dim, out_dim, config["dropout"]))
                in_dim = out_dim
            if index != len(config["dim_mult"]) - 1:
                mode = "downsample3d" if config["temperal_downsample"][index] else "downsample2d"
                blocks.append(_Resample(out_dim, mode))
        self.down_blocks = nn.ModuleList(blocks)
        self.mid_block = _MidBlock(dims[-1], config["dropout"])
        self.norm_out = _RMSNorm(dims[-1], images=False)
        self.conv_out = _CausalConv3d(dims[-1], config["z_dim"] * 2, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.norm_out(self.mid_block(x))))


class _UpBlock(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, count: int, dropout: float, mode: str | None
    ) -> None:
        super().__init__()
        blocks = []
        for _ in range(count + 1):
            blocks.append(_ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(blocks)
        self.upsamplers = nn.ModuleList([_Resample(out_dim, mode)]) if mode else None

    def forward(self, x: Tensor) -> Tensor:
        for block in self.resnets:
            x = block(x)
        return self.upsamplers[0](x) if self.upsamplers is not None else x


class _Decoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = config["decoder_base_dim"] or config["base_dim"]
        mult = config["dim_mult"]
        dims = [dim * value for value in [mult[-1]] + list(reversed(mult))]
        self.conv_in = _CausalConv3d(config["z_dim"], dims[0], 3, padding=1)
        self.mid_block = _MidBlock(dims[0], config["dropout"])
        upsample = list(reversed(config["temperal_downsample"]))
        blocks = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            if index > 0:
                in_dim //= 2
            mode = None
            if index != len(mult) - 1:
                mode = "upsample3d" if upsample[index] else "upsample2d"
            blocks.append(
                _UpBlock(in_dim, out_dim, config["num_res_blocks"], config["dropout"], mode)
            )
        self.up_blocks = nn.ModuleList(blocks)
        self.norm_out = _RMSNorm(dims[-1], images=False)
        self.conv_out = _CausalConv3d(dims[-1], config["out_channels"], 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class AutoencoderKLWan(NativeModel):
    """WAN causal VAE specialized for native single-frame inference."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(WAN_VAE_DEFAULTS | dict(config))
        if self.config["is_residual"]:
            raise ValueError("residual WAN VAEs are not supported")
        self.encoder = _Encoder(self.config)
        z_dim = self.config["z_dim"]
        self.quant_conv = _CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = _CausalConv3d(z_dim, z_dim, 1)
        self.decoder = _Decoder(self.config)

    @property
    def latents_mean(self) -> list[float]:
        return self.config["latents_mean"]

    @property
    def latents_std(self) -> list[float]:
        return self.config["latents_std"]

    def _moments(self, image: Tensor) -> Tensor:
        return self.quant_conv(self.encoder(image.unsqueeze(2)))

    def encode(self, image: Tensor) -> Tensor:
        return self._moments(image).chunk(2, 1)[0][:, :, 0]

    def encode_sample(self, image: Tensor) -> Tensor:
        mean, logvar = self._moments(image).chunk(2, 1)
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype)
        return (mean + torch.exp(0.5 * logvar.clamp(-30, 20)) * noise)[:, :, 0]

    def decode(self, latents: Tensor) -> Tensor:
        hidden = self.post_quant_conv(latents.unsqueeze(2))
        return self.decoder(hidden).clamp(-1, 1)[:, :, 0]
