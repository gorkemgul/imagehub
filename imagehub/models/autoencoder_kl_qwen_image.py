"""Single-image causal VAE used by Krea 2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

CACHE_T = 2

VAE_DEFAULTS: dict[str, Any] = {
    "base_dim": 96,
    "z_dim": 16,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "input_channels": 3,
    "latents_mean": [
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    ],
    "latents_std": [
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.916,
    ],
}


class QwenImageCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        pad = cast(tuple[int, int, int], self.padding)
        self._padding: tuple[int, ...] = (pad[2], pad[2], pad[1], pad[1], 2 * pad[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x: Tensor, cache_x: Tensor | None = None) -> Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class QwenImageRMSNorm(nn.Module):
    def __init__(
        self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: Tensor) -> Tensor:
        needs_fp32_normalize = x.dtype in (torch.float16, torch.bfloat16) or any(
            value in str(x.dtype) for value in ("float4_", "float8_")
        )
        normalized = F.normalize(
            x.float() if needs_fp32_normalize else x,
            dim=(1 if self.channel_first else -1),
        ).to(x.dtype)
        return normalized * self.scale * self.gamma + self.bias


class QwenImageUpsample(nn.Upsample):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type_as(x)


FeatureCache = list[Tensor | str | None]


class QwenImageResample(nn.Module):
    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.resample: nn.Module
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = QwenImageCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = QwenImageCausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        else:
            self.resample = nn.Identity()

    def forward(
        self,
        x: Tensor,
        feat_cache: FeatureCache | None = None,
        feat_idx: list[int] | None = None,
    ) -> Tensor:
        if feat_idx is None:
            feat_idx = [0]
        batch, channels, frames, height, width = x.size()
        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -min(CACHE_T, x.shape[2]) :, :, :].clone()
                if (
                    cache_x.shape[2] < 2
                    and feat_cache[idx] is not None
                    and feat_cache[idx] != "Rep"
                ):
                    previous = feat_cache[idx]
                    if not isinstance(previous, Tensor):
                        raise TypeError("Invalid feature cache")
                    cache_x = torch.cat(
                        [previous[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                if (
                    cache_x.shape[2] < 2
                    and feat_cache[idx] is not None
                    and feat_cache[idx] == "Rep"
                ):
                    cache_x = torch.cat(
                        [torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2
                    )
                previous = feat_cache[idx]
                x = self.time_conv(x, None if previous == "Rep" else previous)
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                x = x.reshape(batch, 2, channels, frames, height, width)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(batch, channels, frames * 2, height, width)
        frames = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        x = self.resample(x)
        x = x.view(batch, frames, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        if self.mode == "downsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                previous = feat_cache[idx]
                if not isinstance(previous, Tensor):
                    raise TypeError("Invalid feature cache")
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(torch.cat([previous[:, :, -1:, :, :], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x


class QwenImageResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = QwenImageRMSNorm(in_dim, images=False)
        self.conv1 = QwenImageCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMSNorm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImageCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            QwenImageCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    @staticmethod
    def _cached_conv(
        conv: QwenImageCausalConv3d,
        x: Tensor,
        feat_cache: FeatureCache,
        feat_idx: list[int],
    ) -> Tensor:
        idx = feat_idx[0]
        cache_x = x[:, :, -min(CACHE_T, x.shape[2]) :, :, :].clone()
        previous = feat_cache[idx]
        if cache_x.shape[2] < 2 and isinstance(previous, Tensor):
            cache_x = torch.cat(
                [previous[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
            )
        x = conv(x, previous if isinstance(previous, Tensor) else None)
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
        return x

    def forward(
        self,
        x: Tensor,
        feat_cache: FeatureCache | None = None,
        feat_idx: list[int] | None = None,
    ) -> Tensor:
        if feat_idx is None:
            feat_idx = [0]
        residual = self.conv_shortcut(x)
        x = F.silu(self.norm1(x))
        x = (
            self.conv1(x)
            if feat_cache is None
            else self._cached_conv(self.conv1, x, feat_cache, feat_idx)
        )
        x = self.dropout(F.silu(self.norm2(x)))
        x = (
            self.conv2(x)
            if feat_cache is None
            else self._cached_conv(self.conv2, x, feat_cache, feat_idx)
        )
        return x + residual


class QwenImageAttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.norm = QwenImageRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        batch, channels, frames, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        x = self.norm(x)
        qkv = self.to_qkv(x).reshape(batch * frames, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        query, key, value = qkv.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(query, key, value)
        x = x.squeeze(1).permute(0, 2, 1).reshape(batch * frames, channels, height, width)
        x = self.proj(x)
        x = x.view(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        return x + identity


class QwenImageMidBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, num_layers: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self.attentions = nn.ModuleList(QwenImageAttentionBlock(dim) for _ in range(num_layers))
        self.resnets = nn.ModuleList(
            [QwenImageResidualBlock(dim, dim, dropout)]
            + [QwenImageResidualBlock(dim, dim, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: Tensor,
        feat_cache: FeatureCache | None = None,
        feat_idx: list[int] | None = None,
    ) -> Tensor:
        if feat_idx is None:
            feat_idx = [0]
        x = self.resnets[0](x, feat_cache, feat_idx)
        for attention, resnet in zip(self.attentions, self.resnets[1:], strict=True):
            x = attention(x)
            x = resnet(x, feat_cache, feat_idx)
        return x


class QwenImageEncoder3d(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = config["base_dim"]
        z_dim = config["z_dim"] * 2  # mean and log-variance
        dim_mult = config["dim_mult"]
        num_res_blocks = config["num_res_blocks"]
        attn_scales = config["attn_scales"]
        temperal_downsample = config["temperal_downsample"]
        dropout = config["dropout"]
        dims = [dim * value for value in [1] + dim_mult]
        scale = 1.0
        self.conv_in = QwenImageCausalConv3d(config["input_channels"], dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            for _ in range(num_res_blocks):
                self.down_blocks.append(QwenImageResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    self.down_blocks.append(QwenImageAttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.down_blocks.append(QwenImageResample(out_dim, mode=mode))
                scale /= 2.0
        self.mid_block = QwenImageMidBlock(out_dim, dropout, num_layers=1)
        self.norm_out = QwenImageRMSNorm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, z_dim, 3, padding=1)

    def forward(
        self,
        x: Tensor,
        feat_cache: FeatureCache | None = None,
        feat_idx: list[int] | None = None,
    ) -> Tensor:
        if feat_idx is None:
            feat_idx = [0]
        x = (
            self.conv_in(x)
            if feat_cache is None
            else QwenImageResidualBlock._cached_conv(self.conv_in, x, feat_cache, feat_idx)
        )
        for layer in self.down_blocks:
            x = layer(x, feat_cache, feat_idx)
        x = self.mid_block(x, feat_cache, feat_idx)
        x = F.silu(self.norm_out(x))
        return (
            self.conv_out(x)
            if feat_cache is None
            else QwenImageResidualBlock._cached_conv(self.conv_out, x, feat_cache, feat_idx)
        )


class QwenImageUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = (
            nn.ModuleList([QwenImageResample(out_dim, mode=upsample_mode)])
            if upsample_mode is not None
            else None
        )

    def forward(
        self,
        x: Tensor,
        feat_cache: FeatureCache | None = None,
        feat_idx: list[int] | None = None,
    ) -> Tensor:
        if feat_idx is None:
            feat_idx = [0]
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, feat_cache, feat_idx)
        return x


class QwenImageDecoder3d(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = config["base_dim"]
        z_dim = config["z_dim"]
        dim_mult = config["dim_mult"]
        num_res_blocks = config["num_res_blocks"]
        temperal_upsample = list(reversed(config["temperal_downsample"]))
        dropout = config["dropout"]
        dims = [dim * value for value in [dim_mult[-1]] + list(reversed(dim_mult))]
        self.conv_in = QwenImageCausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImageMidBlock(dims[0], dropout, num_layers=1)
        self.up_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            if i > 0:
                in_dim = in_dim // 2
            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
            self.up_blocks.append(
                QwenImageUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                )
            )
        self.norm_out = QwenImageRMSNorm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, config["input_channels"], 3, padding=1)

    def forward(
        self,
        x: Tensor,
        feat_cache: FeatureCache | None = None,
        feat_idx: list[int] | None = None,
    ) -> Tensor:
        if feat_idx is None:
            feat_idx = [0]
        x = (
            self.conv_in(x)
            if feat_cache is None
            else QwenImageResidualBlock._cached_conv(self.conv_in, x, feat_cache, feat_idx)
        )
        x = self.mid_block(x, feat_cache, feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)
        x = F.silu(self.norm_out(x))
        return (
            self.conv_out(x)
            if feat_cache is None
            else QwenImageResidualBlock._cached_conv(self.conv_out, x, feat_cache, feat_idx)
        )


class AutoencoderKLQwenImage(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(VAE_DEFAULTS | dict(config))
        z_dim = self.config["z_dim"]
        self.encoder = QwenImageEncoder3d(self.config)
        self.quant_conv = QwenImageCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = QwenImageCausalConv3d(z_dim, z_dim, 1)
        self.decoder = QwenImageDecoder3d(self.config)
        self._decoder_conv_count = sum(
            isinstance(module, QwenImageCausalConv3d) for module in self.decoder.modules()
        )
        self._encoder_conv_count = sum(
            isinstance(module, QwenImageCausalConv3d) for module in self.encoder.modules()
        )

    @property
    def latents_mean(self) -> list[float]:
        return self.config["latents_mean"]

    @property
    def latents_std(self) -> list[float]:
        return self.config["latents_std"]

    def encode(self, image: Tensor) -> Tensor:
        """Return the posterior mode for a batch of single images in ``[-1, 1]``."""
        return self.encode_moments(image).chunk(2, dim=1)[0]

    def encode_moments(self, image: Tensor) -> Tensor:
        """Return posterior mean and log-variance, as diffusers' latent distribution does."""
        feat_cache: FeatureCache = [None] * self._encoder_conv_count
        return self.quant_conv(self.encoder(image, feat_cache=feat_cache, feat_idx=[0]))

    def decode(self, latents: Tensor) -> Tensor:
        """Decode single-image latents and clamp the result to ``[-1, 1]``."""
        feat_cache: FeatureCache = [None] * self._decoder_conv_count
        hidden = self.post_quant_conv(latents)
        image = self.decoder(hidden, feat_cache=feat_cache, feat_idx=[0])
        return torch.clamp(image, min=-1.0, max=1.0)
