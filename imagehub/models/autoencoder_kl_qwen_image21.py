"""Single-image causal VAE used by Qwen-Image 2.1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

CACHE_T = 2

VAE_DEFAULTS: dict[str, Any] = {
    "base_dim": 96,
    "decoder_base_dim": 144,
    "z_dim": 64,
    "dim_mult": [1, 2, 4, 8, 8],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temperal_downsample": [False, True, True, True],
    "dropout": 0.0,
    "latents_mean": None,
    "latents_std": None,
    "is_residual": True,
    "in_channels": 4,
    "out_channels": 4,
    "patch_size": None,
    "scale_factor_temporal": 8,
    "scale_factor_spatial": 16,
}


class QwenImage21AvgDown3D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1
    ) -> None:
        super().__init__()
        factor = factor_t * factor_s * factor_s
        if in_channels * factor % out_channels != 0:
            raise ValueError(
                f"`in_channels` ({in_channels}) times the downsampling factor ({factor}) must be "
                f"divisible by `out_channels` ({out_channels})."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor
        self.group_size = in_channels * factor // out_channels

    def forward(self, x: Tensor) -> Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        batch, channels, frames, height, width = x.shape
        x = x.view(
            batch,
            channels,
            frames // self.factor_t,
            self.factor_t,
            height // self.factor_s,
            self.factor_s,
            width // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            batch,
            channels * self.factor,
            frames // self.factor_t,
            height // self.factor_s,
            width // self.factor_s,
        )
        x = x.view(
            batch,
            self.out_channels,
            self.group_size,
            frames // self.factor_t,
            height // self.factor_s,
            width // self.factor_s,
        )
        return x.mean(dim=2)


class QwenImage21DupUp3D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if out_channels * self.factor % in_channels != 0:
            raise ValueError("The upsampling channels must divide evenly")
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: Tensor, first_chunk: bool = False) -> Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class QwenImage21CausalConv3d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self._padding = (
            int(self.padding[1]),
            int(self.padding[1]),
            int(self.padding[0]),
            int(self.padding[0]),
        )
        self.padding = (0, 0)

    def forward(self, x: Tensor, cache_x: Tensor | None = None) -> Tensor:
        if cache_x is not None:
            raise ValueError(
                "This convolution is the image specialization of Wan's causal 3D one: it folds "
                "the single frame away and has no temporal context to prepend, so it cannot take "
                "a feature cache."
            )
        x = F.pad(x.squeeze(2), list(self._padding))
        return super().forward(x).unsqueeze(2)


class QwenImage21RMSNorm(nn.Module):
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


class QwenImage21Upsample(nn.Upsample):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type_as(x)


FeatureCache = list[Tensor | str | None]


class QwenImage21Resample(nn.Module):
    def __init__(self, dim: int, mode: str, upsample_out_dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2
        self.resample: nn.Module
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                QwenImage21Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                QwenImage21Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = QwenImage21CausalConv3d(dim, dim * 2, (1, 1), padding=(0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = QwenImage21CausalConv3d(
                dim, dim, (1, 1), stride=(1, 1), padding=(0, 0)
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
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
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
                if cache_x.shape[2] < 2 and feat_cache[idx] == "Rep":
                    cache_x = torch.cat(
                        [torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2
                    )
                previous = feat_cache[idx]
                x = self.time_conv(x, None if previous == "Rep" else previous)
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                x = x.reshape(batch, 2, channels, frames, height, width)
                x = torch.stack((x[:, 0], x[:, 1]), 3)
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


class QwenImage21ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = QwenImage21RMSNorm(in_dim, images=False)
        self.conv1 = QwenImage21CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImage21RMSNorm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImage21CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            QwenImage21CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    @staticmethod
    def _cached_conv(
        conv: QwenImage21CausalConv3d,
        x: Tensor,
        feat_cache: FeatureCache,
        feat_idx: list[int],
    ) -> Tensor:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
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


class QwenImage21AttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.norm = QwenImage21RMSNorm(dim)
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


class QwenImage21MidBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, num_layers: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self.attentions = nn.ModuleList(QwenImage21AttentionBlock(dim) for _ in range(num_layers))
        self.resnets = nn.ModuleList(
            [QwenImage21ResidualBlock(dim, dim, dropout)]
            + [QwenImage21ResidualBlock(dim, dim, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: Tensor,
        feat_cache: FeatureCache | None = None,
        feat_idx: list[int] | None = None,
    ) -> Tensor:
        if feat_idx is None:
            feat_idx = [0]
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
        for attention, resnet in zip(self.attentions, self.resnets[1:], strict=True):
            x = resnet(attention(x), feat_cache=feat_cache, feat_idx=feat_idx)
        return x


class QwenImage21ResidualDownBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        num_res_blocks: int,
        temperal_downsample: bool = False,
        down_flag: bool = False,
    ) -> None:
        super().__init__()
        self.avg_shortcut = QwenImage21AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(QwenImage21ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler: QwenImage21Resample | None = QwenImage21Resample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(
        self,
        x: Tensor,
        feat_cache: FeatureCache | None = None,
        feat_idx: list[int] | None = None,
    ) -> Tensor:
        if feat_idx is None:
            feat_idx = [0]
        shortcut = x.clone()
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x + self.avg_shortcut(shortcut)


class QwenImage21Encoder3d(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = config["base_dim"]
        z_dim = config["z_dim"]
        dim_mult = config["dim_mult"]
        num_res_blocks = config["num_res_blocks"]
        dropout = config["dropout"]
        temperal_downsample = config["temperal_downsample"]
        dims = [dim * value for value in [1] + dim_mult]
        self.conv_in = QwenImage21CausalConv3d(config["in_channels"], dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            self.down_blocks.append(
                QwenImage21ResidualDownBlock(
                    in_dim,
                    out_dim,
                    dropout,
                    num_res_blocks,
                    temperal_downsample=temperal_downsample[i] if i != len(dim_mult) - 1 else False,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
        self.mid_block = QwenImage21MidBlock(out_dim, dropout, num_layers=1)
        self.norm_out = QwenImage21RMSNorm(out_dim, images=False)
        self.conv_out = QwenImage21CausalConv3d(out_dim, z_dim * 2, 3, padding=1)

    @staticmethod
    def _cached_conv(
        conv: QwenImage21CausalConv3d,
        x: Tensor,
        feat_cache: FeatureCache,
        feat_idx: list[int],
    ) -> Tensor:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
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
        x = (
            self.conv_in(x)
            if feat_cache is None
            else self._cached_conv(self.conv_in, x, feat_cache, feat_idx)
        )
        for layer in self.down_blocks:
            x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x = F.silu(self.norm_out(x))
        return (
            self.conv_out(x)
            if feat_cache is None
            else self._cached_conv(self.conv_out, x, feat_cache, feat_idx)
        )


class QwenImage21ResidualUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.avg_shortcut = (
            QwenImage21DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
            if up_flag
            else None
        )
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImage21ResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler: QwenImage21Resample | None = QwenImage21Resample(
                out_dim, mode=mode, upsample_out_dim=out_dim
            )
        else:
            self.upsampler = None

    def forward(
        self,
        x: Tensor,
        feat_cache: FeatureCache | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> Tensor:
        if feat_idx is None:
            feat_idx = [0]
        shortcut = x.clone()
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.upsampler is not None:
            x = self.upsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(shortcut, first_chunk=first_chunk)
        return x


class QwenImage21Decoder3d(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = config["decoder_base_dim"]
        z_dim = config["z_dim"]
        dim_mult = config["dim_mult"]
        num_res_blocks = config["num_res_blocks"]
        dropout = config["dropout"]
        temperal_upsample = list(reversed(config["temperal_downsample"]))
        dims = [dim * value for value in [dim_mult[-1]] + list(reversed(dim_mult))]
        self.conv_in = QwenImage21CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImage21MidBlock(dims[0], dropout, num_layers=1)
        self.up_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            up_flag = i != len(dim_mult) - 1
            self.up_blocks.append(
                QwenImage21ResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                )
            )
        self.norm_out = QwenImage21RMSNorm(out_dim, images=False)
        self.conv_out = QwenImage21CausalConv3d(out_dim, config["out_channels"], 3, padding=1)

    def forward(
        self,
        x: Tensor,
        feat_cache: FeatureCache | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> Tensor:
        if feat_idx is None:
            feat_idx = [0]
        x = (
            QwenImage21Encoder3d._cached_conv(self.conv_in, x, feat_cache, feat_idx)
            if feat_cache is not None
            else self.conv_in(x)
        )
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
        x = F.silu(self.norm_out(x))
        return (
            QwenImage21Encoder3d._cached_conv(self.conv_out, x, feat_cache, feat_idx)
            if feat_cache is not None
            else self.conv_out(x)
        )


class AutoencoderKLQwenImage21(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(VAE_DEFAULTS | dict(config))
        if self.config["decoder_base_dim"] is None:
            self.config["decoder_base_dim"] = self.config["base_dim"]
        z_dim = self.config["z_dim"]
        self.encoder = QwenImage21Encoder3d(self.config)
        self.quant_conv = QwenImage21CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = QwenImage21CausalConv3d(z_dim, z_dim, 1)
        self.decoder = QwenImage21Decoder3d(self.config)
        self._decoder_conv_count = sum(
            isinstance(module, QwenImage21CausalConv3d) for module in self.decoder.modules()
        )
        self._encoder_conv_count = sum(
            isinstance(module, QwenImage21CausalConv3d) for module in self.encoder.modules()
        )

    @property
    def latents_mean(self) -> list[float] | None:
        return self.config["latents_mean"]

    @property
    def latents_std(self) -> list[float] | None:
        return self.config["latents_std"]

    @property
    def scale_factor_spatial(self) -> int:
        return int(self.config["scale_factor_spatial"])

    def encode(self, image: Tensor) -> Tensor:
        """Return the posterior mean for a batch of single images in ``[-1, 1]``."""
        return self.encode_moments(image).chunk(2, dim=1)[0][:, :, 0]

    def encode_moments(self, image: Tensor) -> Tensor:
        """Return posterior mean and log-variance before the single-frame squeeze."""
        feat_cache: FeatureCache = [None] * self._encoder_conv_count
        moments = self.encoder(image.unsqueeze(2), feat_cache=feat_cache, feat_idx=[0])
        return self.quant_conv(moments)[:, :, :1]

    def decode(self, latents: Tensor) -> Tensor:
        """Decode a batch of single-image latents and clamp the result to ``[-1, 1]``."""
        feat_cache: FeatureCache = [None] * self._decoder_conv_count
        hidden = self.post_quant_conv(latents.unsqueeze(2))
        image = self.decoder(hidden, feat_cache=feat_cache, feat_idx=[0], first_chunk=True)
        return torch.clamp(image, min=-1.0, max=1.0)[:, :, 0]
