"""Native DC-AE autoencoder as used by Sana (RMS norm, SiLU, convolutional resampling)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

_BLOCKS = ("ResBlock", "EfficientViTBlock")
DC_AE_DEFAULTS: dict[str, Any] = {
    "in_channels": 3,
    "latent_channels": 32,
    "attention_head_dim": 32,
    "encoder_block_types": "ResBlock",
    "decoder_block_types": "ResBlock",
    "encoder_block_out_channels": (128, 256, 512, 512, 1024, 1024),
    "decoder_block_out_channels": (128, 256, 512, 512, 1024, 1024),
    "encoder_layers_per_block": (2, 2, 2, 3, 3, 3),
    "decoder_layers_per_block": (3, 3, 3, 3, 3, 3),
    "encoder_qkv_multiscales": ((), (), (), (5,), (5,), (5,)),
    "decoder_qkv_multiscales": ((), (), (), (5,), (5,), (5,)),
    "upsample_block_type": "interpolate",
    "downsample_block_type": "Conv",
    "decoder_norm_types": "rms_norm",
    "decoder_act_fns": "silu",
    "encoder_out_shortcut": True,
    "decoder_in_shortcut": True,
    "decoder_conv_act_fn": "relu",
    "scaling_factor": 1.0,
}
_REQUIRED = {
    "upsample_block_type": "interpolate",
    "downsample_block_type": "Conv",
    "decoder_norm_types": "rms_norm",
    "decoder_act_fns": "silu",
    "encoder_out_shortcut": True,
    "decoder_in_shortcut": True,
    "decoder_conv_act_fn": "relu",
}


class RMSNorm(nn.Module):
    """RMS norm over the last dimension with weight and bias."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight + self.bias


def _channel_norm(norm: RMSNorm, hidden_states: Tensor) -> Tensor:
    return norm(hidden_states.movedim(1, -1)).movedim(-1, 1)


class GLUMBConv(nn.Module):
    def __init__(self, channels: int, expand_ratio: float = 4) -> None:
        super().__init__()
        hidden = int(expand_ratio * channels)
        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(channels, hidden * 2, 1, 1, 0)
        self.conv_depth = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2)
        self.conv_point = nn.Conv2d(hidden, channels, 1, 1, 0, bias=False)
        self.norm = RMSNorm(channels)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.nonlinearity(self.conv_inverted(hidden_states))
        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)
        hidden_states = self.conv_point(hidden_states)
        hidden_states = _channel_norm(self.norm, hidden_states)
        return hidden_states + residual


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.nonlinearity = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.norm = RMSNorm(channels)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = _channel_norm(self.norm, hidden_states)
        return hidden_states + residual


class MultiscaleProjection(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, kernel_size: int) -> None:
        super().__init__()
        channels = 3 * in_channels
        self.proj_in = nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size // 2, groups=channels, bias=False
        )
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0, groups=3 * num_heads, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.proj_out(self.proj_in(hidden_states))


class MultiscaleLinearAttention(nn.Module):
    eps = 1e-15

    def __init__(self, channels: int, head_dim: int, kernel_sizes: Sequence[int]) -> None:
        super().__init__()
        self.head_dim = head_dim
        num_heads = int(channels // head_dim * 1.0)
        inner_dim = num_heads * head_dim
        self.to_q = nn.Linear(channels, inner_dim, bias=False)
        self.to_k = nn.Linear(channels, inner_dim, bias=False)
        self.to_v = nn.Linear(channels, inner_dim, bias=False)
        self.to_qkv_multiscale = nn.ModuleList(
            [MultiscaleProjection(inner_dim, num_heads, size) for size in kernel_sizes]
        )
        self.nonlinearity = nn.ReLU()
        self.to_out = nn.Linear(inner_dim * (1 + len(kernel_sizes)), channels, bias=False)
        self.norm_out = RMSNorm(channels)

    def _linear(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)
        scores = torch.matmul(value, key.transpose(-1, -2))
        hidden_states = torch.matmul(scores, query)
        hidden_states = hidden_states.to(dtype=torch.float32)
        return hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)

    def _quadratic(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        scores = torch.matmul(key.transpose(-1, -2), query)
        scores = scores.to(dtype=torch.float32)
        scores = scores / (torch.sum(scores, dim=2, keepdim=True) + self.eps)
        return torch.matmul(value, scores.to(value.dtype))

    def forward(self, hidden_states: Tensor) -> Tensor:
        height, width = hidden_states.shape[-2:]
        use_linear = height * width > self.head_dim
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        original_dtype = hidden_states.dtype

        hidden_states = hidden_states.movedim(1, -1)
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        hidden_states = torch.cat([query, key, value], dim=3)
        hidden_states = hidden_states.movedim(-1, 1)

        multi_scale = [hidden_states]
        for block in self.to_qkv_multiscale:
            multi_scale.append(block(hidden_states))
        hidden_states = torch.cat(multi_scale, dim=1)

        if use_linear:
            hidden_states = hidden_states.to(dtype=torch.float32)
        hidden_states = hidden_states.reshape(batch_size, -1, 3 * self.head_dim, height * width)
        query, key, value = hidden_states.chunk(3, dim=2)
        query = self.nonlinearity(query)
        key = self.nonlinearity(key)

        if use_linear:
            hidden_states = self._linear(query, key, value).to(dtype=original_dtype)
        else:
            hidden_states = self._quadratic(query, key, value)

        hidden_states = torch.reshape(hidden_states, (batch_size, -1, height, width))
        hidden_states = self.to_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = _channel_norm(self.norm_out, hidden_states)
        return hidden_states + residual


class EfficientViTBlock(nn.Module):
    def __init__(self, channels: int, head_dim: int, kernel_sizes: Sequence[int]) -> None:
        super().__init__()
        self.attn = MultiscaleLinearAttention(channels, head_dim, kernel_sizes)
        self.conv_out = GLUMBConv(channels)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.conv_out(self.attn(hidden_states))


def _block(kind: str, channels: int, head_dim: int, kernel_sizes: Sequence[int]) -> nn.Module:
    if kind == "ResBlock":
        return ResBlock(channels)
    return EfficientViTBlock(channels, head_dim, kernel_sizes)


class DownBlock(nn.Module):
    """Strided 3x3 convolution with an average-pooled pixel-unshuffle shortcut."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.group_size = in_channels * 4 // out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = self.conv(hidden_states)
        y = F.pixel_unshuffle(hidden_states, 2)
        y = y.unflatten(1, (-1, self.group_size))
        y = y.mean(dim=2)
        return x + y


class UpBlock(nn.Module):
    """Nearest upsampling followed by a 3x3 convolution, with a repeated-channel shortcut."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.repeats = out_channels * 4 // in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = F.interpolate(hidden_states, scale_factor=2, mode="nearest")
        x = self.conv(x)
        y = hidden_states.repeat_interleave(
            self.repeats, dim=1, output_size=hidden_states.shape[1] * self.repeats
        )
        y = F.pixel_shuffle(y, 2)
        return x + y


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        head_dim: int,
        block_types: Sequence[str],
        out_channels: Sequence[int],
        layers: Sequence[int],
        multiscales: Sequence[Sequence[int]],
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding=1)
        down_blocks = []
        for i, (channels, count) in enumerate(zip(out_channels, layers, strict=True)):
            stage: list[nn.Module] = [
                _block(block_types[i], channels, head_dim, multiscales[i]) for _ in range(count)
            ]
            if i < len(out_channels) - 1:
                stage.append(DownBlock(channels, out_channels[i + 1]))
            down_blocks.append(nn.Sequential(*stage))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv_out = nn.Conv2d(out_channels[-1], latent_channels, 3, 1, 1)
        self.group_size = out_channels[-1] // latent_channels

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        x = hidden_states.unflatten(1, (-1, self.group_size))
        x = x.mean(dim=2)
        return self.conv_out(hidden_states) + x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        head_dim: int,
        block_types: Sequence[str],
        out_channels: Sequence[int],
        layers: Sequence[int],
        multiscales: Sequence[Sequence[int]],
    ) -> None:
        super().__init__()
        count = len(out_channels)
        self.conv_in = nn.Conv2d(latent_channels, out_channels[-1], 3, 1, 1)
        self.repeats = out_channels[-1] // latent_channels
        up_blocks: list[nn.Module] = []
        for i in reversed(range(count)):
            stage: list[nn.Module] = []
            if i < count - 1:
                stage.append(UpBlock(out_channels[i + 1], out_channels[i]))
            stage.extend(
                _block(block_types[i], out_channels[i], head_dim, multiscales[i])
                for _ in range(layers[i])
            )
            up_blocks.insert(0, nn.Sequential(*stage))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.norm_out = RMSNorm(out_channels[0])
        self.conv_act = nn.ReLU()
        self.conv_out = nn.Conv2d(out_channels[0], in_channels, 3, 1, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = hidden_states.repeat_interleave(
            self.repeats, dim=1, output_size=hidden_states.shape[1] * self.repeats
        )
        hidden_states = self.conv_in(hidden_states) + x
        for up_block in reversed(self.up_blocks):
            hidden_states = up_block(hidden_states)
        hidden_states = _channel_norm(self.norm_out, hidden_states)
        hidden_states = self.conv_act(hidden_states)
        return self.conv_out(hidden_states)


def _per_stage(value: str | Sequence[str], count: int, name: str) -> tuple[str, ...]:
    stages = (value,) * count if isinstance(value, str) else tuple(value)
    if len(stages) != count:
        raise ValueError(f"{name} must name one block type per stage")
    for kind in stages:
        if kind not in _BLOCKS:
            raise ValueError(f"Unsupported DC-AE block type {kind!r} in {name}")
    return stages


def _check_stages(side: str, channels: Sequence[int], layers: Sequence[int], scales: Any) -> None:
    if not (len(channels) == len(layers) == len(scales)):
        raise ValueError(f"DC-AE {side} stage settings must have the same length")
    if len(channels) < 2:
        raise ValueError(f"DC-AE {side} needs at least two stages")
    if any(count < 1 for count in layers):
        raise ValueError(f"Unsupported DC-AE {side}_layers_per_block: every stage needs a layer")


class AutoencoderDC(NativeModel):
    """DC-AE with published tensor names; ``encode`` and ``decode`` act on tensors."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in DC_AE_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported DC-AE setting {unknown[0]}")
        cfg = DC_AE_DEFAULTS | {k: v for k, v in config.items() if not k.startswith("_")}
        for key, expected in _REQUIRED.items():
            if cfg[key] != expected:
                raise ValueError(f"Unsupported DC-AE setting {key}={cfg[key]!r}")
        enc_channels = tuple(cfg["encoder_block_out_channels"])
        dec_channels = tuple(cfg["decoder_block_out_channels"])
        _check_stages(
            "encoder", enc_channels, cfg["encoder_layers_per_block"], cfg["encoder_qkv_multiscales"]
        )
        _check_stages(
            "decoder", dec_channels, cfg["decoder_layers_per_block"], cfg["decoder_qkv_multiscales"]
        )
        self.scaling_factor = float(cfg["scaling_factor"])
        self.spatial_compression_ratio = 2 ** (len(enc_channels) - 1)
        self.encoder = Encoder(
            cfg["in_channels"],
            cfg["latent_channels"],
            cfg["attention_head_dim"],
            _per_stage(cfg["encoder_block_types"], len(enc_channels), "encoder_block_types"),
            enc_channels,
            tuple(cfg["encoder_layers_per_block"]),
            tuple(tuple(s) for s in cfg["encoder_qkv_multiscales"]),
        )
        self.decoder = Decoder(
            cfg["in_channels"],
            cfg["latent_channels"],
            cfg["attention_head_dim"],
            _per_stage(cfg["decoder_block_types"], len(dec_channels), "decoder_block_types"),
            dec_channels,
            tuple(cfg["decoder_layers_per_block"]),
            tuple(tuple(s) for s in cfg["decoder_qkv_multiscales"]),
        )

    def encode(self, images: Tensor) -> Tensor:
        return self.encoder(images)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)
