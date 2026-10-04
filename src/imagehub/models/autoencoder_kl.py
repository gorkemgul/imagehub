"""KL autoencoder (LDM f=8) with the weight names of the published SD 1.x diffusers checkpoints.

The mid-block attention keeps the legacy names ``query``/``key``/``value``/``proj_attn`` used by
every released SD 1.x VAE file, so checkpoints load strictly without key conversion.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.layers import Downsample, ResnetBlock, Upsample, attention

VAE_DEFAULTS: dict[str, Any] = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "scaling_factor": 0.18215,
}
_EPS = 1e-6


class VAEAttention(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, eps=_EPS)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj_attn = nn.Linear(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        residual = x
        hidden = self.group_norm(x).view(b, c, h * w).transpose(1, 2)
        out = attention(self.query(hidden), self.key(hidden), self.value(hidden), heads=1)
        out = self.proj_attn(out).transpose(1, 2).reshape(b, c, h, w)
        return out + residual


class VAEMidBlock(nn.Module):
    def __init__(self, ch: int, groups: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(ch, ch, temb_channels=None, eps=_EPS, groups=groups) for _ in range(2)
        )
        self.attentions = nn.ModuleList([VAEAttention(ch, groups)])

    def forward(self, x: Tensor) -> Tensor:
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, layers: int, groups: int, down: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(
                in_ch if i == 0 else out_ch, out_ch, temb_channels=None, eps=_EPS, groups=groups
            )
            for i in range(layers)
        )
        self.downsamplers = nn.ModuleList([Downsample(out_ch, padding=0)]) if down else None

    def forward(self, x: Tensor) -> Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, layers: int, groups: int, up: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(
                in_ch if i == 0 else out_ch, out_ch, temb_channels=None, eps=_EPS, groups=groups
            )
            for i in range(layers)
        )
        self.upsamplers = nn.ModuleList([Upsample(out_ch)]) if up else None

    def forward(self, x: Tensor) -> Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        ch, groups, layers = (
            cfg["block_out_channels"],
            cfg["norm_num_groups"],
            cfg["layers_per_block"],
        )
        self.conv_in = nn.Conv2d(cfg["in_channels"], ch[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(
            EncoderBlock(ch[max(i - 1, 0)], ch[i], layers, groups, down=i < len(ch) - 1)
            for i in range(len(ch))
        )
        self.mid_block = VAEMidBlock(ch[-1], groups)
        self.conv_norm_out = nn.GroupNorm(groups, ch[-1], eps=_EPS)
        self.conv_out = nn.Conv2d(ch[-1], 2 * cfg["latent_channels"], 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class Decoder(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        ch, groups, layers = (
            cfg["block_out_channels"],
            cfg["norm_num_groups"],
            cfg["layers_per_block"],
        )
        rev = list(reversed(ch))
        self.conv_in = nn.Conv2d(cfg["latent_channels"], rev[0], 3, padding=1)
        self.mid_block = VAEMidBlock(rev[0], groups)
        self.up_blocks = nn.ModuleList(
            DecoderBlock(rev[max(i - 1, 0)], rev[i], layers + 1, groups, up=i < len(ch) - 1)
            for i in range(len(ch))
        )
        self.conv_norm_out = nn.GroupNorm(groups, ch[0], eps=_EPS)
        self.conv_out = nn.Conv2d(ch[0], cfg["out_channels"], 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        x = self.mid_block(self.conv_in(z))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class AutoencoderKL(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(VAE_DEFAULTS | dict(config))
        latent = self.config["latent_channels"]
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.quant_conv = nn.Conv2d(2 * latent, 2 * latent, 1)
        self.post_quant_conv = nn.Conv2d(latent, latent, 1)

    @property
    def scaling_factor(self) -> float:
        return float(self.config["scaling_factor"])

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert current diffusers VAE attention names to the legacy SD 1.x names."""
        renamed = {}
        for name, value in state.items():
            if ".attentions." in name:
                for current, legacy in (
                    ("to_q", "query"),
                    ("to_k", "key"),
                    ("to_v", "value"),
                    ("to_out.0", "proj_attn"),
                ):
                    name = name.replace(f".{current}.", f".{legacy}.")
            renamed[name] = value
        return renamed

    def encode(self, image: Tensor) -> Tensor:
        """Return the posterior mean (unscaled latents) for images in [-1, 1]."""
        moments = self.quant_conv(self.encoder(image))
        return moments.chunk(2, dim=1)[0]

    def decode(self, latents: Tensor) -> Tensor:
        """Decode unscaled latents to images in roughly [-1, 1]."""
        return self.decoder(self.post_quant_conv(latents))
