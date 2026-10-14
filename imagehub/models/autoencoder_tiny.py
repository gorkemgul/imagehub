"""Tiny distilled autoencoder with diffusers ``AutoencoderTiny`` tensor names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from imagehub.checkpoints import NativeModel


class AutoencoderTinyBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.fuse = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.fuse(self.conv(x) + self.skip(x))


class EncoderTiny(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = cfg["encoder_block_out_channels"]
        for index, count in enumerate(cfg["num_encoder_blocks"]):
            if index == 0:
                layers.append(nn.Conv2d(cfg["in_channels"], channels[index], 3, padding=1))
            else:
                layers.append(
                    nn.Conv2d(channels[index], channels[index], 3, stride=2, padding=1, bias=False)
                )
            layers.extend(
                AutoencoderTinyBlock(channels[index], channels[index]) for _ in range(count)
            )
        layers.append(nn.Conv2d(channels[-1], cfg["latent_channels"], 3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x.add(1).div(2))


class DecoderTiny(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        channels = cfg["decoder_block_out_channels"]
        layers: list[nn.Module] = [
            nn.Conv2d(cfg["latent_channels"], channels[0], 3, padding=1),
            nn.ReLU(),
        ]
        counts = cfg["num_decoder_blocks"]
        for index, count in enumerate(counts):
            layers.extend(
                AutoencoderTinyBlock(channels[index], channels[index]) for _ in range(count)
            )
            final = index == len(counts) - 1
            if not final:
                layers.append(
                    nn.Upsample(
                        scale_factor=cfg["upsampling_scaling_factor"], mode=cfg["upsample_fn"]
                    )
                )
            layers.append(
                nn.Conv2d(
                    channels[index],
                    cfg["out_channels"] if final else channels[index],
                    3,
                    padding=1,
                    bias=final,
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(torch.tanh(x / 3) * 3).mul(2).sub(1)


_DEFAULTS: dict[str, Any] = {
    "in_channels": 3,
    "out_channels": 3,
    "encoder_block_out_channels": [64, 64, 64, 64],
    "decoder_block_out_channels": [64, 64, 64, 64],
    "latent_channels": 4,
    "upsampling_scaling_factor": 2,
    "num_encoder_blocks": [1, 3, 3, 3],
    "num_decoder_blocks": [3, 3, 3, 1],
    "latent_magnitude": 3,
    "latent_shift": 0.5,
    "scaling_factor": 1.0,
    "shift_factor": 0.0,
    "act_fn": "relu",
    "upsample_fn": "nearest",
    "force_upcast": False,
}


class AutoencoderTiny(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(_DEFAULTS | dict(config))
        if self.config["act_fn"] != "relu" or self.config["upsample_fn"] != "nearest":
            raise ValueError("Unsupported TinyVAE activation or upsampling mode")
        self.encoder = EncoderTiny(self.config)
        self.decoder = DecoderTiny(self.config)

    @property
    def scaling_factor(self) -> float:
        return float(self.config["scaling_factor"])

    @property
    def shift_factor(self) -> float:
        return float(self.config["shift_factor"])

    def encode(self, image: Tensor) -> Tensor:
        return self.encoder(image)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def scale_latents(self, latents: Tensor) -> Tensor:
        return (
            latents.div(2 * self.config["latent_magnitude"])
            .add(self.config["latent_shift"])
            .clamp(0, 1)
        )

    def unscale_latents(self, latents: Tensor) -> Tensor:
        return latents.sub(self.config["latent_shift"]).mul(2 * self.config["latent_magnitude"])
