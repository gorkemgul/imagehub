"""Ideogram 4 KL autoencoder and latent normalization."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from imagehub.checkpoints import NativeModel

IDEOGRAM4_VAE_DEFAULTS: dict[str, Any] = {
    "sample_size": 256,
    "in_channels": 3,
    "out_channels": 3,
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "latent_channels": 32,
    "norm_num_groups": 32,
    "batch_norm_eps": 1e-4,
    "batch_norm_momentum": 0.1,
    "patch_size": [2, 2],
}


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int, groups: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, hidden: Tensor) -> Tensor:
        hidden = self.norm(hidden)
        q = self.q(hidden)
        k = self.k(hidden)
        v = self.v(hidden)
        batch, channels, height, width = q.shape
        q = q.permute(0, 2, 3, 1).reshape(batch, 1, height * width, channels).contiguous()
        k = k.permute(0, 2, 3, 1).reshape(batch, 1, height * width, channels).contiguous()
        v = v.permute(0, 2, 3, 1).reshape(batch, 1, height * width, channels).contiguous()
        hidden = nn.functional.scaled_dot_product_attention(q, k, v)
        return hidden.reshape(batch, height, width, channels).permute(0, 3, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x: Tensor) -> Tensor:
        hidden = x
        hidden = self.norm1(hidden)
        hidden = swish(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden)
        hidden = swish(hidden)
        hidden = self.conv2(hidden)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        channels = config["block_out_channels"]
        layers = config["layers_per_block"]
        groups = config["norm_num_groups"]
        latent_channels = config["latent_channels"]
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.num_resolutions = len(channels)
        self.num_res_blocks = layers
        self.conv_in = nn.Conv2d(config["in_channels"], channels[0], 3, padding=1)
        self.down = nn.ModuleList()
        block_in = channels[0]
        for level, block_out in enumerate(channels):
            down = nn.Module()
            down.block = nn.ModuleList()
            down.attn = nn.ModuleList()
            for _ in range(layers):
                down.block.append(ResnetBlock(block_in, block_out, groups))
                block_in = block_out
            if level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, groups)
        self.mid.attn_1 = AttnBlock(block_in, groups)
        self.mid.block_2 = ResnetBlock(block_in, block_in, groups)
        self.norm_out = nn.GroupNorm(groups, block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * latent_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        states = [self.conv_in(x)]
        for level in range(self.num_resolutions):
            for block in range(self.num_res_blocks):
                hidden = self.down[level].block[block](  # type: ignore[index, operator]
                    states[-1]
                )
                if len(self.down[level].attn) > 0:  # type: ignore[arg-type]
                    hidden = self.down[level].attn[block](hidden)  # type: ignore[index, operator]
                states.append(hidden)
            if level != self.num_resolutions - 1:
                states.append(self.down[level].downsample(states[-1]))  # type: ignore[operator]
        hidden = states[-1]
        hidden = self.mid.block_1(hidden)  # type: ignore[operator]
        hidden = self.mid.attn_1(hidden)  # type: ignore[operator]
        hidden = self.mid.block_2(hidden)  # type: ignore[operator]
        hidden = self.norm_out(hidden)
        hidden = swish(hidden)
        hidden = self.conv_out(hidden)
        return self.quant_conv(hidden)


class Decoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        channels = config["block_out_channels"]
        layers = config["layers_per_block"]
        groups = config["norm_num_groups"]
        latent_channels = config["latent_channels"]
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.num_resolutions = len(channels)
        self.num_res_blocks = layers
        block_in = channels[-1]
        self.conv_in = nn.Conv2d(latent_channels, block_in, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, groups)
        self.mid.attn_1 = AttnBlock(block_in, groups)
        self.mid.block_2 = ResnetBlock(block_in, block_in, groups)
        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            up = nn.Module()
            up.block = nn.ModuleList()
            up.attn = nn.ModuleList()
            block_out = channels[level]
            for _ in range(layers + 1):
                up.block.append(ResnetBlock(block_in, block_out, groups))
                block_in = block_out
            if level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)
        self.norm_out = nn.GroupNorm(groups, block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, config["out_channels"], 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)
        upscale_dtype = next(self.up.parameters()).dtype
        hidden = self.conv_in(z)
        hidden = self.mid.block_1(hidden)  # type: ignore[operator]
        hidden = self.mid.attn_1(hidden)  # type: ignore[operator]
        hidden = self.mid.block_2(hidden)  # type: ignore[operator]
        hidden = hidden.to(upscale_dtype)
        for level in reversed(range(self.num_resolutions)):
            for block in range(self.num_res_blocks + 1):
                hidden = self.up[level].block[block](hidden)  # type: ignore[index, operator]
                if len(self.up[level].attn) > 0:  # type: ignore[arg-type]
                    hidden = self.up[level].attn[block](hidden)  # type: ignore[index, operator]
            if level != 0:
                hidden = self.up[level].upsample(hidden)  # type: ignore[operator]
        hidden = self.norm_out(hidden)
        hidden = swish(hidden)
        return self.conv_out(hidden)


class AutoencoderIdeogram4(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(IDEOGRAM4_VAE_DEFAULTS | dict(config))
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.bn = nn.BatchNorm2d(
            math.prod(self.config["patch_size"]) * self.config["latent_channels"],
            eps=self.config["batch_norm_eps"],
            momentum=self.config["batch_norm_momentum"],
            affine=False,
            track_running_stats=True,
        )

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert the published diffusers-style VAE tensor names."""
        return convert_diffusers_state_dict(state, len(self.config["block_out_channels"]))

    def encode(self, image: Tensor) -> Tensor:
        return self.encoder(image).chunk(2, dim=1)[0]

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)


def convert_diffusers_state_dict(
    source: dict[str, Tensor], num_resolutions: int = 4
) -> dict[str, Tensor]:
    output: dict[str, Tensor] = {}
    for source_key, tensor in source.items():
        destination_key = _rewrite_diffusers_key(source_key, num_resolutions)
        if destination_key is None:
            raise KeyError(f"Unrecognized diffusers VAE state-dict key: {source_key}")
        if (
            ".mid.attn_1." in destination_key
            and destination_key.endswith(".weight")
            and tensor.ndim == 2
        ):
            tensor = tensor.unsqueeze(-1).unsqueeze(-1)
        output[destination_key] = tensor
    return output


def _rewrite_diffusers_key(key: str, num_resolutions: int) -> str | None:
    if key.startswith("bn."):
        return key
    if key.startswith("quant_conv."):
        return key.replace("quant_conv.", "encoder.quant_conv.", 1)
    if key.startswith("post_quant_conv."):
        return key.replace("post_quant_conv.", "decoder.post_quant_conv.", 1)
    if key == "encoder.conv_norm_out.weight":
        return "encoder.norm_out.weight"
    if key == "encoder.conv_norm_out.bias":
        return "encoder.norm_out.bias"
    if key == "decoder.conv_norm_out.weight":
        return "decoder.norm_out.weight"
    if key == "decoder.conv_norm_out.bias":
        return "decoder.norm_out.bias"
    match = re.match(r"^(encoder|decoder)\.mid_block\.resnets\.(\d+)\.(.+)$", key)
    if match:
        side, index, rest = match.group(1), int(match.group(2)), match.group(3)
        rest = rest.replace("conv_shortcut", "nin_shortcut")
        return f"{side}.mid.block_{index + 1}.{rest}"
    match = re.match(r"^(encoder|decoder)\.mid_block\.attentions\.0\.(.+)$", key)
    if match:
        side, rest = match.group(1), match.group(2)
        rest = (
            rest.replace("group_norm.", "norm.")
            .replace("to_q.", "q.")
            .replace("to_k.", "k.")
            .replace("to_v.", "v.")
            .replace("to_out.0.", "proj_out.")
        )
        return f"{side}.mid.attn_1.{rest}"
    match = re.match(r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
    if match:
        level, resnet, rest = match.group(1), match.group(2), match.group(3)
        return (
            f"encoder.down.{level}.block.{resnet}.{rest.replace('conv_shortcut', 'nin_shortcut')}"
        )
    match = re.match(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.(.+)$", key)
    if match:
        return f"encoder.down.{match.group(1)}.downsample.conv.{match.group(2)}"
    match = re.match(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
    if match:
        level = num_resolutions - 1 - int(match.group(1))
        rest = match.group(3).replace("conv_shortcut", "nin_shortcut")
        return f"decoder.up.{level}.block.{match.group(2)}.{rest}"
    match = re.match(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.(.+)$", key)
    if match:
        level = num_resolutions - 1 - int(match.group(1))
        return f"decoder.up.{level}.upsample.conv.{match.group(2)}"
    if key.startswith(
        ("encoder.conv_in.", "encoder.conv_out.", "decoder.conv_in.", "decoder.conv_out.")
    ):
        return key
    return None


LATENT_SHIFT = (
    0.01984364,
    0.10149707,
    0.29689495,
    0.27188619,
    -0.21445648,
    -0.15979549,
    0.05021099,
    -0.15083604,
    -0.15360136,
    -0.20131799,
    0.01922352,
    0.0622626,
    0.10140969,
    -0.06739428,
    0.3758261,
    -0.233712,
    0.35164491,
    -0.02590912,
    -0.0271935,
    -0.10833897,
    -0.1476848,
    -0.01130957,
    -0.2298372,
    0.23526423,
    -0.10893522,
    0.11957631,
    0.04047799,
    0.3134589,
    -0.17225064,
    -0.18646109,
    -0.34691978,
    -0.03571246,
    0.02583857,
    0.10190072,
    0.28402294,
    0.26952152,
    -0.21634675,
    -0.17938656,
    0.04358909,
    -0.15007621,
    -0.1548502,
    -0.18971131,
    0.02710861,
    0.05609494,
    0.10697846,
    -0.06854968,
    0.38167698,
    -0.24269937,
    0.35705471,
    -0.03063305,
    -0.02946109,
    -0.11244286,
    -0.14336038,
    -0.01362137,
    -0.21863696,
    0.23228983,
    -0.11739769,
    0.11693044,
    0.02563311,
    0.31356594,
    -0.17420591,
    -0.19006285,
    -0.34905377,
    -0.04025005,
    0.01924137,
    0.07652984,
    0.2995608,
    0.2628057,
    -0.22011674,
    -0.12715361,
    0.04879879,
    -0.14075719,
    -0.15935895,
    -0.2123584,
    0.01974813,
    0.05523547,
    0.10011992,
    -0.06428964,
    0.37781868,
    -0.21491644,
    0.34254215,
    -0.03153528,
    -0.0310082,
    -0.10761415,
    -0.14730405,
    -0.02475182,
    -0.2285588,
    0.2515081,
    -0.10445128,
    0.12446,
    0.07062869,
    0.30880162,
    -0.18016875,
    -0.18869164,
    -0.34533499,
    -0.0129177,
    0.02578168,
    0.07993659,
    0.28642181,
    0.26038408,
    -0.22459419,
    -0.14820155,
    0.04059549,
    -0.14043529,
    -0.16111187,
    -0.2020305,
    0.02602069,
    0.04852717,
    0.10432153,
    -0.06309942,
    0.38402443,
    -0.22397003,
    0.34814481,
    -0.03774432,
    -0.03381438,
    -0.11245691,
    -0.14128767,
    -0.02853208,
    -0.21752016,
    0.24872463,
    -0.11399775,
    0.1222687,
    0.05620835,
    0.309178,
    -0.18065738,
    -0.19401479,
    -0.34495114,
    -0.01760592,
)
LATENT_SCALE = (
    1.63933691,
    1.70204478,
    1.73642566,
    1.90004803,
    1.6675316,
    1.69059584,
    1.56853198,
    1.62314944,
    1.89106626,
    1.58086668,
    1.60822129,
    1.60962993,
    1.63322129,
    1.56074359,
    1.73419528,
    1.7919265,
    1.64040632,
    1.66802808,
    1.60390303,
    1.75480492,
    1.63187587,
    1.64334594,
    1.61722884,
    1.60146046,
    1.63459219,
    1.55291476,
    1.68771497,
    1.68415657,
    1.78966054,
    1.66631641,
    1.65626686,
    1.65976433,
    1.63487607,
    1.69513249,
    1.72933756,
    1.91310663,
    1.67035057,
    1.72286863,
    1.56719251,
    1.61934825,
    1.88628859,
    1.56911539,
    1.59455129,
    1.60829869,
    1.62470611,
    1.56052853,
    1.73677003,
    1.77563606,
    1.63732541,
    1.66370527,
    1.59508952,
    1.75153949,
    1.63029275,
    1.64517667,
    1.61659342,
    1.59722044,
    1.64103121,
    1.5408531,
    1.68610394,
    1.67772755,
    1.78998563,
    1.66621713,
    1.65458955,
    1.66041308,
    1.64710857,
    1.68163503,
    1.74000294,
    1.92784786,
    1.67411194,
    1.67395548,
    1.57406532,
    1.62199356,
    1.87618195,
    1.5584375,
    1.57438785,
    1.61711053,
    1.63094305,
    1.55644029,
    1.73124302,
    1.80666627,
    1.6463621,
    1.65932006,
    1.60816188,
    1.75682671,
    1.64695873,
    1.63121722,
    1.61380832,
    1.60478651,
    1.63396035,
    1.53505068,
    1.65534289,
    1.67132281,
    1.80317197,
    1.6767314,
    1.65700938,
    1.68426259,
    1.65339716,
    1.67540638,
    1.73298504,
    1.94067348,
    1.67893609,
    1.70635117,
    1.5730906,
    1.61928553,
    1.87148809,
    1.56244866,
    1.56697152,
    1.61584394,
    1.62759496,
    1.55480378,
    1.73484107,
    1.79055143,
    1.64688773,
    1.66121492,
    1.60135887,
    1.75254572,
    1.64798332,
    1.62989921,
    1.61381592,
    1.60792883,
    1.63939668,
    1.53075757,
    1.65371318,
    1.66801185,
    1.80029087,
    1.67591476,
    1.65655173,
    1.68533454,
)


def get_latent_norm() -> tuple[Tensor, Tensor]:
    shift = torch.tensor(LATENT_SHIFT, dtype=torch.float32)
    scale = torch.tensor(LATENT_SCALE, dtype=torch.float32)
    assert shift.shape == (128,) and scale.shape == (128,)
    return shift, scale
