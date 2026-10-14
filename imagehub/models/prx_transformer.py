"""Native PRX denoiser transformer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

PRX_TRANSFORMER_DEFAULTS: dict[str, Any] = {
    "in_channels": 16,
    "patch_size": 2,
    "context_in_dim": 2304,
    "hidden_size": 1792,
    "mlp_ratio": 3.5,
    "num_heads": 28,
    "depth": 16,
    "axes_dim": (32, 32),
    "theta": 10000,
    "time_factor": 1000.0,
    "time_max_period": 10000,
}


def _timestep_embedding(timesteps: Tensor, dimension: int, max_period: int, scale: float) -> Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    values = scale * (timesteps.float()[:, None] * frequencies[None])
    result = torch.cat((values.cos(), values.sin()), dim=-1)
    return F.pad(result, (0, 1)) if dimension % 2 else result


def _image_ids(batch: int, height: int, width: int, patch: int, device: torch.device) -> Tensor:
    ids = torch.zeros(height // patch, width // patch, 2, device=device)
    ids[..., 0] = torch.arange(height // patch, device=device)[:, None]
    ids[..., 1] = torch.arange(width // patch, device=device)[None]
    return ids.reshape(-1, 2).unsqueeze(0).repeat(batch, 1, 1)


def _rope(ids: Tensor, dimensions: tuple[int, ...], theta: int) -> Tensor:
    embeddings = []
    for index, dimension in enumerate(dimensions):
        scale = torch.arange(0, dimension, 2, dtype=torch.float64, device=ids.device) / dimension
        frequencies = ids[:, :, index].unsqueeze(-1) / (theta**scale)
        embeddings.append(
            torch.stack(
                (frequencies.cos(), -frequencies.sin(), frequencies.sin(), frequencies.cos()),
                dim=-1,
            ).reshape(*frequencies.shape, 2, 2)
        )
    return torch.cat(embeddings, dim=-3).unsqueeze(1).float()


def _apply_rope(values: Tensor, frequencies: Tensor) -> Tensor:
    pairs = values.float().reshape(*values.shape[:-1], -1, 1, 2)
    frequencies = frequencies.to(device=values.device, dtype=pairs.dtype)
    output = frequencies[..., 0] * pairs[..., 0] + frequencies[..., 1] * pairs[..., 1]
    return output.reshape_as(values).type_as(values)


def _patchify(values: Tensor, patch: int) -> Tensor:
    batch, channels, height, width = values.shape
    return (
        values.reshape(batch, channels, height // patch, patch, width // patch, patch)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, -1, channels * patch * patch)
    )


def _unpatchify(values: Tensor, patch: int, height: int, width: int) -> Tensor:
    batch, _, features = values.shape
    channels = features // (patch * patch)
    return (
        values.reshape(batch, height // patch, width // patch, channels, patch, patch)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels, height, width)
    )


class _MLPEmbedder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.in_layer = nn.Linear(input_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, values: Tensor) -> Tensor:
        return self.out_layer(F.silu(self.in_layer(values)))


class _RMSNorm(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, values: Tensor) -> Tensor:
        variance = values.float().pow(2).mean(-1, keepdim=True)
        return values * torch.rsqrt(variance + 1e-6) * self.weight


class _Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, hidden_size // heads
        self.img_qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.norm_q = _RMSNorm(self.head_dim)
        self.norm_k = _RMSNorm(self.head_dim)
        self.txt_kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.norm_added_k = _RMSNorm(self.head_dim)
        self.to_out = nn.ModuleList(
            (nn.Linear(hidden_size, hidden_size, bias=False), nn.Dropout(0.0))
        )

    def forward(self, image: Tensor, text: Tensor, mask: Tensor | None, rope: Tensor) -> Tensor:
        batch, image_length, _ = image.shape
        qkv = (
            self.img_qkv_proj(image)
            .reshape(batch, image_length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        query, key = _apply_rope(self.norm_q(query), rope), _apply_rope(self.norm_k(key), rope)
        text_length = text.shape[1]
        text_key, text_value = (
            self.txt_kv_proj(text)
            .reshape(batch, text_length, 2, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        key = torch.cat((self.norm_added_k(text_key), key), dim=2)
        value = torch.cat((text_value, value), dim=2)
        attention_mask = None
        if mask is not None:
            attention_mask = torch.cat(
                (
                    mask.to(torch.bool),
                    torch.ones(batch, image_length, dtype=torch.bool, device=image.device),
                ),
                dim=-1,
            )[:, None, None]
        output = (
            F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0
            )
            .transpose(1, 2)
            .reshape(batch, image_length, -1)
        )
        return self.to_out[1](self.to_out[0](output))


class _Modulation(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden_size, 6 * hidden_size)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)


class _Block(nn.Module):
    def __init__(self, hidden_size: int, heads: int, ratio: float) -> None:
        super().__init__()
        intermediate = int(hidden_size * ratio)
        self.img_pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention = _Attention(hidden_size, heads)
        self.post_attention_layernorm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)
        self.modulation = _Modulation(hidden_size)

    def forward(
        self, image: Tensor, text: Tensor, conditioning: Tensor, rope: Tensor, mask: Tensor | None
    ) -> Tensor:
        attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate = self.modulation.lin(
            F.silu(conditioning)
        )[:, None].chunk(6, dim=-1)
        image = image + attn_gate * self.attention(
            (1 + attn_scale) * self.img_pre_norm(image) + attn_shift, text, mask, rope
        )
        values = (1 + mlp_scale) * self.post_attention_layernorm(image) + mlp_shift
        return image + mlp_gate * self.down_proj(
            F.gelu(self.gate_proj(values), approximate="tanh") * self.up_proj(values)
        )


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_size)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, values: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
        return self.linear((1 + scale[:, None]) * self.norm_final(values) + shift[:, None])


class PRXTransformer2DModel(NativeModel):
    """PRX image-latent denoiser with published tensor names."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = PRX_TRANSFORMER_DEFAULTS | {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        cfg["axes_dim"] = tuple(cfg["axes_dim"])
        if (
            cfg["hidden_size"] % cfg["num_heads"]
            or sum(cfg["axes_dim"]) != cfg["hidden_size"] // cfg["num_heads"]
        ):
            raise ValueError("Unsupported PRX transformer configuration")
        super().__init__(cfg)
        patch_features = cfg["in_channels"] * cfg["patch_size"] ** 2
        self.img_in = nn.Linear(patch_features, cfg["hidden_size"])
        self.time_in = _MLPEmbedder(256, cfg["hidden_size"])
        self.txt_in = nn.Linear(cfg["context_in_dim"], cfg["hidden_size"])
        self.blocks = nn.ModuleList(
            _Block(cfg["hidden_size"], cfg["num_heads"], cfg["mlp_ratio"])
            for _ in range(cfg["depth"])
        )
        self.final_layer = _FinalLayer(cfg["hidden_size"], patch_features)

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch, _, height, width = hidden_states.shape
        image = self.img_in(_patchify(hidden_states, self.config["patch_size"]))
        text = self.txt_in(encoder_hidden_states)
        conditioning = self.time_in(
            _timestep_embedding(
                timestep, 256, self.config["time_max_period"], self.config["time_factor"]
            ).to(image.dtype)
        )
        rope = _rope(
            _image_ids(batch, height, width, self.config["patch_size"], hidden_states.device),
            self.config["axes_dim"],
            self.config["theta"],
        )
        for block in self.blocks:
            image = block(image, text, conditioning, rope, attention_mask)
        return _unpatchify(
            self.final_layer(image, conditioning), self.config["patch_size"], height, width
        )
