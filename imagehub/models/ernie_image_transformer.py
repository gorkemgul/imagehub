"""Native ERNIE-Image single-stream diffusion transformer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.layers import timestep_embedding

ERNIE_IMAGE_TRANSFORMER_DEFAULTS: dict[str, Any] = {
    "hidden_size": 3072,
    "num_attention_heads": 24,
    "num_layers": 24,
    "ffn_hidden_size": 8192,
    "in_channels": 128,
    "out_channels": 128,
    "patch_size": 1,
    "text_in_dim": 2560,
    "rope_theta": 256,
    "rope_axes_dim": (32, 48, 48),
    "eps": 1e-6,
    "qk_layernorm": True,
}


def _rope(positions: Tensor, dim: int, theta: float) -> Tensor:
    if dim % 2:
        raise ValueError("rope dimension must be even")
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device) / dim
    omega = 1.0 / (theta**scale)
    return torch.einsum("...n,d->...nd", positions, omega).float()


class ErnieImageRotaryEmbedding(nn.Module):
    def __init__(self, theta: float, axes_dim: Sequence[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)], dim=-1
        )
        emb = emb.unsqueeze(2)
        return torch.stack([emb, emb], dim=-1).reshape(*emb.shape[:-1], -1)


class ErnieImagePatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        batch, dim, height, width = x.shape
        return x.reshape(batch, dim, height * width).transpose(1, 2).contiguous()


class ErnieImageRMSNorm(nn.Module):
    """Root-mean-square norm with the float32 variance of the reference implementation."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight


def _apply_rotary(x_in: Tensor, freqs: Tensor) -> Tensor:
    rot_dim = freqs.shape[-1]
    x, x_pass = x_in[..., :rot_dim], x_in[..., rot_dim:]
    cos = torch.cos(freqs).to(x.dtype)
    sin = torch.sin(freqs).to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return torch.cat((x * cos + rotated * sin, x_pass), dim=-1)


class ErnieImageAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, eps: float) -> None:
        super().__init__()
        head_dim = hidden_size // heads
        self.heads = heads
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False)])

    def forward(self, hidden_states: Tensor, mask: Tensor, rotary: Tensor) -> Tensor:
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))
        query = self.norm_q(query)
        key = self.norm_k(key)
        query = _apply_rotary(query, rotary)
        key = _apply_rotary(key, rotary)
        query, key, value = (item.permute(0, 2, 1, 3) for item in (query, key, value))
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0)
        out = out.permute(0, 2, 1, 3).flatten(2, 3)
        return self.to_out[0](out)


class ErnieImageFeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_fc2(self.up_proj(x) * F.gelu(self.gate_proj(x)))


class ErnieImageBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, ffn_hidden_size: int, eps: float) -> None:
        super().__init__()
        self.adaLN_sa_ln = ErnieImageRMSNorm(hidden_size, eps)
        self.self_attention = ErnieImageAttention(hidden_size, heads, eps)
        self.adaLN_mlp_ln = ErnieImageRMSNorm(hidden_size, eps)
        self.mlp = ErnieImageFeedForward(hidden_size, ffn_hidden_size)

    def forward(
        self, x: Tensor, rotary: Tensor, modulation: Sequence[Tensor], mask: Tensor
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        residual = x
        x = self.adaLN_sa_ln(x)
        x = (x.float() * (1 + scale_msa.float()) + shift_msa.float()).to(x.dtype)
        attn_out = self.self_attention(x.permute(1, 0, 2), mask, rotary).permute(1, 0, 2)
        x = residual + (gate_msa.float() * attn_out.float()).to(x.dtype)
        residual = x
        x = self.adaLN_mlp_ln(x)
        x = (x.float() * (1 + scale_mlp.float()) + shift_mlp.float()).to(x.dtype)
        return residual + (gate_mlp.float() * self.mlp(x).float()).to(x.dtype)


class ErnieImageAdaLNContinuous(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)
        self.linear = nn.Linear(hidden_size, hidden_size * 2)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        scale, shift = self.linear(conditioning).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


class _TimestepEmbedding(nn.Module):
    def __init__(self, channels: int, dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(channels, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, sample: Tensor) -> Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class ErnieImageTransformer2DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(ERNIE_IMAGE_TRANSFORMER_DEFAULTS | dict(config))
        c = self.config
        hidden = c["hidden_size"]
        heads = c["num_attention_heads"]
        if hidden % heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if not c["qk_layernorm"]:
            raise ValueError("Unsupported ERNIE-Image setting qk_layernorm=False")
        if c["patch_size"] != 1:
            raise ValueError("Unsupported ERNIE-Image patch_size; the published models use 1")
        self.head_dim = hidden // heads
        if sum(c["rope_axes_dim"]) != self.head_dim or len(c["rope_axes_dim"]) != 3:
            raise ValueError("rope_axes_dim must have three entries summing to the head dimension")
        self.x_embedder = ErnieImagePatchEmbed(c["in_channels"], hidden, c["patch_size"])
        self.text_proj = (
            nn.Linear(c["text_in_dim"], hidden, bias=False)
            if c["text_in_dim"] != hidden
            else None
        )
        self.time_embedding = _TimestepEmbedding(hidden, hidden)
        self.pos_embed = ErnieImageRotaryEmbedding(c["rope_theta"], c["rope_axes_dim"])
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        self.layers = nn.ModuleList(
            ErnieImageBlock(hidden, heads, c["ffn_hidden_size"], c["eps"])
            for _ in range(c["num_layers"])
        )
        self.final_norm = ErnieImageAdaLNContinuous(hidden, c["eps"])
        self.final_linear = nn.Linear(hidden, c["patch_size"] ** 2 * c["out_channels"])

    def forward(
        self, hidden_states: Tensor, timestep: Tensor, text_bth: Tensor, text_lens: Tensor
    ) -> Tensor:
        """Predict the flow velocity for ``(batch, channels, height, width)`` latents."""
        device, dtype = hidden_states.device, hidden_states.dtype
        batch, _, height, width = hidden_states.shape
        patch = self.config["patch_size"]
        grid_h, grid_w = height // patch, width // patch
        image_tokens = grid_h * grid_w

        img_sbh = self.x_embedder(hidden_states).transpose(0, 1).contiguous()
        if self.text_proj is not None and text_bth.numel() > 0:
            text_bth = self.text_proj(text_bth)
        text_max = text_bth.shape[1]
        text_sbh = text_bth.transpose(0, 1).contiguous()

        x = torch.cat([img_sbh, text_sbh], dim=0)
        seq = x.shape[0]

        text_ids = (
            torch.cat(
                [
                    torch.arange(text_max, device=device, dtype=torch.float32)
                    .view(1, text_max, 1)
                    .expand(batch, -1, -1),
                    torch.zeros((batch, text_max, 2), device=device),
                ],
                dim=-1,
            )
            if text_max > 0
            else torch.zeros((batch, 0, 3), device=device)
        )
        grid_yx = torch.stack(
            torch.meshgrid(
                torch.arange(grid_h, device=device, dtype=torch.float32),
                torch.arange(grid_w, device=device, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        image_ids = torch.cat(
            [
                text_lens.float().view(batch, 1, 1).expand(-1, image_tokens, -1),
                grid_yx.view(1, image_tokens, 2).expand(batch, -1, -1),
            ],
            dim=-1,
        )
        rotary = self.pos_embed(torch.cat([image_ids, text_ids], dim=1))

        valid_text = (
            torch.arange(text_max, device=device).view(1, text_max) < text_lens.view(batch, 1)
            if text_max > 0
            else torch.zeros((batch, 0), device=device, dtype=torch.bool)
        )
        mask = torch.cat(
            [torch.ones((batch, image_tokens), device=device, dtype=torch.bool), valid_text], dim=1
        )[:, None, None, :]

        sample = timestep_embedding(
            timestep, self.config["hidden_size"], flip_sin_to_cos=False, shift=0
        ).to(dtype=dtype)
        c = self.time_embedding(sample)
        modulation = [
            t.unsqueeze(0).expand(seq, -1, -1).contiguous()
            for t in self.adaLN_modulation(c).chunk(6, dim=-1)
        ]
        for layer in self.layers:
            x = layer(x, rotary, modulation, mask)
        x = self.final_norm(x, c).type_as(x)
        patches = self.final_linear(x)[:image_tokens].transpose(0, 1).contiguous()
        out_channels = self.config["out_channels"]
        return (
            patches.view(batch, grid_h, grid_w, patch, patch, out_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(batch, out_channels, height, width)
        )
