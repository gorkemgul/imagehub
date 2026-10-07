"""Native Krea 2 transformer for text-to-image inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

KREA2_TRANSFORMER_DEFAULTS: dict[str, Any] = {
    "in_channels": 64,
    "num_layers": 28,
    "attention_head_dim": 128,
    "num_attention_heads": 48,
    "num_key_value_heads": 12,
    "intermediate_size": 16384,
    "timestep_embed_dim": 256,
    "text_hidden_dim": 2560,
    "num_text_layers": 12,
    "text_num_attention_heads": 20,
    "text_num_key_value_heads": 20,
    "text_intermediate_size": 6912,
    "num_layerwise_text_blocks": 2,
    "num_refiner_text_blocks": 2,
    "axes_dims_rope": (32, 48, 48),
    "rope_theta": 1000.0,
    "norm_eps": 1e-5,
}


def prepare_krea2_position_ids(
    text_seq_len: int, grid_height: int, grid_width: int, device: torch.device
) -> Tensor:
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
    image_ids = image_ids.reshape(grid_height * grid_width, 3)
    return torch.cat([text_ids, image_ids], dim=0)


class Krea2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        hidden_states = F.rms_norm(
            hidden_states.float(), (self.dim,), weight=self.weight + 1.0, eps=self.eps
        )
        return hidden_states.to(dtype)


def _apply_rotary_emb(hidden_states: Tensor, rotary_emb: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = rotary_emb
    cos = cos[None, :, None, :].to(hidden_states.device)
    sin = sin[None, :, None, :].to(hidden_states.device)
    real, imaginary = hidden_states.reshape(*hidden_states.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([-imaginary, real], dim=-1).flatten(3)
    return (hidden_states.float() * cos + rotated.float() * sin).to(hidden_states.dtype)


def _attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None) -> Tensor:
    query, key, value = (item.permute(0, 2, 1, 3) for item in (query, key, value))
    hidden_states = F.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    )
    return hidden_states.permute(0, 2, 1, 3)


class Krea2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads

        self.to_q = nn.Linear(hidden_size, self.head_dim * self.num_heads, bias=False)
        self.to_k = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_v = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = Krea2RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Krea2RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False), nn.Dropout(0.0)]
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        query = self.to_q(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        key = self.to_k(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
        value = self.to_v(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
        gate = self.to_gate(hidden_states)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            query = _apply_rotary_emb(query, image_rotary_emb)
            key = _apply_rotary_emb(key, image_rotary_emb)

        num_key_value_groups = self.num_heads // self.num_kv_heads
        if num_key_value_groups > 1:
            key = key.repeat_interleave(num_key_value_groups, dim=2)
            value = value.repeat_interleave(num_key_value_groups, dim=2)

        hidden_states = _attention(query, key, value, attention_mask)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return self.to_out[0](hidden_states)


class Krea2SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


class Krea2TextFusionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = Krea2RMSNorm(dim, eps=eps)
        self.norm2 = Krea2RMSNorm(dim, eps=eps)
        self.attn = Krea2Attention(dim, num_heads, num_kv_heads, eps=eps)
        self.ff = Krea2SwiGLU(dim, intermediate_size)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), attention_mask=attention_mask
        )
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class Krea2TextFusion(nn.Module):
    def __init__(
        self,
        num_text_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        num_layerwise_blocks: int,
        num_refiner_blocks: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
                for _ in range(num_layerwise_blocks)
            ]
        )
        self.projector = nn.Linear(num_text_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
                for _ in range(num_refiner_blocks)
            ]
        )

    def forward(
        self, encoder_hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        batch_size, seq_len, num_text_layers, dim = encoder_hidden_states.shape

        hidden_states = encoder_hidden_states.reshape(batch_size * seq_len, num_text_layers, dim)
        for block in self.layerwise_blocks:
            hidden_states = block(hidden_states.contiguous())

        hidden_states = hidden_states.reshape(batch_size, seq_len, num_text_layers, dim).permute(
            0, 1, 3, 2
        )
        hidden_states = self.projector(hidden_states).squeeze(-1)

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        return hidden_states


class Krea2TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))
        self.norm1 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Krea2Attention(hidden_size, num_heads, num_kv_heads, eps=norm_eps)
        self.ff = Krea2SwiGLU(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        modulation = temb.unflatten(-1, (6, -1)) + self.scale_shift_table
        prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)

        attn_out = self.attn(
            (1.0 + prescale) * self.norm1(hidden_states) + preshift,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + pregate * attn_out
        ff_out = self.ff((1.0 + postscale) * self.norm2(hidden_states) + postshift)
        hidden_states = hidden_states + postgate * ff_out
        return hidden_states


class Krea2TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = nn.Linear(embed_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, timestep: Tensor, dtype: torch.dtype) -> Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(1e4) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half
        )
        args = (timestep.float() * 1e3)[:, None, None] * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        return self.linear_2(F.gelu(self.linear_1(emb), approximate="tanh"))


class Krea2TextProjection(nn.Module):
    def __init__(self, text_dim: int, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = Krea2RMSNorm(text_dim, eps=eps)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.linear_1(self.norm(hidden_states))
        return self.linear_2(F.gelu(hidden_states, approximate="tanh"))


class Krea2FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, eps: float) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))
        self.norm = Krea2RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, hidden_states: Tensor, temb: Tensor) -> Tensor:
        modulation = temb + self.scale_shift_table
        scale, shift = modulation.chunk(2, dim=1)
        hidden_states = (1.0 + scale) * self.norm(hidden_states) + shift
        return self.linear(hidden_states)


class Krea2RotaryPosEmbed(nn.Module):
    def __init__(self, theta: float, axes_dim: list[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        for index in range(n_axes):
            dim = self.axes_dim[index]
            freqs = 1.0 / (
                self.theta
                ** (torch.arange(0, dim, 2, dtype=torch.float64, device=ids.device) / dim)
            )
            freqs = torch.outer(pos[:, index], freqs)
            cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
            sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class Krea2Transformer2DModel(NativeModel):
    # The reference keeps its float32 tensors at their file offsets (see NativeModel).
    clone_shards = False

    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = KREA2_TRANSFORMER_DEFAULTS | {
            name: value for name, value in config.items() if not name.startswith("_")
        }
        hidden_size = cfg["attention_head_dim"] * cfg["num_attention_heads"]
        if sum(cfg["axes_dims_rope"]) != cfg["attention_head_dim"]:
            raise ValueError(
                f"sum(axes_dims_rope)={sum(cfg['axes_dims_rope'])} must equal "
                f"attention_head_dim={cfg['attention_head_dim']}"
            )
        super().__init__(cfg)

        self.in_channels = cfg["in_channels"]
        self.out_channels = cfg["in_channels"]
        self.hidden_size = hidden_size

        self.img_in = nn.Linear(cfg["in_channels"], hidden_size, bias=True)
        self.time_embed = Krea2TimestepEmbedding(cfg["timestep_embed_dim"], hidden_size)
        self.time_mod_proj = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.text_fusion = Krea2TextFusion(
            num_text_layers=cfg["num_text_layers"],
            dim=cfg["text_hidden_dim"],
            num_heads=cfg["text_num_attention_heads"],
            num_kv_heads=cfg["text_num_key_value_heads"],
            intermediate_size=cfg["text_intermediate_size"],
            num_layerwise_blocks=cfg["num_layerwise_text_blocks"],
            num_refiner_blocks=cfg["num_refiner_text_blocks"],
            eps=cfg["norm_eps"],
        )
        self.txt_in = Krea2TextProjection(cfg["text_hidden_dim"], hidden_size, cfg["norm_eps"])
        self.rotary_emb = Krea2RotaryPosEmbed(
            theta=cfg["rope_theta"], axes_dim=list(cfg["axes_dims_rope"])
        )
        self.transformer_blocks = nn.ModuleList(
            [
                Krea2TransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=cfg["intermediate_size"],
                    num_heads=cfg["num_attention_heads"],
                    num_kv_heads=cfg["num_key_value_heads"],
                    norm_eps=cfg["norm_eps"],
                )
                for _ in range(cfg["num_layers"])
            ]
        )
        self.final_layer = Krea2FinalLayer(
            hidden_size, out_channels=cfg["in_channels"], eps=cfg["norm_eps"]
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        position_ids: Tensor,
        encoder_attention_mask: Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool | None = None,
    ) -> Tensor:
        del attention_kwargs, return_dict
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"`position_ids` must have shape (sequence_length, 3), got "
                f"{tuple(position_ids.shape)}."
            )

        batch_size, image_seq_len, _ = hidden_states.shape
        text_seq_len = encoder_hidden_states.shape[1]

        temb = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))

        text_attention_mask = None
        attention_mask = None
        if encoder_attention_mask is not None:
            text_attention_mask = encoder_attention_mask[:, None, None, :]
            image_mask = encoder_attention_mask.new_ones((batch_size, image_seq_len))
            attention_mask = torch.cat([encoder_attention_mask, image_mask], dim=1)[
                :, None, None, :
            ]

        encoder_hidden_states = self.text_fusion(
            encoder_hidden_states, attention_mask=text_attention_mask
        )
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        hidden_states = self.img_in(hidden_states)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        image_rotary_emb = self.rotary_emb(position_ids)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, temb_mod, image_rotary_emb, attention_mask)

        hidden_states = hidden_states[:, text_seq_len:]
        return self.final_layer(hidden_states, temb)
