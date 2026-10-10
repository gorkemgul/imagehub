"""Native OmniGen transformer: a Phi-3 style decoder over text, input-image and noisy-image tokens."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

OMNIGEN_DEFAULTS: dict[str, Any] = {
    "in_channels": 4,
    "patch_size": 2,
    "hidden_size": 3072,
    "rms_norm_eps": 1e-5,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "intermediate_size": 8192,
    "num_layers": 32,
    "pad_token_id": 32000,
    "vocab_size": 32064,
    "max_position_embeddings": 131072,
    "original_max_position_embeddings": 4096,
    "rope_base": 10000,
    "rope_scaling": None,
    "pos_embed_max_size": 192,
    "time_step_dim": 256,
    "flip_sin_to_cos": True,
    "downscale_freq_shift": 0,
    "timestep_activation_fn": "silu",
}
_NORM_OUT_EPS = 1e-6


def _timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding with the cosine half first (``flip_sin_to_cos``), no frequency shift."""
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")
    half_dim = dim // 2
    exponent = -math.log(10000) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / half_dim
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class RMSNorm(nn.Module):
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


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states)


class PatchEmbed(nn.Module):
    """Two patch projections (noisy image, input image) sharing a cropped 2D position table."""

    def __init__(self, patch_size: int, in_channels: int, embed_dim: int, max_size: int) -> None:
        super().__init__()
        self.output_image_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size
        )
        self.input_image_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size
        )
        self.patch_size = patch_size
        self.pos_embed_max_size = max_size
        # Stored in the checkpoint (a fixed sin-cos table); loaded, never recomputed.
        self.register_buffer("pos_embed", torch.zeros(1, max_size * max_size, embed_dim))

    def _cropped_pos_embed(self, height: int, width: int) -> Tensor:
        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: "
                f"{self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: "
                f"{self.pos_embed_max_size}."
            )
        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        table = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        table = table[:, top : top + height, left : left + width, :]
        return table.reshape(1, -1, table.shape[-1])

    def forward(self, hidden_states: Tensor | Sequence[Tensor], is_input_image: bool) -> Any:
        projection = self.input_image_proj if is_input_image else self.output_image_proj
        if isinstance(hidden_states, Tensor):
            height, width = hidden_states.shape[-2:]
            pos_embed = self._cropped_pos_embed(height, width)
            patched = projection(hidden_states).flatten(2).transpose(1, 2)
            return patched + pos_embed
        out = []
        for sub_latent in hidden_states:
            height, width = sub_latent.shape[-2:]
            patched = projection(sub_latent).flatten(2).transpose(1, 2)
            out.append(patched + self._cropped_pos_embed(height, width))
        return out


class SuScaledRotaryEmbedding(nn.Module):
    """Phi-3 "su" rotary embedding. The cos/sin table of the first batch item serves all items."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        original_max_position_embeddings: int,
        base: int,
        rope_scaling: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.short_factor = list(rope_scaling["short_factor"])
        self.long_factor = list(rope_scaling["long_factor"])

    def forward(self, hidden_states: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        seq_len = torch.max(position_ids) + 1
        factors = (
            self.long_factor
            if seq_len > self.original_max_position_embeddings
            else self.short_factor
        )
        ext_factors = torch.tensor(factors, dtype=torch.float32, device=hidden_states.device)
        inv_freq_shape = (
            torch.arange(0, self.dim, 2, dtype=torch.int64, device=hidden_states.device).float()
            / self.dim
        )
        inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)[0]
        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(
                1 + math.log(scale) / math.log(self.original_max_position_embeddings)
            )
        return emb.cos() * scaling_factor, emb.sin() * scaling_factor


def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    cos = cos[None, None, :, :].to(x.device)
    sin = sin[None, None, :, :].to(x.device)
    x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
    x_rotated = torch.cat([-x_imag, x_real], dim=-1)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False)])

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor, rotary: tuple[Tensor, Tensor]
    ) -> Tensor:
        batch_size, q_len, query_dim = hidden_states.shape
        head_dim = query_dim // self.heads
        query = self.to_q(hidden_states).view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = self.to_k(hidden_states).view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = self.to_v(hidden_states).view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        query = _apply_rotary(query, *rotary)
        key = _apply_rotary(key, *rotary)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        out = out.transpose(1, 2).type_as(query)
        return self.to_out[0](out.reshape(batch_size, q_len, query_dim))


class Block(nn.Module):
    def __init__(self, hidden_size: int, heads: int, intermediate_size: int, eps: float) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.self_attn = Attention(hidden_size, heads)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.mlp = FeedForward(hidden_size, intermediate_size)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor, rotary: tuple[Tensor, Tensor]
    ) -> Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, rotary
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class AdaLayerNorm(nn.Module):
    """LayerNorm without affine parameters, modulated by ``(shift, scale)`` from the time embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)
        self.dim = dim

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        temb = self.linear(F.silu(temb))
        shift, scale = temb.chunk(2, dim=1)
        x = F.layer_norm(x, (self.dim,), None, None, _NORM_OUT_EPS)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


class OmniGenTransformer2DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in OMNIGEN_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported OmniGen transformer setting {unknown[0]}")
        cfg = OMNIGEN_DEFAULTS | {k: v for k, v in config.items() if not k.startswith("_")}
        if cfg["num_key_value_heads"] != cfg["num_attention_heads"]:
            raise ValueError("Unsupported OmniGen transformer setting num_key_value_heads")
        if cfg["hidden_size"] % cfg["num_attention_heads"] or (
            cfg["hidden_size"] // cfg["num_attention_heads"]
        ) % 2:
            raise ValueError("Unsupported OmniGen transformer setting num_attention_heads")
        if not isinstance(cfg["rope_scaling"], Mapping) or not {
            "short_factor",
            "long_factor",
        } <= set(cfg["rope_scaling"]):
            raise ValueError("Unsupported OmniGen transformer setting rope_scaling")
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        if any(len(cfg["rope_scaling"][k]) != head_dim // 2 for k in ("short_factor", "long_factor")):
            raise ValueError("Unsupported OmniGen transformer setting rope_scaling")
        for key, expected in (
            ("flip_sin_to_cos", True),
            ("downscale_freq_shift", 0),
            ("timestep_activation_fn", "silu"),
        ):
            if cfg[key] != expected:
                raise ValueError(f"Unsupported OmniGen transformer setting {key}={cfg[key]!r}")
        super().__init__(cfg)
        hidden = cfg["hidden_size"]
        self.in_channels = cfg["in_channels"]
        self.out_channels = cfg["in_channels"]
        self.patch_embedding = PatchEmbed(
            cfg["patch_size"], cfg["in_channels"], hidden, cfg["pos_embed_max_size"]
        )
        self.time_token = TimestepEmbedding(cfg["time_step_dim"], hidden)
        self.t_embedder = TimestepEmbedding(cfg["time_step_dim"], hidden)
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], hidden, cfg["pad_token_id"])
        self.rope = SuScaledRotaryEmbedding(
            head_dim,
            cfg["max_position_embeddings"],
            cfg["original_max_position_embeddings"],
            cfg["rope_base"],
            cfg["rope_scaling"],
        )
        self.layers = nn.ModuleList(
            Block(hidden, cfg["num_attention_heads"], cfg["intermediate_size"], cfg["rms_norm_eps"])
            for _ in range(cfg["num_layers"])
        )
        self.norm = RMSNorm(hidden, cfg["rms_norm_eps"])
        self.norm_out = AdaLayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, cfg["patch_size"] ** 2 * self.out_channels)

    def _condition_tokens(
        self,
        input_ids: Tensor,
        input_img_latents: Sequence[Tensor],
        input_image_sizes: Mapping[int, Sequence[Sequence[int]]],
        dtype: torch.dtype,
    ) -> Tensor:
        latents = [x.to(dtype) for x in input_img_latents]
        condition_tokens = self.embed_tokens(input_ids)
        image_tokens = self.patch_embedding(latents, is_input_image=True)
        index = 0
        for batch_index in input_image_sizes:
            for start, end in input_image_sizes[batch_index]:
                condition_tokens[batch_index, start:end] = image_tokens[index].to(
                    condition_tokens.dtype
                )
                index += 1
        return condition_tokens

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        input_ids: Tensor,
        input_img_latents: Sequence[Tensor],
        input_image_sizes: Mapping[int, Sequence[Sequence[int]]],
        attention_mask: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        batch_size, _, height, width = hidden_states.shape
        p = self.config["patch_size"]
        post_patch_height, post_patch_width = height // p, width // p

        hidden_states = self.patch_embedding(hidden_states, is_input_image=False)
        num_tokens_for_output_image = hidden_states.size(1)

        timestep_proj = _timestep_embedding(timestep, self.config["time_step_dim"]).type_as(
            hidden_states
        )
        time_token = self.time_token(timestep_proj).unsqueeze(1)
        temb = self.t_embedder(timestep_proj)

        condition_tokens = self._condition_tokens(
            input_ids, input_img_latents, input_image_sizes, hidden_states.dtype
        )
        hidden_states = torch.cat([condition_tokens, time_token, hidden_states], dim=1)

        seq_length = hidden_states.size(1)
        position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask.dim() == 3:
            min_dtype = torch.finfo(hidden_states.dtype).min
            attention_mask = ((1 - attention_mask) * min_dtype).unsqueeze(1).type_as(hidden_states)

        rotary = self.rope(hidden_states, position_ids)
        for block in self.layers:
            hidden_states = block(hidden_states, attention_mask, rotary)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states[:, -num_tokens_for_output_image:]
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, post_patch_height, post_patch_width, p, p, -1)
        return hidden_states.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
