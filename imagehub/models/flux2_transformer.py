"""Native FLUX.2 transformer for text-to-image inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

FLUX2_TRANSFORMER_DEFAULTS: dict[str, Any] = {
    "patch_size": 1,
    "in_channels": 128,
    "out_channels": None,
    "num_layers": 8,
    "num_single_layers": 48,
    "attention_head_dim": 128,
    "num_attention_heads": 48,
    "joint_attention_dim": 15360,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": (32, 32, 32, 32),
    "rope_theta": 2000,
    "eps": 1e-6,
    "guidance_embeds": True,
}


def _get_timestep_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / half_dim
    embedding = torch.exp(exponent)
    embedding = timesteps[:, None].float() * embedding[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
    embedding = torch.cat([embedding[:, half_dim:], embedding[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1, 0, 0))
    return embedding


def _apply_rotary_emb(hidden_states: Tensor, rotary_emb: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = rotary_emb
    cos = cos[None, :, None, :].to(hidden_states.device)
    sin = sin[None, :, None, :].to(hidden_states.device)
    real, imaginary = hidden_states.reshape(*hidden_states.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([-imaginary, real], dim=-1).flatten(3)
    return (hidden_states.float() * cos + rotated.float() * sin).to(hidden_states.dtype)


def _attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    query, key, value = (item.permute(0, 2, 1, 3) for item in (query, key, value))
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False
    )
    return hidden_states.permute(0, 2, 1, 3)


class Flux2SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        half = hidden_states.shape[-1] // 2
        return self.gate_fn(hidden_states[..., :half]) * hidden_states[..., half:]


class Flux2FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        inner_dim = int(dim * mlp_ratio)
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.act_fn = Flux2SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.linear_in(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return self.linear_out(hidden_states)


class Flux2Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, eps: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        inner_dim = heads * head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=False), nn.Dropout(0.0)])
        self.norm_added_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=eps)
        self.add_q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.add_k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.add_v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.to_add_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        query = self.to_q(hidden_states).unflatten(-1, (-1, self.head_dim))
        key = self.to_k(hidden_states).unflatten(-1, (-1, self.head_dim))
        value = self.to_v(hidden_states).unflatten(-1, (-1, self.head_dim))
        query = self.norm_q(query)
        key = self.norm_k(key)

        encoder_query = self.add_q_proj(encoder_hidden_states).unflatten(-1, (-1, self.head_dim))
        encoder_key = self.add_k_proj(encoder_hidden_states).unflatten(-1, (-1, self.head_dim))
        encoder_value = self.add_v_proj(encoder_hidden_states).unflatten(-1, (-1, self.head_dim))
        encoder_query = self.norm_added_q(encoder_query)
        encoder_key = self.norm_added_k(encoder_key)

        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)
        query = _apply_rotary_emb(query, rotary_emb)
        key = _apply_rotary_emb(key, rotary_emb)
        output = _attention(query, key, value).flatten(2, 3).to(query.dtype)
        encoder_output, output = output.split_with_sizes(
            [encoder_hidden_states.shape[1], output.shape[1] - encoder_hidden_states.shape[1]],
            dim=1,
        )
        encoder_output = self.to_add_out(encoder_output)
        output = self.to_out[1](self.to_out[0](output))
        return output, encoder_output


class Flux2ParallelSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: float, eps: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.to_qkv_mlp_proj = nn.Linear(
            dim, self.inner_dim * 3 + self.mlp_hidden_dim * 2, bias=False
        )
        self.mlp_act_fn = Flux2SwiGLU()
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_out = nn.Linear(self.inner_dim + self.mlp_hidden_dim, dim, bias=False)

    def forward(self, hidden_states: Tensor, rotary_emb: tuple[Tensor, Tensor]) -> Tensor:
        hidden_states = self.to_qkv_mlp_proj(hidden_states)
        qkv_dim = 3 * self.inner_dim
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [qkv_dim, hidden_states.shape[-1] - qkv_dim], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = self.norm_q(query.unflatten(-1, (-1, self.head_dim)))
        key = self.norm_k(key.unflatten(-1, (-1, self.head_dim)))
        value = value.unflatten(-1, (-1, self.head_dim))
        query = _apply_rotary_emb(query, rotary_emb)
        key = _apply_rotary_emb(key, rotary_emb)
        hidden_states = _attention(query, key, value).flatten(2, 3).to(query.dtype)
        mlp_hidden_states = self.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        return self.to_out(hidden_states)


class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim * 3 * mod_param_sets, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, temb: Tensor) -> Tensor:
        return self.linear(self.act_fn(temb))

    @staticmethod
    def split(modulation: Tensor, sets: int) -> tuple[tuple[Tensor, Tensor, Tensor], ...]:
        if modulation.ndim == 2:
            modulation = modulation.unsqueeze(1)
        params = torch.chunk(modulation, 3 * sets, dim=-1)
        return tuple(
            cast(tuple[Tensor, Tensor, Tensor], params[3 * index : 3 * (index + 1)])
            for index in range(sets)
        )


class Flux2TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: float, eps: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux2Attention(dim, heads, head_dim, eps)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = Flux2FeedForward(dim, mlp_ratio)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = Flux2FeedForward(dim, mlp_ratio)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        image_modulation: Tensor,
        text_modulation: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        (shift, scale, gate), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(
            image_modulation, 2
        )
        (
            (text_shift, text_scale, text_gate),
            (
                text_shift_mlp,
                text_scale_mlp,
                text_gate_mlp,
            ),
        ) = Flux2Modulation.split(text_modulation, 2)

        normalized = self.norm1(hidden_states)
        normalized = (1 + scale) * normalized + shift
        normalized_text = self.norm1_context(encoder_hidden_states)
        normalized_text = (1 + text_scale) * normalized_text + text_shift
        attention_output, text_attention_output = self.attn(normalized, normalized_text, rotary_emb)
        hidden_states = hidden_states + gate * attention_output
        normalized = self.norm2(hidden_states)
        normalized = normalized * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.ff(normalized)

        encoder_hidden_states = encoder_hidden_states + text_gate * text_attention_output
        normalized_text = self.norm2_context(encoder_hidden_states)
        normalized_text = normalized_text * (1 + text_scale_mlp) + text_shift_mlp
        encoder_hidden_states = encoder_hidden_states + text_gate_mlp * self.ff_context(
            normalized_text
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


class Flux2SingleTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: float, eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux2ParallelSelfAttention(dim, heads, head_dim, mlp_ratio, eps)

    def forward(
        self,
        hidden_states: Tensor,
        modulation: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
    ) -> Tensor:
        shift, scale, gate = Flux2Modulation.split(modulation, 1)[0]
        normalized = self.norm(hidden_states)
        normalized = (1 + scale) * normalized + shift
        hidden_states = hidden_states + gate * self.attn(normalized, rotary_emb)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


class Flux2PosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        cosines = []
        sines = []
        positions = ids.float()
        for index, dim in enumerate(self.axes_dim):
            frequencies = 1.0 / (
                self.theta
                ** (torch.arange(0, dim, 2, dtype=torch.float64, device=ids.device) / dim)
            )
            frequencies = torch.outer(positions[..., index], frequencies)
            cosines.append(frequencies.cos().repeat_interleave(2, dim=1).float())
            sines.append(frequencies.sin().repeat_interleave(2, dim=1).float())
        return torch.cat(cosines, dim=-1), torch.cat(sines, dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, embedding_dim, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, sample: Tensor) -> Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, guidance_embeds: bool) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.timestep_embedder = TimestepEmbedding(in_channels, embedding_dim)
        self.guidance_embedder = (
            TimestepEmbedding(in_channels, embedding_dim) if guidance_embeds else None
        )

    def forward(self, timestep: Tensor, guidance: Tensor | None) -> Tensor:
        timestep_projection = _get_timestep_embedding(timestep, self.in_channels)
        timestep_embedding = self.timestep_embedder(timestep_projection.to(timestep.dtype))
        if guidance is not None and self.guidance_embedder is not None:
            guidance_projection = _get_timestep_embedding(guidance, self.in_channels)
            guidance_embedding = self.guidance_embedder(guidance_projection.to(guidance.dtype))
            return timestep_embedding + guidance_embedding
        return timestep_embedding


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2, bias=False)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False, bias=False)

    def forward(self, hidden_states: Tensor, conditioning: Tensor) -> Tensor:
        embedding = self.linear(self.silu(conditioning).to(hidden_states.dtype))
        scale, shift = torch.chunk(embedding, 2, dim=1)
        return self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]


class Flux2Transformer2DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = FLUX2_TRANSFORMER_DEFAULTS | {
            name: value for name, value in config.items() if not name.startswith("_")
        }
        if sum(cfg["axes_dims_rope"]) != cfg["attention_head_dim"]:
            raise ValueError("Unsupported FLUX.2 transformer configuration")
        super().__init__(cfg)
        self.out_channels = cfg["out_channels"] or cfg["in_channels"]
        self.inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        self.pos_embed = Flux2PosEmbed(cfg["rope_theta"], list(cfg["axes_dims_rope"]))
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            cfg["timestep_guidance_channels"], self.inner_dim, cfg["guidance_embeds"]
        )
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, 2)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, 2)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, 1)
        self.x_embedder = nn.Linear(cfg["in_channels"], self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(cfg["joint_attention_dim"], self.inner_dim, bias=False)
        self.transformer_blocks = nn.ModuleList(
            Flux2TransformerBlock(
                self.inner_dim,
                cfg["num_attention_heads"],
                cfg["attention_head_dim"],
                cfg["mlp_ratio"],
                cfg["eps"],
            )
            for _ in range(cfg["num_layers"])
        )
        self.single_transformer_blocks = nn.ModuleList(
            Flux2SingleTransformerBlock(
                self.inner_dim,
                cfg["num_attention_heads"],
                cfg["attention_head_dim"],
                cfg["mlp_ratio"],
                cfg["eps"],
            )
            for _ in range(cfg["num_single_layers"])
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, cfg["eps"])
        self.proj_out = nn.Linear(
            self.inner_dim,
            cfg["patch_size"] * cfg["patch_size"] * self.out_channels,
            bias=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor | None = None,
        *,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        if joint_attention_kwargs:
            raise ValueError("Unsupported FLUX.2 transformer option")
        text_length = encoder_hidden_states.shape[1]
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)
        image_modulation = self.double_stream_modulation_img(temb)
        text_modulation = self.double_stream_modulation_txt(temb)
        single_modulation = self.single_stream_modulation(temb)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        image_rotary = self.pos_embed(img_ids)
        text_rotary = self.pos_embed(txt_ids)
        rotary_emb = (
            torch.cat([text_rotary[0], image_rotary[0]], dim=0),
            torch.cat([text_rotary[1], image_rotary[1]], dim=0),
        )
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                image_modulation,
                text_modulation,
                rotary_emb,
            )
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(hidden_states, single_modulation, rotary_emb)
        hidden_states = hidden_states[:, text_length:, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)
