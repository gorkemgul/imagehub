"""Native Chroma transformer for text-to-image inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

CHROMA_TRANSFORMER_DEFAULTS: dict[str, Any] = {
    "patch_size": 1,
    "in_channels": 64,
    "out_channels": None,
    "num_layers": 19,
    "num_single_layers": 38,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "guidance_embeds": False,
    "axes_dims_rope": (16, 56, 56),
    "approximator_num_channels": 64,
    "approximator_hidden_dim": 5120,
    "approximator_layers": 5,
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


def _attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None = None,
) -> Tensor:
    query, key, value = (item.permute(0, 2, 1, 3) for item in (query, key, value))
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    return hidden_states.permute(0, 2, 1, 3)


class ChromaPosEmbed(nn.Module):
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
            frequencies = torch.outer(positions[:, index], frequencies)
            cosines.append(frequencies.cos().repeat_interleave(2, dim=1).float())
            sines.append(frequencies.sin().repeat_interleave(2, dim=1).float())
        return torch.cat(cosines, dim=-1), torch.cat(sines, dim=-1)


class ChromaTimestepEmbedding(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps: Tensor) -> Tensor:
        return _get_timestep_embedding(timesteps, self.num_channels)


class ChromaCombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, num_channels: int, out_dim: int) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_dim = out_dim
        self.time_proj = ChromaTimestepEmbedding(num_channels)
        self.guidance_proj = ChromaTimestepEmbedding(num_channels)
        self.register_buffer(
            "mod_proj",
            _get_timestep_embedding(
                torch.arange(out_dim, dtype=torch.float32) * 1000,
                2 * num_channels,
            ),
            persistent=False,
        )

    def forward(self, timestep: Tensor) -> Tensor:
        batch_size = timestep.shape[0]
        timesteps_proj = self.time_proj(timestep).to(dtype=timestep.dtype)
        guidance_proj = self.guidance_proj(
            torch.zeros(batch_size, dtype=timestep.dtype, device=timestep.device)
        )
        mod_proj = cast(
            Tensor,
            self.mod_proj.to(dtype=timesteps_proj.dtype, device=timesteps_proj.device),
        ).repeat(batch_size, 1, 1)
        timestep_guidance = (
            torch.cat([timesteps_proj, guidance_proj], dim=1)
            .unsqueeze(1)
            .repeat(1, self.out_dim, 1)
        )
        input_vec = torch.cat([timestep_guidance, mod_proj], dim=-1)
        return input_vec.to(timestep.dtype)


class ChromaApproximatorLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.linear_2(hidden_states)


class ChromaApproximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(ChromaApproximatorLayer(hidden_dim) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.RMSNorm(hidden_dim) for _ in range(n_layers))
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        for layer, norm in zip(self.layers, self.norms, strict=True):
            x = x + layer(norm(x))
        return self.out_proj(x)


class ChromaAdaLayerNormZero(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, hidden_states: Tensor, embedding: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = embedding.flatten(1, 2).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return hidden_states, gate, shift_mlp, scale_mlp, gate_mlp


class ChromaAdaLayerNormZeroSingle(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: Tensor, embedding: Tensor) -> tuple[Tensor, Tensor]:
        shift, scale, gate = embedding.flatten(1, 2).chunk(3, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return hidden_states, gate


class ChromaAdaLayerNormContinuous(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)

    def forward(self, hidden_states: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = torch.chunk(conditioning.flatten(1, 2), 2, dim=1)
        return self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]


class ChromaGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = "tanh"

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.gelu(self.proj(hidden_states), approximate=self.approximate)


class ChromaFeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            [
                ChromaGELU(dim, dim * 4),
                nn.Dropout(0.0),
                nn.Linear(dim * 4, dim),
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class ChromaFluxAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, *, added_projection: bool) -> None:
        super().__init__()
        self.head_dim = head_dim
        inner_dim = heads * head_dim
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        if added_projection:
            self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim), nn.Dropout(0.0)])
            self.norm_added_q = nn.RMSNorm(head_dim, eps=1e-6)
            self.norm_added_k = nn.RMSNorm(head_dim, eps=1e-6)
            self.add_q_proj = nn.Linear(dim, inner_dim)
            self.add_k_proj = nn.Linear(dim, inner_dim)
            self.add_v_proj = nn.Linear(dim, inner_dim)
            self.to_add_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        hidden_states: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        query = self.to_q(hidden_states).unflatten(-1, (-1, self.head_dim))
        key = self.to_k(hidden_states).unflatten(-1, (-1, self.head_dim))
        value = self.to_v(hidden_states).unflatten(-1, (-1, self.head_dim))
        query = self.norm_q(query)
        key = self.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_query = self.add_q_proj(encoder_hidden_states).unflatten(
                -1, (-1, self.head_dim)
            )
            encoder_key = self.add_k_proj(encoder_hidden_states).unflatten(-1, (-1, self.head_dim))
            encoder_value = self.add_v_proj(encoder_hidden_states).unflatten(
                -1, (-1, self.head_dim)
            )
            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        query = _apply_rotary_emb(query, rotary_emb)
        key = _apply_rotary_emb(key, rotary_emb)
        output = _attention(query, key, value, attention_mask).flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is None:
            return output
        encoder_output, output = output.split_with_sizes(
            [encoder_hidden_states.shape[1], output.shape[1] - encoder_hidden_states.shape[1]],
            dim=1,
        )
        output = self.to_out[1](self.to_out[0](output.contiguous()))
        encoder_output = self.to_add_out(encoder_output.contiguous())
        return output, encoder_output


class ChromaTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.norm1 = ChromaAdaLayerNormZero(dim)
        self.norm1_context = ChromaAdaLayerNormZero(dim)
        self.attn = ChromaFluxAttention(dim, heads, head_dim, added_projection=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = ChromaFeedForward(dim)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = ChromaFeedForward(dim)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        temb_img, temb_txt = temb[:, :6], temb[:, 6:]
        normalized, gate, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb_img)
        (
            normalized_context,
            context_gate,
            context_shift_mlp,
            context_scale_mlp,
            context_gate_mlp,
        ) = self.norm1_context(encoder_hidden_states, temb_txt)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :] * attention_mask[:, None, :, None]

        attention_output, context_attention_output = self.attn(
            normalized, rotary_emb, normalized_context, attention_mask
        )

        hidden_states = hidden_states + gate.unsqueeze(1) * attention_output
        normalized = self.norm2(hidden_states)
        normalized = normalized * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(normalized)

        encoder_hidden_states = (
            encoder_hidden_states + context_gate.unsqueeze(1) * context_attention_output
        )
        normalized_context = self.norm2_context(encoder_hidden_states)
        normalized_context = (
            normalized_context * (1 + context_scale_mlp[:, None]) + context_shift_mlp[:, None]
        )
        encoder_hidden_states = encoder_hidden_states + context_gate_mlp.unsqueeze(
            1
        ) * self.ff_context(normalized_context)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


class ChromaSingleTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.norm = ChromaAdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, dim * 4)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim * 5, dim)
        self.attn = ChromaFluxAttention(dim, heads, head_dim, added_projection=False)

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        residual = hidden_states
        normalized, gate = self.norm(hidden_states, temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(normalized))

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :] * attention_mask[:, None, :, None]

        attention_output = self.attn(normalized, rotary_emb, attention_mask=attention_mask)
        hidden_states = torch.cat([attention_output, mlp_hidden_states], dim=2)
        hidden_states = residual + gate.unsqueeze(1) * self.proj_out(hidden_states)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


class ChromaTransformer2DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = CHROMA_TRANSFORMER_DEFAULTS | {
            name: value for name, value in config.items() if not name.startswith("_")
        }
        if sum(cfg["axes_dims_rope"]) != cfg["attention_head_dim"]:
            raise ValueError("Unsupported Chroma transformer configuration")
        super().__init__(cfg)
        self.out_channels = cfg["out_channels"] or cfg["in_channels"]
        self.inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        self.pos_embed = ChromaPosEmbed(10000, list(cfg["axes_dims_rope"]))
        num_channels = cfg["approximator_num_channels"] // 4
        out_dim = 3 * cfg["num_single_layers"] + 2 * 6 * cfg["num_layers"] + 2
        self.time_text_embed = ChromaCombinedTimestepTextProjEmbeddings(num_channels, out_dim)
        self.distilled_guidance_layer = ChromaApproximator(
            in_dim=cfg["approximator_num_channels"],
            out_dim=self.inner_dim,
            hidden_dim=cfg["approximator_hidden_dim"],
            n_layers=cfg["approximator_layers"],
        )
        self.context_embedder = nn.Linear(cfg["joint_attention_dim"], self.inner_dim)
        self.x_embedder = nn.Linear(cfg["in_channels"], self.inner_dim)
        self.transformer_blocks = nn.ModuleList(
            ChromaTransformerBlock(
                self.inner_dim,
                cfg["num_attention_heads"],
                cfg["attention_head_dim"],
            )
            for _ in range(cfg["num_layers"])
        )
        self.single_transformer_blocks = nn.ModuleList(
            ChromaSingleTransformerBlock(
                self.inner_dim,
                cfg["num_attention_heads"],
                cfg["attention_head_dim"],
            )
            for _ in range(cfg["num_single_layers"])
        )
        self.norm_out = ChromaAdaLayerNormContinuous(self.inner_dim)
        self.proj_out = nn.Linear(
            self.inner_dim,
            cfg["patch_size"] * cfg["patch_size"] * self.out_channels,
            bias=True,
        )

    def init_non_persistent_buffers(self) -> None:
        self.time_text_embed.mod_proj = _get_timestep_embedding(
            torch.arange(self.time_text_embed.out_dim, dtype=torch.float32) * 1000,
            2 * self.time_text_embed.num_channels,
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        if joint_attention_kwargs:
            raise ValueError("Unsupported Chroma transformer option")
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        input_vec = self.time_text_embed(timestep)
        pooled_temb = self.distilled_guidance_layer(input_vec)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        rotary_emb = self.pos_embed(torch.cat((txt_ids, img_ids), dim=0))

        num_single_layers = len(self.single_transformer_blocks)
        num_layers = len(self.transformer_blocks)
        img_offset = 3 * num_single_layers
        txt_offset = img_offset + 6 * num_layers
        for index_block, block in enumerate(self.transformer_blocks):
            img_modulation = img_offset + 6 * index_block
            text_modulation = txt_offset + 6 * index_block
            temb = torch.cat(
                (
                    pooled_temb[:, img_modulation : img_modulation + 6],
                    pooled_temb[:, text_modulation : text_modulation + 6],
                ),
                dim=1,
            )
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                rotary_emb=rotary_emb,
                attention_mask=attention_mask,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for index_block, block in enumerate(self.single_transformer_blocks):
            start_idx = 3 * index_block
            temb = pooled_temb[:, start_idx : start_idx + 3]
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                rotary_emb=rotary_emb,
                attention_mask=attention_mask,
            )

        text_length = encoder_hidden_states.shape[1]
        hidden_states = hidden_states[:, text_length:, ...]
        temb = pooled_temb[:, -2:]
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)
