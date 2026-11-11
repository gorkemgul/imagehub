"""Native HunyuanImage 2.1 transformer for text-to-image inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.flux_transformer import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
    CombinedTimestepTextProjEmbeddings,
    FeedForward,
    TimestepEmbedding,
    _apply_rotary_emb,
    _get_timestep_embedding,
)

HUNYUANIMAGE_TRANSFORMER_DEFAULTS: dict[str, Any] = {
    "in_channels": 64,
    "out_channels": 64,
    "num_attention_heads": 28,
    "attention_head_dim": 128,
    "num_layers": 20,
    "num_single_layers": 40,
    "num_refiner_layers": 2,
    "mlp_ratio": 4.0,
    "patch_size": (1, 1),
    "qk_norm": "rms_norm",
    "guidance_embeds": False,
    "text_embed_dim": 3584,
    "text_embed_2_dim": None,
    "rope_theta": 256.0,
    "rope_axes_dim": (64, 64),
    "use_meanflow": False,
}


class HunyuanImagePatchEmbed(nn.Module):
    def __init__(self, patch_size: tuple[int, int], in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.proj(hidden_states).flatten(2).transpose(1, 2)


class HunyuanImageByT5TextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, out_features)
        self.act_fn = nn.GELU()

    def forward(self, encoder_hidden_states: Tensor) -> Tensor:
        hidden_states = self.norm(encoder_hidden_states)
        hidden_states = self.act_fn(self.linear_1(hidden_states))
        hidden_states = self.act_fn(self.linear_2(hidden_states))
        return self.linear_3(hidden_states)


class HunyuanImageTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)

    def forward(self, timestep: Tensor) -> Tensor:
        projection = _get_timestep_embedding(timestep, 256)
        return self.timestep_embedder(projection.to(dtype=timestep.dtype))


class HunyuanImageAdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(self, temb: Tensor) -> tuple[Tensor, Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        return gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)


class HunyuanImageRefinerAttention(nn.Module):
    """Plain multi-head self-attention with a per-query boolean mask (no QK norm)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
        batch_size = hidden_states.shape[0]
        head_dim = hidden_states.shape[-1] // self.heads
        if attention_mask is not None:
            # As the reference processor: one copy of the mask per head.
            attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
            attention_mask = attention_mask.view(
                batch_size, self.heads, -1, attention_mask.shape[-1]
            )
        query = self.to_q(hidden_states).view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = self.to_k(hidden_states).view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = self.to_v(hidden_states).view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        output = output.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        output = output.to(query.dtype)
        return self.to_out[1](self.to_out[0](output))


class HunyuanImageIndividualTokenRefinerBlock(nn.Module):
    def __init__(self, heads: int, head_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden_size = heads * head_dim
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = HunyuanImageRefinerAttention(hidden_size, heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.ff = _LinearSiluFeedForward(hidden_size, mlp_ratio)
        self.norm_out = HunyuanImageAdaNorm(hidden_size, 2 * hidden_size)

    def forward(self, hidden_states: Tensor, temb: Tensor, attention_mask: Tensor | None) -> Tensor:
        attn_output = self.attn(self.norm1(hidden_states), attention_mask)
        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa
        return hidden_states + self.ff(self.norm2(hidden_states)) * gate_mlp


class _LinearSilu(nn.Module):
    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.silu(self.proj(hidden_states))


class _LinearSiluFeedForward(nn.Module):
    def __init__(self, dim: int, mult: float) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.ModuleList(
            [_LinearSilu(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim)]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class HunyuanImageIndividualTokenRefiner(nn.Module):
    def __init__(self, heads: int, head_dim: int, num_layers: int, mlp_ratio: float) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            HunyuanImageIndividualTokenRefinerBlock(heads, head_dim, mlp_ratio)
            for _ in range(num_layers)
        )

    def forward(self, hidden_states: Tensor, temb: Tensor, attention_mask: Tensor | None) -> Tensor:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size, seq_len = attention_mask.shape
            mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            mask_2 = mask_1.transpose(2, 3)
            self_attn_mask = (mask_1 & mask_2).bool()
            self_attn_mask[:, :, :, 0] = True
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)
        return hidden_states


class HunyuanImageTokenRefiner(nn.Module):
    def __init__(
        self, in_channels: int, heads: int, head_dim: int, num_layers: int, mlp_ratio: float = 4.0
    ) -> None:
        super().__init__()
        hidden_size = heads * head_dim
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(hidden_size, in_channels)
        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = HunyuanImageIndividualTokenRefiner(
            heads, head_dim, num_layers, mlp_ratio
        )

    def forward(self, hidden_states: Tensor, timestep: Tensor, attention_mask: Tensor) -> Tensor:
        original_dtype = hidden_states.dtype
        mask_float = attention_mask.float().unsqueeze(-1)
        pooled = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        pooled = pooled.to(original_dtype)
        temb = self.time_text_embed(timestep, pooled)
        hidden_states = self.proj_in(hidden_states)
        return self.token_refiner(hidden_states, temb, attention_mask)


class HunyuanImageRotaryPosEmbed(nn.Module):
    def __init__(
        self, patch_size: tuple[int, int], rope_dim: tuple[int, ...], theta: float
    ) -> None:
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.rope_dim = tuple(rope_dim)
        self.theta = theta

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        _, _, height, width = hidden_states.shape
        sizes = [height // self.patch_size[0], width // self.patch_size[1]]
        grids = [
            torch.arange(0, size, device=hidden_states.device, dtype=torch.float32)
            for size in sizes
        ]
        grid = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=0)
        cosines, sines = [], []
        for index, dim in enumerate(self.rope_dim):
            frequencies = 1.0 / (
                self.theta
                ** (torch.arange(0, dim, 2, dtype=torch.float32, device=hidden_states.device) / dim)
            )
            frequencies = torch.outer(grid[index].reshape(-1), frequencies)
            cosines.append(frequencies.cos().repeat_interleave(2, dim=1).float())
            sines.append(frequencies.sin().repeat_interleave(2, dim=1).float())
        return torch.cat(cosines, dim=1), torch.cat(sines, dim=1)


def _joint_attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:
    query, key, value = (item.permute(0, 2, 1, 3) for item in (query, key, value))
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    return output.permute(0, 2, 1, 3)


class HunyuanImageAttention(nn.Module):
    """Joint attention; image tokens come first and carry the rotary embedding."""

    def __init__(self, dim: int, heads: int, head_dim: int, *, dual_stream: bool) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.dual_stream = dual_stream
        inner_dim = heads * head_dim
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        if dual_stream:
            self.norm_added_q = nn.RMSNorm(head_dim, eps=1e-6)
            self.norm_added_k = nn.RMSNorm(head_dim, eps=1e-6)
            self.add_q_proj = nn.Linear(dim, inner_dim)
            self.add_k_proj = nn.Linear(dim, inner_dim)
            self.add_v_proj = nn.Linear(dim, inner_dim)
            self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim), nn.Dropout(0.0)])
            self.to_add_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        text_length = encoder_hidden_states.shape[1]
        if not self.dual_stream:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        query = self.norm_q(self.to_q(hidden_states).unflatten(2, (-1, self.head_dim)))
        key = self.norm_k(self.to_k(hidden_states).unflatten(2, (-1, self.head_dim)))
        value = self.to_v(hidden_states).unflatten(2, (-1, self.head_dim))
        if self.dual_stream:
            query = _apply_rotary_emb(query, rotary_emb)
            key = _apply_rotary_emb(key, rotary_emb)
            encoder_query = self.norm_added_q(
                self.add_q_proj(encoder_hidden_states).unflatten(2, (-1, self.head_dim))
            )
            encoder_key = self.norm_added_k(
                self.add_k_proj(encoder_hidden_states).unflatten(2, (-1, self.head_dim))
            )
            encoder_value = self.add_v_proj(encoder_hidden_states).unflatten(2, (-1, self.head_dim))
            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)
        else:
            query = torch.cat(
                [_apply_rotary_emb(query[:, :-text_length], rotary_emb), query[:, -text_length:]],
                dim=1,
            )
            key = torch.cat(
                [_apply_rotary_emb(key[:, :-text_length], rotary_emb), key[:, -text_length:]],
                dim=1,
            )
        output = _joint_attention(query, key, value, attention_mask).flatten(2, 3)
        output = output.to(query.dtype)
        output, encoder_output = output[:, :-text_length], output[:, -text_length:]
        if self.dual_stream:
            output = self.to_out[1](self.to_out[0](output))
            encoder_output = self.to_add_out(encoder_output)
        return output, encoder_output


class HunyuanImageTransformerBlock(nn.Module):
    def __init__(self, heads: int, head_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        dim = heads * head_dim
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)
        self.attn = HunyuanImageAttention(dim, heads, head_dim, dual_stream=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        attention_mask: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        (
            normalized_context,
            context_gate_msa,
            context_shift_mlp,
            context_scale_mlp,
            context_gate_mlp,
        ) = self.norm1_context(encoder_hidden_states, temb)
        attn_output, context_attn_output = self.attn(
            normalized, normalized_context, attention_mask, rotary_emb
        )
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = (
            encoder_hidden_states + context_attn_output * context_gate_msa.unsqueeze(1)
        )
        normalized = self.norm2(hidden_states)
        normalized_context = self.norm2_context(encoder_hidden_states)
        normalized = normalized * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        normalized_context = (
            normalized_context * (1 + context_scale_mlp[:, None]) + context_shift_mlp[:, None]
        )
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(normalized)
        encoder_hidden_states = encoder_hidden_states + context_gate_mlp.unsqueeze(
            1
        ) * self.ff_context(normalized_context)
        return hidden_states, encoder_hidden_states


class HunyuanImageSingleTransformerBlock(nn.Module):
    def __init__(self, heads: int, head_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        dim = heads * head_dim
        mlp_dim = int(dim * mlp_ratio)
        self.attn = HunyuanImageAttention(dim, heads, head_dim, dual_stream=False)
        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, mlp_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + mlp_dim, dim)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        attention_mask: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        text_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        residual = hidden_states
        normalized, gate = self.norm(hidden_states, temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(normalized))
        attn_output, context_attn_output = self.attn(
            normalized[:, :-text_length],
            normalized[:, -text_length:],
            attention_mask,
            rotary_emb,
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states) + residual
        return hidden_states[:, :-text_length], hidden_states[:, -text_length:]


class HunyuanImageTransformer2DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = HUNYUANIMAGE_TRANSFORMER_DEFAULTS | {
            name: value for name, value in config.items() if not name.startswith("_")
        }
        unsupported = {
            "guidance embeddings": cfg["guidance_embeds"],
            "mean-flow timesteps": cfg["use_meanflow"],
            "QK norm": cfg["qk_norm"] != "rms_norm",
            "MLP ratio": cfg["mlp_ratio"] != 4.0,
            "patch size": tuple(cfg["patch_size"]) != (1, 1),
            "rotary axes": len(cfg["rope_axes_dim"]) != 2,
        }
        for option, is_unsupported in unsupported.items():
            if is_unsupported:
                raise ValueError(f"Unsupported HunyuanImage transformer {option}")
        super().__init__(cfg)
        heads, head_dim = cfg["num_attention_heads"], cfg["attention_head_dim"]
        inner_dim = heads * head_dim
        self.out_channels = cfg["out_channels"] or cfg["in_channels"]
        self.x_embedder = HunyuanImagePatchEmbed(
            tuple(cfg["patch_size"]), cfg["in_channels"], inner_dim
        )
        self.context_embedder = HunyuanImageTokenRefiner(
            cfg["text_embed_dim"], heads, head_dim, cfg["num_refiner_layers"]
        )
        if cfg["text_embed_2_dim"] is not None:
            self.context_embedder_2 = HunyuanImageByT5TextProjection(
                cfg["text_embed_2_dim"], 2048, inner_dim
            )
        self.time_guidance_embed = HunyuanImageTimeEmbedding(inner_dim)
        self.rope = HunyuanImageRotaryPosEmbed(
            tuple(cfg["patch_size"]), tuple(cfg["rope_axes_dim"]), cfg["rope_theta"]
        )
        self.transformer_blocks = nn.ModuleList(
            HunyuanImageTransformerBlock(heads, head_dim, cfg["mlp_ratio"])
            for _ in range(cfg["num_layers"])
        )
        self.single_transformer_blocks = nn.ModuleList(
            HunyuanImageSingleTransformerBlock(heads, head_dim, cfg["mlp_ratio"])
            for _ in range(cfg["num_single_layers"])
        )
        self.norm_out = AdaLayerNormContinuous(inner_dim)
        self.proj_out = nn.Linear(inner_dim, math.prod(cfg["patch_size"]) * self.out_channels)

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
        encoder_hidden_states_2: Tensor | None = None,
        encoder_attention_mask_2: Tensor | None = None,
    ) -> Tensor:
        """Predict the flow velocity for latents of shape ``(batch, channels, height, width)``.

        ``timestep`` is on the scheduler's 0-1000 scale. Padding of both text streams is
        moved behind the valid tokens of both, as the reference does.
        """
        batch_size, _, height, width = hidden_states.shape
        patch_height, patch_width = self.config["patch_size"]
        rotary_emb = self.rope(hidden_states)
        encoder_attention_mask = encoder_attention_mask.bool()
        temb = self.time_guidance_embed(timestep)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states, timestep, encoder_attention_mask
        )
        if hasattr(self, "context_embedder_2") and encoder_hidden_states_2 is not None:
            assert encoder_attention_mask_2 is not None
            encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)
            encoder_attention_mask_2 = encoder_attention_mask_2.bool()
            states, masks = [], []
            for text, text_mask, text_2, text_mask_2 in zip(
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_hidden_states_2,
                encoder_attention_mask_2,
                strict=True,
            ):
                states.append(
                    torch.cat(
                        [
                            text_2[text_mask_2],
                            text[text_mask],
                            text_2[~text_mask_2],
                            text[~text_mask],
                        ],
                        dim=0,
                    )
                )
                masks.append(
                    torch.cat(
                        [
                            text_mask_2[text_mask_2],
                            text_mask[text_mask],
                            text_mask_2[~text_mask_2],
                            text_mask[~text_mask],
                        ],
                        dim=0,
                    )
                )
            encoder_hidden_states = torch.stack(states)
            encoder_attention_mask = torch.stack(masks)

        attention_mask = F.pad(encoder_attention_mask, (hidden_states.shape[1], 0), value=True)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask, rotary_emb
            )
        for single_block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = single_block(
                hidden_states, encoder_hidden_states, temb, attention_mask, rotary_emb
            )
        hidden_states = self.proj_out(self.norm_out(hidden_states, temb))
        grid_height, grid_width = height // patch_height, width // patch_width
        hidden_states = hidden_states.reshape(
            batch_size, grid_height, grid_width, self.out_channels, patch_height, patch_width
        )
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
        return hidden_states.reshape(batch_size, self.out_channels, height, width)
