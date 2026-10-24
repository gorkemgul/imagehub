"""Native BAGEL Mixture-of-Transformers and autoencoder modules."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.nextstep import NextStepAutoencoder


class BagelRMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype
        value = value.float()
        return self.weight * (
            value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        ).to(dtype)


class BagelMLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden, intermediate = config["hidden_size"], config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


def _rotate_half(value: Tensor) -> Tensor:
    left, right = value.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class BagelAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = hidden // self.heads
        self.groups = self.heads // self.kv_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=True)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.q_norm = BagelRMSNorm(self.head_dim, config["rms_norm_eps"])
        self.k_norm = BagelRMSNorm(self.head_dim, config["rms_norm_eps"])
        self.q_proj_moe_gen = nn.Linear(hidden, hidden, bias=True)
        self.k_proj_moe_gen = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(hidden, hidden, bias=False)
        self.q_norm_moe_gen = BagelRMSNorm(self.head_dim, config["rms_norm_eps"])
        self.k_norm_moe_gen = BagelRMSNorm(self.head_dim, config["rms_norm_eps"])

    def _route(self, value: Tensor, generated: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q = self.q_proj(value)
        k = self.k_proj(value)
        v = self.v_proj(value)
        if generated.any():
            q[generated] = self.q_proj_moe_gen(value[generated])
            k[generated] = self.k_proj_moe_gen(value[generated])
            v[generated] = self.v_proj_moe_gen(value[generated])
        q = q.view(-1, self.heads, self.head_dim)
        k = k.view(-1, self.kv_heads, self.head_dim)
        if generated.any():
            q[generated] = self.q_norm_moe_gen(q[generated])
            k[generated] = self.k_norm_moe_gen(k[generated])
        q[~generated] = self.q_norm(q[~generated])
        k[~generated] = self.k_norm(k[~generated])
        return q, k, v.view(-1, self.kv_heads, self.head_dim)

    def forward(
        self,
        value: Tensor,
        cos: Tensor,
        sin: Tensor,
        generated: Tensor,
        past: tuple[Tensor, Tensor] | None,
        *,
        causal: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        q, k, v = self._route(value, generated)
        q = q * cos[:, None] + _rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + _rotate_half(k) * sin[:, None]
        current = (k, v)
        if past is not None:
            k, v = torch.cat((past[0], k)), torch.cat((past[1], v))
        k = k[:, :, None].expand(-1, -1, self.groups, -1).reshape(-1, self.heads, self.head_dim)
        v = v[:, :, None].expand(-1, -1, self.groups, -1).reshape(-1, self.heads, self.head_dim)
        query = q.transpose(0, 1).unsqueeze(0)
        keys = k.transpose(0, 1).unsqueeze(0)
        values = v.transpose(0, 1).unsqueeze(0)
        if causal and past is not None:
            q_length, k_length = value.shape[0], k.shape[0]
            offset = k_length - q_length
            mask = torch.arange(k_length, device=value.device)[None, :] <= (
                torch.arange(q_length, device=value.device)[:, None] + offset
            )
            output = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
        elif not causal:
            mask = torch.zeros(value.shape[0], k.shape[0], device=value.device, dtype=query.dtype)
            output = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask[None])
        else:
            output = F.scaled_dot_product_attention(query, keys, values, is_causal=causal)
        output = output.squeeze(0).transpose(0, 1).reshape(value.shape[0], -1)
        routed = self.o_proj(output)
        if generated.any():
            routed[generated] = self.o_proj_moe_gen(output[generated])
        return routed, current


class BagelDecoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden, eps = config["hidden_size"], config["rms_norm_eps"]
        self.self_attn = BagelAttention(config)
        self.mlp = BagelMLP(config)
        self.mlp_moe_gen = BagelMLP(config)
        self.input_layernorm = BagelRMSNorm(hidden, eps)
        self.input_layernorm_moe_gen = BagelRMSNorm(hidden, eps)
        self.post_attention_layernorm = BagelRMSNorm(hidden, eps)
        self.post_attention_layernorm_moe_gen = BagelRMSNorm(hidden, eps)

    @staticmethod
    def _route_norm(value: Tensor, generated: Tensor, und: nn.Module, gen: nn.Module) -> Tensor:
        result = und(value)
        if generated.any():
            result[generated] = gen(value[generated])
        return result

    def forward(
        self,
        value: Tensor,
        cos: Tensor,
        sin: Tensor,
        generated: Tensor,
        past: tuple[Tensor, Tensor] | None,
        *,
        causal: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        normalized = self._route_norm(
            value, generated, self.input_layernorm, self.input_layernorm_moe_gen
        )
        attention, present = self.self_attn(normalized, cos, sin, generated, past, causal=causal)
        value = value + attention
        normalized = self._route_norm(
            value, generated, self.post_attention_layernorm, self.post_attention_layernorm_moe_gen
        )
        output = self.mlp(normalized)
        if generated.any():
            output[generated] = self.mlp_moe_gen(normalized[generated])
        return value + output, present


class BagelTransformer(nn.Module):
    inv_freq: Tensor

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            BagelDecoderLayer(config) for _ in range(config["num_hidden_layers"])
        )
        self.norm = BagelRMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.norm_moe_gen = BagelRMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.register_buffer("inv_freq", self._make_inv_freq(), persistent=False)

    def _make_inv_freq(self) -> Tensor:
        dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        return 1 / (self.config["rope_theta"] ** (torch.arange(0, dim, 2).float() / dim))

    def forward(
        self,
        value: Tensor,
        positions: Tensor,
        generated: Tensor,
        cache: list[tuple[Tensor, Tensor]] | None = None,
        *,
        causal: bool,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        frequencies = torch.outer(positions.float(), self.inv_freq.float())
        embedding = torch.cat((frequencies, frequencies), -1)
        cos, sin = embedding.cos().to(value.dtype), embedding.sin().to(value.dtype)
        presents = []
        for index, layer in enumerate(self.layers):
            past = None if cache is None else cache[index]
            value, present = layer(value, cos, sin, generated, past, causal=causal)
            presents.append(
                present
                if past is None
                else (torch.cat((past[0], present[0])), torch.cat((past[1], present[1])))
            )
        normalized = self.norm(value)
        if generated.any():
            normalized[generated] = self.norm_moe_gen(value[generated])
        return normalized, presents


class BagelLanguageModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.model = BagelTransformer(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)


class BagelVisionAttention(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)


class BagelVisionMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden)


class BagelVisionLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = config["hidden_size"]
        self.self_attn = BagelVisionAttention(hidden)
        self.layer_norm1 = nn.LayerNorm(hidden, eps=config.get("layer_norm_eps", 1e-6))
        self.mlp = BagelVisionMLP(hidden, config["intermediate_size"])
        self.layer_norm2 = nn.LayerNorm(hidden, eps=config.get("layer_norm_eps", 1e-6))


class BagelVisionEmbeddings(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        patch = config["patch_size"]
        self.patch_embedding = nn.Linear(
            config.get("num_channels", 3) * patch**2, config["hidden_size"]
        )
        side = config.get("image_size", 980) // patch
        self.position_embedding = nn.Embedding(side**2, config["hidden_size"])


class BagelVisionEncoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            BagelVisionLayer(config) for _ in range(config["num_hidden_layers"] - 1)
        )


class BagelVisionTransformer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.embeddings = BagelVisionEmbeddings(config)
        self.encoder = BagelVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config["hidden_size"], eps=config.get("layer_norm_eps", 1e-6)
        )


class BagelVisionModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.vision_model = BagelVisionTransformer(config)


class BagelConnector(nn.Module):
    def __init__(self, input_size: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, value: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(value), approximate="tanh"))


class BagelTimestepEmbedder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, timestep: Tensor) -> Tensor:
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(128, device=timestep.device).float() / 128
        )
        args = timestep[:, None].float() * frequencies[None]
        value = torch.cat((args.cos(), args.sin()), -1)
        return self.mlp(value.to(next(self.mlp.parameters()).dtype))


class BagelPositionEmbedding(nn.Module):
    def __init__(self, side: int, hidden: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(side**2, hidden), requires_grad=False)

    def forward(self, positions: Tensor) -> Tensor:
        return self.pos_embed[positions]


class BagelModel(NativeModel):
    """BAGEL with tensor names identical to the author checkpoint."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        llm, vit = config["llm_config"], config["vit_config"]
        hidden = llm["hidden_size"]
        patch, channels = config.get("latent_patch_size", 2), config["vae_config"]["z_channels"]
        self.language_model = BagelLanguageModel(llm)
        self.vit_model = BagelVisionModel(vit)
        self.connector = BagelConnector(vit["hidden_size"], hidden)
        self.time_embedder = BagelTimestepEmbedder(hidden)
        self.vae2llm = nn.Linear(patch * patch * channels, hidden)
        self.llm2vae = nn.Linear(hidden, patch * patch * channels)
        self.latent_pos_embed = BagelPositionEmbedding(config.get("max_latent_size", 64), hidden)
        self.vit_pos_embed = BagelPositionEmbedding(
            config.get("vit_max_num_patch_per_side", 70), hidden
        )

    def init_non_persistent_buffers(self) -> None:
        self.language_model.model.inv_freq = self.language_model.model._make_inv_freq()


class BagelAutoencoder(NextStepAutoencoder):
    """BAGEL's frozen FLUX autoencoder with its published legacy keys."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        source = {
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "z_channels": 16,
            "scale_factor": 0.3611,
            "shift_factor": 0.1159,
        }
        source.update(config or {})
        super().__init__(source)
