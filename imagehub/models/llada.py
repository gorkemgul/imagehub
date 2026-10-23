"""LLaDA bidirectional masked-token transformer used by Lumina-DiMOO."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

LLADA_DEFAULTS: dict[str, Any] = {
    "activation_type": "silu",
    "alibi": False,
    "attention_dropout": 0.0,
    "attention_layer_norm": False,
    "block_group_size": 1,
    "block_type": "llama",
    "d_model": 4096,
    "embedding_dropout": 0.0,
    "embedding_size": 134548,
    "include_bias": False,
    "include_qkv_bias": False,
    "input_emb_norm": False,
    "layer_norm_type": "rms",
    "max_sequence_length": 4096,
    "mlp_hidden_size": 12288,
    "n_heads": 32,
    "n_kv_heads": 32,
    "n_layers": 32,
    "residual_dropout": 0.0,
    "rms_norm_eps": 1e-5,
    "rope": True,
    "rope_full_precision": True,
    "rope_theta": 500000.0,
    "scale_logits": False,
    "vocab_size": 134548,
    "weight_tying": False,
}

# Published metadata that does not alter inference.
_IGNORED = {
    "activation_checkpointing_strategy",
    "alibi_bias_max",
    "architectures",
    "auto_map",
    "attention_layer_norm_with_affine",
    "bias_for_layer_norm",
    "eos_token_id",
    "flash_attention",
    "init_cutoff_factor",
    "init_device",
    "init_fn",
    "init_std",
    "layer_norm_with_affine",
    "mask_token_id",
    "mlp_ratio",
    "model_type",
    "multi_query_attention",
    "pad_token_id",
    "precision",
    "torch_dtype",
    "transformers_version",
    "use_cache",
}


class LLaDARMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * values.to(dtype)


def _rotate_half(values: Tensor) -> Tensor:
    first, second = values.reshape(*values.shape[:-1], 2, values.shape[-1] // 2).unbind(-2)
    return torch.cat((-second, first), dim=-1)


class LLaDARotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float, full_precision: bool) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.full_precision = full_precision
        self.register_buffer(
            "inv_freq",
            1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)),
            persistent=False,
        )

    def init_non_persistent_buffers(self) -> None:
        self.inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        query_dtype, key_dtype = query.dtype, key.dtype
        q = query.float() if self.full_precision else query
        k = key.float() if self.full_precision else key
        positions = torch.arange(k.shape[-2], device=k.device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq.to(k.device))
        angles = torch.cat((frequencies, frequencies), dim=-1)[None, None]
        sin, cos = angles.sin().to(k.dtype), angles.cos().to(k.dtype)
        query_start = k.shape[-2] - q.shape[-2]
        q = q * cos[:, :, query_start:] + _rotate_half(q) * sin[:, :, query_start:]
        k = k * cos + _rotate_half(k) * sin
        return q.to(query_dtype), k.to(key_dtype)


class LLaDABlock(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width = int(config["d_model"])
        hidden = int(config["mlp_hidden_size"])
        bias = bool(config["include_bias"])
        qkv_bias = bias or bool(config["include_qkv_bias"])
        self.n_heads = int(config["n_heads"])
        self.n_kv_heads = int(config["n_kv_heads"])
        self.head_dim = width // self.n_heads
        eps = float(config["rms_norm_eps"])
        self.attn_norm = LLaDARMSNorm(width, eps)
        self.ff_norm = LLaDARMSNorm(width, eps)
        self.q_proj = nn.Linear(width, width, bias=qkv_bias)
        self.k_proj = nn.Linear(width, self.n_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(width, self.n_kv_heads * self.head_dim, bias=qkv_bias)
        self.attn_out = nn.Linear(width, width, bias=bias)
        self.ff_proj = nn.Linear(width, hidden, bias=bias)
        self.up_proj = nn.Linear(width, hidden, bias=bias)
        self.ff_out = nn.Linear(hidden, width, bias=bias)
        self.rotary_emb = LLaDARotaryEmbedding(
            self.head_dim, float(config["rope_theta"]), bool(config["rope_full_precision"])
        )
        self.attention_dropout = float(config["attention_dropout"])
        self.residual_dropout = nn.Dropout(float(config["residual_dropout"]))

    def init_non_persistent_buffers(self) -> None:
        self.rotary_emb.init_non_persistent_buffers()

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch, length, width = hidden_states.shape
        normed = self.attn_norm(hidden_states)
        query = self.q_proj(normed).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        key = (
            self.k_proj(normed).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        )
        value = (
            self.v_proj(normed).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        )
        query, key = self.rotary_emb(query, key)
        if self.n_heads != self.n_kv_heads:
            groups = self.n_heads // self.n_kv_heads
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        hidden_states = hidden_states + self.residual_dropout(self.attn_out(attended))
        normed = self.ff_norm(hidden_states)
        feed_forward = F.silu(self.ff_proj(normed)) * self.up_proj(normed)
        return hidden_states + self.residual_dropout(self.ff_out(feed_forward))


class LLaDATransformer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        embedding_size = int(config["embedding_size"])
        width = int(config["d_model"])
        self.wte = nn.Embedding(embedding_size, width)
        self.emb_drop = nn.Dropout(float(config["embedding_dropout"]))
        self.blocks = nn.ModuleList(LLaDABlock(config) for _ in range(int(config["n_layers"])))
        self.ln_f = LLaDARMSNorm(width, float(config["rms_norm_eps"]))
        self.ff_out = nn.Linear(width, embedding_size, bias=bool(config["include_bias"]))

    def init_non_persistent_buffers(self) -> None:
        for block in self.blocks:
            assert isinstance(block, LLaDABlock)
            block.init_non_persistent_buffers()


class _LLaDABody(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.transformer = LLaDATransformer(config)


class LLaDAForMultiModalGeneration(NativeModel):
    """Native inference-only form of the published multimodal LLaDA model."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key
            for key in config
            if not key.startswith("_") and key not in LLADA_DEFAULTS and key not in _IGNORED
        )
        if unknown:
            raise ValueError(f"Unsupported LLaDA setting {unknown[0]}")
        super().__init__(LLADA_DEFAULTS | dict(config))
        cfg = self.config
        unsupported = {
            "activation_type": "silu",
            "alibi": False,
            "attention_layer_norm": False,
            "block_group_size": 1,
            "block_type": "llama",
            "input_emb_norm": False,
            "layer_norm_type": "rms",
            "rope": True,
            "scale_logits": False,
            "weight_tying": False,
        }
        for name, expected in unsupported.items():
            if cfg[name] != expected:
                raise ValueError(f"Unsupported LLaDA setting {name}={cfg[name]!r}")
        if cfg["d_model"] % cfg["n_heads"] or cfg["n_heads"] % cfg["n_kv_heads"]:
            raise ValueError("Unsupported LLaDA attention head dimensions")
        self.model = _LLaDABody(cfg)

    def init_non_persistent_buffers(self) -> None:
        self.model.transformer.init_non_persistent_buffers()

    def forward(self, input_ids: Tensor) -> Tensor:
        transformer = self.model.transformer
        hidden_states = transformer.emb_drop(transformer.wte(input_ids))
        for block in transformer.blocks:
            hidden_states = block(hidden_states)
        return transformer.ff_out(transformer.ln_f(hidden_states))
