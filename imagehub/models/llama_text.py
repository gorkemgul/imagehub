"""Llama decoder used as a text encoder that returns the output of every layer (HiDream-I1)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.llama import (
    LlamaMLP,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

LLAMA_TEXT_DEFAULTS: dict[str, Any] = {
    "vocab_size": 32000,
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": None,
    "head_dim": None,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "rope_parameters": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "mlp_bias": False,
    "hidden_act": "silu",
    "pad_token_id": None,
}
# Keys of a Llama config.json that do not change the forward pass of the decoder stack.
_IGNORED = {
    "architectures",
    "model_type",
    "bos_token_id",
    "eos_token_id",
    "initializer_range",
    "max_position_embeddings",
    "pretraining_tp",
    "tie_word_embeddings",
    "torch_dtype",
    "dtype",
    "transformers_version",
    "use_cache",
    "generation_config",
}


def _rope_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    """Merge the legacy ``rope_theta``/``rope_scaling`` and ``rope_parameters`` config forms."""
    parameters: dict[str, Any] = {"rope_type": "default", "rope_theta": config["rope_theta"]}
    for source in (config["rope_scaling"], config["rope_parameters"]):
        if source:
            parameters.update(source)
            if "type" in parameters and "rope_type" not in source:
                parameters["rope_type"] = parameters.pop("type")
    return parameters


def normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fill defaults and derived values of a Llama ``config.json``."""
    cfg = LLAMA_TEXT_DEFAULTS | {
        name: config[name] for name in config if name in LLAMA_TEXT_DEFAULTS
    }
    if cfg["num_key_value_heads"] is None:
        cfg["num_key_value_heads"] = cfg["num_attention_heads"]
    if cfg["head_dim"] is None:
        cfg["head_dim"] = cfg["hidden_size"] // cfg["num_attention_heads"]
    return cfg


def inverse_frequencies(config: Mapping[str, Any]) -> Tensor:
    """Rotary inverse frequencies, with the ``llama3`` long-context scaling when configured."""
    cfg = normalize_config(config)
    parameters = _rope_parameters(cfg)
    base = parameters["rope_theta"]
    dim = cfg["head_dim"]
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(torch.float) / dim))
    if parameters["rope_type"] == "default":
        return inv_freq
    factor = parameters["factor"]
    low_freq_factor = parameters["low_freq_factor"]
    high_freq_factor = parameters["high_freq_factor"]
    old_context_len = parameters["original_max_position_embeddings"]
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1 - smooth_factor) * scaled / factor + smooth_factor * scaled
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed, scaled)


class LlamaTextAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size, head_dim = config["hidden_size"], config["head_dim"]
        heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
        self.head_dim = head_dim
        self.num_key_value_groups = heads // kv_heads
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor,
    ) -> Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        # Eager attention: the reference runs it when the encoder is loaded with
        # ``output_attentions=True``, as the pipeline documentation does.
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        weights = weights + attention_mask
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attention = torch.matmul(weights, value).transpose(1, 2).contiguous()
        return self.o_proj(attention.reshape(*input_shape, -1).contiguous())


class LlamaTextLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.self_attn = LlamaTextAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.post_attention_layernorm = LlamaRMSNorm(config["hidden_size"], config["rms_norm_eps"])

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class LlamaTextModel(NativeModel):
    """``LlamaModel`` of a ``LlamaForCausalLM`` checkpoint; its output head is not loaded."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        user = dict(config)
        unknown = sorted(set(user) - set(LLAMA_TEXT_DEFAULTS) - _IGNORED)
        if unknown:
            raise ValueError(f"Unsupported Llama setting {unknown[0]}")
        cfg = normalize_config(user)
        unsupported = {
            "activation": cfg["hidden_act"] != "silu",
            "attention bias": cfg["attention_bias"],
            "attention dropout": cfg["attention_dropout"] != 0.0,
            "MLP bias": cfg["mlp_bias"],
            "head layout": cfg["num_attention_heads"] % cfg["num_key_value_heads"] != 0
            or cfg["head_dim"] % 2 != 0,
        }
        for option, is_unsupported in unsupported.items():
            if is_unsupported:
                raise ValueError(f"Unsupported Llama {option}")
        rope_type = _rope_parameters(cfg)["rope_type"]
        if rope_type not in ("default", "llama3"):
            raise ValueError(f"Unsupported Llama rope type {rope_type}")
        super().__init__(cfg)
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], cfg["pad_token_id"])
        self.layers = nn.ModuleList(LlamaTextLayer(cfg) for _ in range(cfg["num_hidden_layers"]))
        self.norm = LlamaRMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        # The checkpoint is a LlamaForCausalLM: drop the output head, strip the "model." prefix.
        return {
            name.removeprefix("model."): value
            for name, value in state.items()
            if name != "lm_head.weight"
        }

    def _rotary(self, hidden_states: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq = inverse_frequencies(self.config).to(hidden_states.device)
        expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        positions = position_ids[:, None, :].float()
        freqs = (expanded @ positions).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(hidden_states.dtype), emb.sin().to(hidden_states.dtype)

    @torch.inference_mode()
    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Return the outputs of all layers stacked as (layers, batch, length, width).

        These equal ``hidden_states[1:]`` of the reference: the output of each decoder layer, the
        last one after the final norm.
        """
        hidden_states = self.embed_tokens(input_ids)
        length = hidden_states.shape[1]
        position_ids = torch.arange(length, device=hidden_states.device)[None]
        position_embeddings = self._rotary(hidden_states, position_ids)
        allowed = torch.ones(length, length, dtype=torch.bool, device=hidden_states.device).tril()
        allowed = allowed[None, None]
        if attention_mask is not None:
            allowed = allowed & attention_mask.bool()[:, None, None, :]
        mask = torch.zeros(allowed.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        mask.masked_fill_(~allowed, torch.finfo(hidden_states.dtype).min)
        outputs = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, mask)
            outputs.append(hidden_states)
        outputs[-1] = self.norm(outputs[-1])
        return torch.stack(outputs, dim=0)
