"""GLM-Image autoregressive image-token generator (text-to-image path).

The published checkpoint also holds a vision tower and a VQ encoder that only the image-input
path needs; they are dropped when loading.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.glm import GlmMLP, GlmRMSNorm, _repeat_kv, _use_gqa_in_sdpa

TEXT_DEFAULTS: dict[str, Any] = {
    "vocab_size": 168064,
    "hidden_size": 4096,
    "intermediate_size": 13696,
    "num_hidden_layers": 40,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-5,
    "attention_dropout": 0.0,
    "attention_bias": True,
    "pad_token_id": 167841,
    "vision_vocab_size": 16512,
    "eos_token_id": 16385,
}
ROPE_DEFAULTS: dict[str, Any] = {
    "rope_theta": 10000.0,
    "partial_rotary_factor": 1.0,
    "mrope_section": [8, 12, 12],
}
_EXCLUDED_PREFIXES = ("model.visual.", "model.vqmodel.")


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def _inv_freq(head_dim: int, rope: Mapping[str, Any]) -> Tensor:
    dim = int(head_dim * rope["partial_rotary_factor"])
    return 1.0 / (rope["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))


class GlmImageTextRotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, head_dim: int, rope: Mapping[str, Any]) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rope = dict(rope)
        self.mrope_section = list(rope["mrope_section"])
        self.register_buffer("inv_freq", _inv_freq(head_dim, rope), persistent=False)

    def reset_inv_freq(self) -> None:
        self.inv_freq = _inv_freq(self.head_dim, self.rope)

    def _recompose(self, freq: Tensor) -> Tensor:
        freq = torch.cat(
            [m[i % 3] for i, m in enumerate(freq.split(self.mrope_section, dim=-1))], dim=-1
        )
        return torch.cat([freq, freq], dim=-1)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """``position_ids`` has shape (3, batch, positions): temporal, height, width."""
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        cos = self._recompose(freqs.cos())
        sin = self._recompose(freqs.sin())
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class GlmImageTextAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        self.head_dim = hidden_size // config["num_attention_heads"]
        self.num_key_value_groups = config["num_attention_heads"] // config["num_key_value_heads"]
        self.scaling = self.head_dim**-0.5
        bias = config["attention_bias"]
        self.q_proj = nn.Linear(hidden_size, config["num_attention_heads"] * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, config["num_key_value_heads"] * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, config["num_key_value_heads"] * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(config["num_attention_heads"] * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        cache: list[Tensor] | None,
    ) -> Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = _apply_rotary_pos_emb(
            query_states, key_states, *position_embeddings
        )
        if cache is not None:
            # Concatenation with the cached states, as the reference's dynamic cache does.
            empty = key_states.new_empty((*key_states.shape[:2], 0, key_states.shape[3]))
            keys = cache[0] if cache else empty
            values = cache[1] if cache else empty
            key_states = torch.cat([keys, key_states], dim=-2)
            value_states = torch.cat([values, value_states], dim=-2)
            cache[:] = [key_states, value_states]
        sdpa_kwargs: dict[str, bool] = {}
        if self.num_key_value_groups > 1:
            if _use_gqa_in_sdpa(None, key_states, value_states):
                sdpa_kwargs = {"enable_gqa": True}
            else:
                key_states = _repeat_kv(key_states, self.num_key_value_groups)
                value_states = _repeat_kv(value_states, self.num_key_value_groups)
        query_length = query_states.shape[2]
        key_length = key_states.shape[2]
        is_causal = query_length > 1
        if is_causal and key_length > query_length:
            key_states = key_states[:, :, :query_length, :]
            value_states = value_states[:, :, :query_length, :]
        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            scale=self.scaling,
            is_causal=is_causal,
            **sdpa_kwargs,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention_output)


class GlmImageTextDecoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.self_attn = GlmImageTextAttention(config)
        self.mlp = GlmMLP(config)
        eps = config["rms_norm_eps"]
        self.input_layernorm = GlmRMSNorm(config["hidden_size"], eps)
        self.post_attention_layernorm = GlmRMSNorm(config["hidden_size"], eps)
        self.post_self_attn_layernorm = GlmRMSNorm(config["hidden_size"], eps)
        self.post_mlp_layernorm = GlmRMSNorm(config["hidden_size"], eps)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        cache: list[Tensor] | None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, cache)
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        return residual + hidden_states


class GlmImageTextModel(nn.Module):
    def __init__(self, config: Mapping[str, Any], rope: Mapping[str, Any]) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config["vocab_size"], config["hidden_size"], config["pad_token_id"]
        )
        self.layers = nn.ModuleList(
            GlmImageTextDecoderLayer(config) for _ in range(config["num_hidden_layers"])
        )
        self.norm = GlmRMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.rotary_emb = GlmImageTextRotaryEmbedding(
            config["hidden_size"] // config["num_attention_heads"], rope
        )

    def forward(
        self, input_ids: Tensor, position_ids: Tensor, caches: list[list[Tensor]] | None
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, position_embeddings, None if caches is None else caches[index]
            )
        return self.norm(hidden_states)


class _InnerModel(nn.Module):
    def __init__(self, config: Mapping[str, Any], rope: Mapping[str, Any]) -> None:
        super().__init__()
        self.language_model = GlmImageTextModel(config, rope)


def decode_position_ids(
    prefill_length: int, grids: list[tuple[int, int]], dtype: torch.dtype = torch.long
) -> Tensor:
    """Positions of the tokens that are fed back while generating the image grids.

    ``grids`` lists (height, width) in the order of the processor's grids, the full-size
    image first; the generation order is the reverse (preview first).
    """
    position = prefill_length
    temporal: list[Tensor] = []
    height: list[Tensor] = []
    width: list[Tensor] = []
    for h, w in reversed(grids):
        h_indices = torch.arange(h).unsqueeze(1).expand(h, w).flatten()
        w_indices = torch.arange(w).unsqueeze(0).expand(h, w).flatten()
        temporal.append(torch.full((h * w,), position, dtype=dtype))
        height.append(position + h_indices)
        width.append(position + w_indices)
        position += max(h, w)
    for values in (temporal, height, width):
        values.append(torch.tensor([position], dtype=dtype))
    return torch.stack([torch.cat(temporal), torch.cat(height), torch.cat(width)], dim=0)


def _filter_logits(
    scores: Tensor, temperature: float | None, top_k: int | None, top_p: float | None
) -> Tensor:
    if temperature is not None and temperature != 1.0:
        scores = scores / temperature
    if top_k is not None and top_k != 0:
        k = min(max(top_k, 1), scores.size(-1))
        remove = scores < torch.topk(scores, k)[0][..., -1, None]
        scores = scores.masked_fill(remove, -float("inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_remove = cumulative <= (1 - top_p)
        sorted_remove[..., -1:] = 0
        remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
        scores = scores.masked_fill(remove, -float("inf"))
    return scores


class GlmImageARModel(NativeModel):
    """Text tower and vision LM head of ``GlmImageForConditionalGeneration``."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        text = TEXT_DEFAULTS | dict(config.get("text_config", {}))
        if text["hidden_act"] != "silu":
            raise ValueError("Unsupported GLM-Image activation")
        rope = ROPE_DEFAULTS | dict(text.get("rope_parameters") or {})
        if rope.get("rope_type", "default") != "default":
            raise ValueError("Unsupported GLM-Image rope type")
        self.text_config = text
        eos = text["eos_token_id"]
        self.eos_token_id = int(eos[0] if isinstance(eos, list | tuple) else eos)
        self.model = _InnerModel(text, rope)
        self.lm_head = nn.Linear(text["hidden_size"], text["vision_vocab_size"], bias=False)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        return {name: value for name, value in state.items() if not name.startswith(_EXCLUDED_PREFIXES)}

    def init_non_persistent_buffers(self) -> None:
        self.model.language_model.rotary_emb.reset_inv_freq()

    def _logits(self, input_ids: Tensor, position_ids: Tensor, caches: list[list[Tensor]]) -> Tensor:
        hidden_states = self.model.language_model(input_ids, position_ids, caches)
        return self.lm_head(hidden_states[:, -1:, :])

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        grids: list[tuple[int, int]],
        *,
        max_new_tokens: int,
        temperature: float | None,
        top_k: int | None,
        top_p: float | None,
        generator: torch.Generator,
    ) -> Tensor:
        """Sample image tokens after a text prompt; returns prompt plus generated ids.

        ``grids`` are the (height, width) token grids of the target image: the full-size
        image first, then the low-resolution preview that is generated before it.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape (1, length)")
        length = input_ids.shape[1]
        caches: list[list[Tensor]] = [[] for _ in self.model.language_model.layers]
        position_ids = torch.arange(length, device=input_ids.device).view(1, 1, -1).expand(3, 1, -1)
        later = decode_position_ids(length, grids).to(input_ids.device)
        sequence = input_ids
        logits = self._logits(input_ids, position_ids, caches)
        for step in range(max_new_tokens):
            scores = _filter_logits(
                logits[:, -1].to(copy=True, dtype=torch.float32), temperature, top_k, top_p
            )
            probabilities = F.softmax(scores, dim=-1)
            next_token = torch.multinomial(
                probabilities.cpu(), num_samples=1, generator=generator
            ).squeeze(1)
            next_token = next_token.to(input_ids.device)
            sequence = torch.cat([sequence, next_token[:, None]], dim=-1)
            if int(next_token[0]) == self.eos_token_id or step == max_new_tokens - 1:
                break
            position = later[:, step : step + 1].view(3, 1, 1)
            logits = self._logits(next_token[:, None], position, caches)
        return sequence
