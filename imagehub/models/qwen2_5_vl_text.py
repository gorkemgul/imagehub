"""Qwen2.5-VL text model used by LongCat-Image."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.qwen3 import Qwen3RMSNorm, repeat_kv, rotate_half

QWEN25_VL_TEXT_DEFAULTS: dict[str, Any] = {
    "vocab_size": 152064,
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "max_position_embeddings": 128000,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "rope_scaling": {"type": "default", "mrope_section": [16, 24, 24]},
    "attention_bias": True,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "use_sliding_window": False,
    "tie_word_embeddings": False,
}


def _use_gqa_in_sdpa(attention_mask: Tensor | None, key: Tensor, value: Tensor) -> bool:
    """Follow transformers: grouped heads go to SDPA directly when the host allows masked GQA.

    The choice depends on the host, not on the tensors' device: with MPS available and torch
    2.13 or newer, transformers passes masked attention with ``enable_gqa``.
    """
    if torch.backends.mps.is_available():
        return tuple(int(part) for part in torch.__version__.split(".")[:2]) >= (2, 13)
    return attention_mask is None and key.shape[-1] == value.shape[-1] <= 256


_LEGACY_TEXT_PREFIX = re.compile(r"^model(?!\.(language_model|vit|vision_tower))")


class Qwen25VLRMSNorm(Qwen3RMSNorm):
    pass


class Qwen25VLMLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen25VLRotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        head_dim = config["hidden_size"] // config["num_attention_heads"]
        base = config["rope_theta"]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        rope_scaling = config.get("rope_scaling") or config.get("rope_parameters") or {}
        self.mrope_section = rope_scaling.get("mrope_section", [16, 24, 24])

    @torch.no_grad()
    def forward(self, hidden_states: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        )
        positions = position_ids[:, :, None, :].float()
        frequencies = (inv_freq @ positions).transpose(2, 3)
        cos = self._recompose(frequencies.cos())
        sin = self._recompose(frequencies.sin())
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)

    def _recompose(self, frequencies: Tensor) -> Tensor:
        chunks = frequencies.split(self.mrope_section, dim=-1)
        frequencies = torch.cat([chunk[index % 3] for index, chunk in enumerate(chunks)], dim=-1)
        return torch.cat((frequencies, frequencies), dim=-1)


def apply_multimodal_rotary_pos_emb(
    query: Tensor, key: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


class Qwen25VLAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.head_dim = hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config["attention_dropout"]
        bias = config["attention_bias"]
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_multimodal_rotary_pos_emb(
            query, key, position_embeddings[0], position_embeddings[1]
        )

        # The pipeline calls the text encoder with transformers' default ``use_cache=True``; the
        # key-value cache stores keys and values as contiguous copies, and the SDPA kernel's
        # summation order depends on that layout.
        key = key.contiguous()
        value = value.contiguous()

        sdpa_kwargs: dict[str, bool] = {}
        if self.num_key_value_groups > 1:
            if _use_gqa_in_sdpa(attention_mask, key, value):
                sdpa_kwargs["enable_gqa"] = True
            else:
                key = repeat_kv(key, self.num_key_value_groups)
                value = repeat_kv(value, self.num_key_value_groups)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
            is_causal=attention_mask is None,
            **sdpa_kwargs,
        )
        output = output.transpose(1, 2).contiguous().reshape(batch_size, sequence_length, -1)
        return self.o_proj(output)


class Qwen25VLDecoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.self_attn = Qwen25VLAttention(config)
        self.mlp = Qwen25VLMLP(config)
        self.input_layernorm = Qwen25VLRMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.post_attention_layernorm = Qwen25VLRMSNorm(
            config["hidden_size"], config["rms_norm_eps"]
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen25VLLanguageModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            Qwen25VLDecoderLayer(config) for _ in range(config["num_hidden_layers"])
        )
        self.norm = Qwen25VLRMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.rotary_emb = Qwen25VLRotaryEmbedding(config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        hidden_states = self.embed_tokens(input_ids)
        sequence_length = hidden_states.shape[1]
        positions = torch.arange(sequence_length, device=hidden_states.device)
        if position_ids is None:
            position_ids = positions.view(1, 1, -1).expand(3, hidden_states.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask = None
        if not bool(attention_mask.bool().all()):
            causal_mask = positions[:, None] >= positions[None, :]
            causal_mask = causal_mask[None, None] & attention_mask[:, None, None, :].bool()

        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, causal_mask)
            all_hidden_states.append(hidden_states)
        all_hidden_states[-1] = self.norm(hidden_states)
        return tuple(all_hidden_states)


class Qwen25VLModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.language_model = Qwen25VLLanguageModel(config)


class Qwen25VLTextModel(NativeModel):
    """Text-only subset of Qwen2.5-VL with the conditional-generation LM head."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        text_config = QWEN25_VL_TEXT_DEFAULTS | dict(config.get("text_config", config))
        super().__init__(config)
        rope_scaling = text_config.get("rope_scaling") or text_config.get("rope_parameters") or {}
        unsupported = {
            "activation": text_config["hidden_act"] != "silu",
            # Published configs use the legacy ``"type": "mrope"``, which transformers reads as the
            # default RoPE with multimodal sections.
            "RoPE type": rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
            not in {"default", "mrope"},
            "sliding window": text_config["use_sliding_window"],
            "tied embeddings": text_config["tie_word_embeddings"],
        }
        for option, is_unsupported in unsupported.items():
            if is_unsupported:
                raise ValueError(f"Unsupported Qwen2.5-VL text {option}")
        self.text_config = text_config
        self.model = Qwen25VLModel(text_config)
        self.lm_head = nn.Linear(text_config["hidden_size"], text_config["vocab_size"], bias=False)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Discard the vision tower from a full conditional-generation checkpoint.

        Checkpoints saved with transformers 4.x name the text model ``model.layers.*``; transformers
        5 renames ``model.*`` (outside ``language_model``) to ``model.language_model.*`` on load.
        """
        renamed = {
            _LEGACY_TEXT_PREFIX.sub("model.language_model", name, count=1): tensor
            for name, tensor in state.items()
        }
        return {
            name: tensor
            for name, tensor in renamed.items()
            if name.startswith("model.language_model.") or name == "lm_head.weight"
        }

    def init_non_persistent_buffers(self) -> None:
        head_dim = self.text_config["hidden_size"] // self.text_config["num_attention_heads"]
        base = self.text_config["rope_theta"]
        self.model.language_model.rotary_emb.inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, ...]:
        """Return decoder hidden states, ending with the normalized pipeline embedding."""
        return self.model.language_model(input_ids, attention_mask)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        max_new_tokens: int,
        generation_config: Mapping[str, Any],
    ) -> Tensor:
        """Generate rewritten-prompt tokens with the published deterministic sampler."""
        generated = input_ids.clone()
        mask = attention_mask.clone()
        eos_token_ids = generation_config["eos_token_id"]
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        eos = torch.tensor(eos_token_ids, device=input_ids.device)
        unfinished = torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        pad_token_id = int(generation_config["pad_token_id"])

        for _ in range(max_new_tokens):
            position_ids = mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(mask == 0, 0)
            hidden_states = self.model.language_model(generated, mask, position_ids)[-1]
            scores = self.lm_head(hidden_states[:, -1])
            penalty = float(generation_config.get("repetition_penalty", 1.0))
            if penalty != 1.0:
                gathered = torch.gather(scores, 1, generated)
                gathered = torch.where(gathered < 0, gathered * penalty, gathered / penalty)
                scores.scatter_(1, generated, gathered)
            temperature = float(generation_config.get("temperature", 1.0))
            if temperature != 1.0:
                scores = scores / temperature
            top_k = int(generation_config.get("top_k", 0))
            if top_k > 0:
                top_k = min(top_k, scores.shape[-1])
                threshold = torch.topk(scores, top_k)[0][..., -1, None]
                scores.masked_fill_(scores < threshold, -float("inf"))
            top_p = float(generation_config.get("top_p", 1.0))
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scores, descending=False)
                remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) <= 1 - top_p
                remove[..., -1:] = False
                remove = remove.scatter(1, sorted_indices, remove)
                scores.masked_fill_(remove, -float("inf"))
            if generation_config.get("do_sample", False):
                next_token = torch.multinomial(scores.softmax(dim=-1), num_samples=1).squeeze(1)
            else:
                next_token = scores.argmax(dim=-1)
            next_token = torch.where(unfinished, next_token, pad_token_id)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            mask = torch.cat((mask, torch.ones_like(next_token[:, None])), dim=1)
            unfinished &= ~torch.isin(next_token, eos)
            if not bool(unfinished.any()):
                break
        return generated
