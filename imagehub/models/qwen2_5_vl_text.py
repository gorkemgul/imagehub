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


_LEGACY_TEXT_PREFIX = re.compile(r"^model(?!\.(language_model|visual|vit|vision_tower))")

KeyValueCache = list[tuple[Tensor, Tensor] | None]


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
    def __init__(self, config: Mapping[str, Any], layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
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
        key_value_cache: KeyValueCache | None = None,
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

        if key_value_cache is not None:
            cached = key_value_cache[self.layer_index]
            if cached is None:
                cached_key = torch.tensor([], dtype=key.dtype, device=key.device)
                cached_value = torch.tensor([], dtype=value.dtype, device=value.device)
            else:
                cached_key, cached_value = cached
            key = torch.cat([cached_key, key], dim=-2)
            value = torch.cat([cached_value, value], dim=-2)
            key_value_cache[self.layer_index] = (key, value)
        else:
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
            is_causal=attention_mask is None and query.shape[2] > 1,
            **sdpa_kwargs,
        )
        output = output.transpose(1, 2).contiguous().reshape(batch_size, sequence_length, -1)
        return self.o_proj(output)


class Qwen25VLDecoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any], layer_index: int) -> None:
        super().__init__()
        self.self_attn = Qwen25VLAttention(config, layer_index)
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
        key_value_cache: KeyValueCache | None = None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_embeddings, attention_mask, key_value_cache
        )
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
            Qwen25VLDecoderLayer(config, index) for index in range(config["num_hidden_layers"])
        )
        self.norm = Qwen25VLRMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.rotary_emb = Qwen25VLRotaryEmbedding(config)

    def forward(
        self,
        input_ids: Tensor | None,
        attention_mask: Tensor | None,
        position_ids: Tensor | None = None,
        key_value_cache: KeyValueCache | None = None,
        *,
        inputs_embeds: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        sequence_length = hidden_states.shape[1]
        past_length = 0
        if key_value_cache is not None and key_value_cache[0] is not None:
            past_length = key_value_cache[0][0].shape[-2]
        positions = torch.arange(
            past_length, past_length + sequence_length, device=hidden_states.device
        )
        if position_ids is None:
            position_ids = positions.view(1, 1, -1).expand(3, hidden_states.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask = None
        if attention_mask is not None and not bool(attention_mask.bool().all()):
            key_positions = torch.arange(attention_mask.shape[-1], device=hidden_states.device)
            causal_mask = positions[:, None] >= key_positions[None, :]
            causal_mask = causal_mask[None, None] & attention_mask[:, None, None, :].bool()

        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, causal_mask, key_value_cache)
            all_hidden_states.append(hidden_states)
        all_hidden_states[-1] = self.norm(hidden_states)
        return tuple(all_hidden_states)


class Qwen25VLModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.language_model = Qwen25VLLanguageModel(config)


class Qwen25VLVisionPatchEmbed(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        kernel = (
            config["temporal_patch_size"],
            config["patch_size"],
            config["patch_size"],
        )
        self.proj = nn.Conv3d(
            config.get("in_channels", config.get("in_chans", 3)),
            config["hidden_size"],
            kernel,
            stride=kernel,
            bias=False,
        )

    def forward(self, pixels: Tensor) -> Tensor:
        c = self.config
        pixels = pixels.reshape(
            -1,
            c.get("in_channels", c.get("in_chans", 3)),
            c["temporal_patch_size"],
            c["patch_size"],
            c["patch_size"],
        )
        return self.proj(pixels.to(self.proj.weight.dtype)).reshape(-1, c["hidden_size"])


class Qwen25VLVisionAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.num_heads = config["num_heads"]
        self.head_dim = config["hidden_size"] // self.num_heads
        self.qkv = nn.Linear(config["hidden_size"], config["hidden_size"] * 3)
        self.proj = nn.Linear(config["hidden_size"], config["hidden_size"])

    def forward(
        self, hidden_states: Tensor, cos: Tensor, sin: Tensor, lengths: list[int]
    ) -> Tensor:
        sequence_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(sequence_length, 3, self.num_heads, self.head_dim)
            .unbind(1)
        )
        query_dtype, key_dtype = query.dtype, key.dtype
        query, key = query.float(), key.float()
        cos, sin = cos[:, None].float(), sin[:, None].float()
        query = (query * cos + rotate_half(query) * sin).to(query_dtype)
        key = (key * cos + rotate_half(key) * sin).to(key_dtype)
        output = []
        for q, k, v in zip(
            torch.split(query, lengths),
            torch.split(key, lengths),
            torch.split(value, lengths),
            strict=True,
        ):
            attended = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None],
                k.transpose(0, 1)[None],
                v.transpose(0, 1)[None],
                scale=self.head_dim**-0.5,
            )
            output.append(attended.squeeze(0).transpose(0, 1))
        return self.proj(torch.cat(output).reshape(sequence_length, -1))


class Qwen25VLVisionBlock(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        self.norm1 = Qwen25VLRMSNorm(hidden_size, 1e-6)
        self.norm2 = Qwen25VLRMSNorm(hidden_size, 1e-6)
        self.attn = Qwen25VLVisionAttention(config)
        self.mlp = Qwen25VLMLP(
            {
                "hidden_size": hidden_size,
                "intermediate_size": config["intermediate_size"],
            }
        )
        # Vision MLP projections have biases; the language projections do not.
        self.mlp.gate_proj = nn.Linear(hidden_size, config["intermediate_size"], bias=True)
        self.mlp.up_proj = nn.Linear(hidden_size, config["intermediate_size"], bias=True)
        self.mlp.down_proj = nn.Linear(config["intermediate_size"], hidden_size, bias=True)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, lengths: list[int]) -> Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, lengths)
        return x + self.mlp(self.norm2(x))


class Qwen25VLPatchMerger(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        merged_size = hidden_size * config["spatial_merge_size"] ** 2
        self.ln_q = Qwen25VLRMSNorm(hidden_size, 1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(merged_size, merged_size),
            nn.GELU(),
            nn.Linear(merged_size, config["out_hidden_size"]),
        )
        self.merged_size = merged_size

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.ln_q(x).reshape(-1, self.merged_size))


class Qwen25VLVisionModel(nn.Module):
    """Qwen2.5-VL vision tower with its published window-attention ordering."""

    inv_freq: Tensor

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.patch_embed = Qwen25VLVisionPatchEmbed(config)
        self.blocks = nn.ModuleList(Qwen25VLVisionBlock(config) for _ in range(config["depth"]))
        self.merger = Qwen25VLPatchMerger(config)
        head_dim = config["hidden_size"] // config["num_heads"]
        self.register_buffer(
            "inv_freq",
            1 / (10_000 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2))),
            persistent=False,
        )

    @staticmethod
    def _position_ids(grid: Tensor, merge: int) -> Tensor:
        result = []
        for time, height, width in grid.tolist():
            rows, columns = torch.meshgrid(
                torch.arange(height, device=grid.device),
                torch.arange(width, device=grid.device),
                indexing="ij",
            )
            shape = (height // merge, merge, width // merge, merge)
            rows = rows.reshape(shape).transpose(1, 2).flatten()
            columns = columns.reshape(shape).transpose(1, 2).flatten()
            result.append(torch.stack((rows, columns), -1).repeat(time, 1))
        return torch.cat(result)

    def _window_indices(self, grid: Tensor) -> tuple[Tensor, list[int]]:
        c = self.config
        merge = c["spatial_merge_size"]
        window = c["window_size"] // merge // c["patch_size"]
        indices, lengths = [], []
        offset = 0
        for time, height, width in grid.tolist():
            gh, gw = height // merge, width // merge
            index = torch.arange(time * gh * gw).reshape(time, gh, gw)
            pad_h = window - gh % window
            pad_w = window - gw % window
            padded = F.pad(index, (0, pad_w, 0, pad_h), value=-100)
            padded = padded.reshape(time, (gh + pad_h) // window, window, -1, window)
            padded = padded.permute(0, 1, 3, 2, 4).reshape(-1, window, window)
            lengths.extend(((padded != -100).sum((1, 2)) * merge**2).tolist())
            flat = padded.flatten()
            indices.append(flat[flat != -100] + offset)
            offset += time * gh * gw
        return torch.cat(indices).to(grid.device), lengths

    def forward(self, pixels: Tensor, grid: Tensor) -> Tensor:
        c = self.config
        merge = c["spatial_merge_size"]
        positions = self._position_ids(grid, merge)
        window_indices, window_lengths = self._window_indices(grid)
        full_lengths = (grid[:, 1] * grid[:, 2]).repeat_interleave(grid[:, 0]).tolist()
        x = self.patch_embed(pixels)
        sequence_length = x.shape[0]
        x = x.reshape(sequence_length // merge**2, merge**2, -1)[window_indices].reshape(
            sequence_length, -1
        )
        positions = positions.reshape(sequence_length // merge**2, merge**2, -1)[
            window_indices
        ].reshape(sequence_length, -1)
        angles = positions[..., None].float() * self.inv_freq.float()
        frequencies = torch.cat((angles[:, 0], angles[:, 1]), -1)
        cos = torch.cat((frequencies, frequencies), -1).cos()
        sin = torch.cat((frequencies, frequencies), -1).sin()
        full_blocks = set(c["fullatt_block_indexes"])
        for index, block in enumerate(self.blocks):
            x = block(x, cos, sin, full_lengths if index in full_blocks else window_lengths)
        reverse = torch.argsort(window_indices)
        return self.merger(x)[reverse]


class Qwen25VLForConditionalGeneration(NativeModel):
    """Full Qwen2.5-VL conditioner used by image-editing pipelines."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(dict(config))
        text_config = QWEN25_VL_TEXT_DEFAULTS | dict(config.get("text_config", config))
        self.text_config = text_config
        self.visual = Qwen25VLVisionModel(config["vision_config"])
        self.model = Qwen25VLLanguageModel(text_config)
        self.lm_head = nn.Linear(text_config["hidden_size"], text_config["vocab_size"], bias=False)

    def init_non_persistent_buffers(self) -> None:
        head_dim = self.text_config["hidden_size"] // self.text_config["num_attention_heads"]
        self.model.rotary_emb.inv_freq = 1.0 / (
            self.text_config["rope_theta"]
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        vision = self.config["vision_config"]
        vision_head_dim = vision["hidden_size"] // vision["num_heads"]
        self.visual.inv_freq = 1 / (
            10_000
            ** (
                torch.arange(0, vision_head_dim // 2, 2, dtype=torch.float32)
                / (vision_head_dim // 2)
            )
        )

    @staticmethod
    def _rope_positions(ids: Tensor, types: Tensor, grid: Tensor, merge: int) -> Tensor:
        positions = torch.zeros(3, *ids.shape, dtype=torch.long, device=ids.device)
        image_index = 0
        for batch, row in enumerate(types.tolist()):
            current = start = 0
            for end in range(1, len(row) + 1):
                if end < len(row) and row[end] == row[start]:
                    continue
                if row[start] == 0:
                    positions[:, batch, start:end] = (
                        torch.arange(end - start, device=ids.device) + current
                    )
                    current += end - start
                else:
                    time, height, width = grid[image_index].tolist()
                    image_index += 1
                    rows, columns = torch.meshgrid(
                        torch.arange(height // merge, device=ids.device),
                        torch.arange(width // merge, device=ids.device),
                        indexing="ij",
                    )
                    temporal = torch.arange(time, device=ids.device)[:, None].expand(
                        -1, rows.numel()
                    )
                    spatial = torch.stack((rows, columns)).reshape(2, 1, -1).expand(-1, time, -1)
                    block = torch.cat((temporal[None], spatial), 0).reshape(3, -1) + current
                    positions[:, batch, start:end] = block
                    current += max(time, height // merge, width // merge)
                start = end
        return positions

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        embeddings = self.model.embed_tokens(input_ids)
        position_ids = None
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required with pixel_values")
            visual = self.visual(pixel_values, image_grid_thw).to(embeddings.dtype)
            mask = input_ids.eq(self.config["image_token_id"])
            if int(mask.sum()) != visual.shape[0]:
                raise ValueError("image token count does not match vision features")
            embeddings = embeddings.masked_scatter(mask[..., None], visual)
            position_ids = self._rope_positions(
                input_ids,
                mask.long(),
                image_grid_thw,
                int(self.config["vision_config"]["spatial_merge_size"]),
            )
        return self.model(
            None,
            attention_mask,
            position_ids=position_ids,
            inputs_embeds=embeddings,
        )


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
        position_ids = mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(mask == 0, 0)
        key_value_cache: KeyValueCache = [None] * len(self.model.language_model.layers)
        eos_token_ids = generation_config["eos_token_id"]
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        eos = torch.tensor(eos_token_ids, device=input_ids.device)
        unfinished = torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        pad_token_id = int(generation_config["pad_token_id"])

        next_input_ids = generated
        for _ in range(max_new_tokens):
            model_attention_mask = None if bool(mask.bool().all()) else mask
            hidden_states = self.model.language_model(
                next_input_ids,
                model_attention_mask,
                position_ids[..., -next_input_ids.shape[1] :],
                key_value_cache,
            )[-1]
            scores = self.lm_head(hidden_states[:, -1:])[:, -1].float()
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
            position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=-1)
            next_input_ids = next_token[:, None]
            unfinished &= ~torch.isin(next_token, eos)
            if not bool(unfinished.any()):
                break
        return generated
