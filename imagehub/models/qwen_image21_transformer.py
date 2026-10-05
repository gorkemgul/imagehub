"""Native Qwen-Image 2.1 transformer for text-to-image inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

_IMG_TOKENS_PER_SLOT = 4


class QwenImage21KVLayerCache:
    def __init__(self) -> None:
        self.k: Tensor | None = None
        self.v: Tensor | None = None

    def store(self, k: Tensor, v: Tensor) -> None:
        self.k = k
        self.v = v

    def get(self) -> tuple[Tensor, Tensor]:
        if self.k is None or self.v is None:
            raise RuntimeError("KV cache has not been populated yet.")
        return self.k, self.v


class QwenImage21KVCache:
    def __init__(self, num_layers: int) -> None:
        self.layer_caches = [QwenImage21KVLayerCache() for _ in range(num_layers)]

    def get_layer(self, layer_idx: int) -> QwenImage21KVLayerCache:
        return self.layer_caches[layer_idx]


def apply_rotary_emb_qwen(x: Tensor, freqs_cis: Tensor) -> Tensor:
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class QwenImage21TemporalTimesteps(nn.Module):
    freqs: Tensor

    def __init__(
        self, timestep_dim: int, max_period: int = 10000, time_factor: float = 1000.0
    ) -> None:
        super().__init__()
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor
        half = timestep_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timestep: Tensor) -> Tensor:
        timestep = self.time_factor * timestep.float()
        args = timestep[:, None] * self.freqs[None].to(timestep.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.timestep_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(timestep.dtype)


class QwenImage21TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(256, embedding_dim, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, sample: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class QwenImage21TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.time_proj = QwenImage21TemporalTimesteps(timestep_dim=256)
        self.timestep_embedder = QwenImage21TimestepEmbedding(embedding_dim)

    def forward(self, timestep: Tensor, hidden_states: Tensor) -> Tensor:
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))


class QwenImage21ZeroCenterRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        rrms = torch.rsqrt(torch.mean(hidden_states**2, dim=-1, keepdim=True) + self.eps)
        return (hidden_states * rrms * (self.weight.float() + 1)).to(input_dtype)


class QwenImage21RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        else:
            hidden_states = hidden_states.to(input_dtype)
        return hidden_states * self.weight


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.text_norm(hidden_states)
        hidden_states = self.in_layer(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.out_layer(hidden_states)


class QwenImage21SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.out(
            self.activation_fn(self.gate_layer(hidden_states)) * self.proj(hidden_states)
        )


def _select_modulation_rows(params: Tensor, target_token_mask: Tensor | None) -> Tensor:
    if target_token_mask is None:
        return params.unsqueeze(1)
    real, zero = params[:-1].unsqueeze(1), params[-1:].unsqueeze(0)
    return torch.where(target_token_mask.view(1, -1, 1), real, zero)


class QwenImage21AdaLayerNormContinuous(nn.Module):
    def __init__(
        self, embedding_dim: int, conditioning_embedding_dim: int, eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=False, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        conditioning_embedding: Tensor,
        target_token_mask: Tensor | None = None,
    ) -> Tensor:
        scale = self.linear(self.silu(conditioning_embedding).to(hidden_states.dtype))
        scale = _select_modulation_rows(scale, target_token_mask)
        return self.norm(hidden_states) * (1 + scale)


def _qwenimage21_prefix_segments(image_ids: Tensor, prefix_len: int) -> list[tuple[int, int, bool]]:
    prefix_ids = image_ids[:prefix_len].tolist()
    segments = []
    start = 0
    for index in range(1, prefix_len + 1):
        if index == prefix_len or prefix_ids[index] != prefix_ids[start]:
            segments.append((start, index, prefix_ids[start] < 0))
            start = index
    return segments


def _attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None) -> Tensor:
    query, key, value = (item.permute(0, 2, 1, 3) for item in (query, key, value))
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    return output.permute(0, 2, 1, 3)


class QwenImage21Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(heads * dim_head, dim, bias=False), nn.Dropout(0.0)])
        self.norm_q = QwenImage21RMSNorm(dim_head, eps=eps)
        self.norm_k = QwenImage21RMSNorm(dim_head, eps=eps)

    def forward(
        self,
        hidden_states: Tensor,
        rotary_emb: Tensor,
        layer_cache: QwenImage21KVLayerCache | None,
        kv_cache_mode: str | None,
        cache_write_slice: slice | None,
        segments: list[tuple[int, int, bool]] | None,
        key_valid: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor:
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))
        query = self.norm_q(query).to(value.dtype)
        key = self.norm_k(key).to(value.dtype)
        query = apply_rotary_emb_qwen(query, rotary_emb)
        key = apply_rotary_emb_qwen(key, rotary_emb)

        if layer_cache is not None:
            if kv_cache_mode == "extract" and cache_write_slice is not None:
                layer_cache.store(
                    key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone()
                )
            elif kv_cache_mode == "cached":
                cached_k, cached_v = layer_cache.get()
                key = torch.cat([cached_k, key], dim=1)
                value = torch.cat([cached_v, value], dim=1)

        if segments is None:
            output = _attention(query, key, value, attention_mask)
        else:
            prefix_len = segments[-1][1] if segments else 0
            outputs = []
            for start, end, is_text in segments:
                segment_mask = None
                if is_text:
                    segment_length = end - start
                    segment_mask = torch.cat(
                        [
                            torch.ones(
                                segment_length, start, dtype=torch.bool, device=query.device
                            ),
                            torch.tril(
                                torch.ones(
                                    segment_length,
                                    segment_length,
                                    dtype=torch.bool,
                                    device=query.device,
                                )
                            ),
                        ],
                        dim=1,
                    )[None, None]
                if key_valid is not None:
                    segment_key_valid = key_valid[:, None, None, :end]
                    segment_mask = (
                        segment_key_valid
                        if segment_mask is None
                        else segment_mask & segment_key_valid
                    )
                outputs.append(
                    _attention(query[:, start:end], key[:, :end], value[:, :end], segment_mask)
                )
            outputs.append(
                _attention(
                    query[:, prefix_len:],
                    key,
                    value,
                    None if key_valid is None else key_valid[:, None, None, :],
                )
            )
            output = torch.cat(outputs, dim=1)
        output = output[:, : query.shape[1]].flatten(2, 3).type_as(query)
        return self.to_out[1](self.to_out[0](output))


class QwenImage21TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: int = 3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenImage21Attention(dim, num_attention_heads, attention_head_dim, eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenImage21SwiGLUFeedForward(dim, dim * mlp_ratio)

    @staticmethod
    def _modulate(
        hidden_states: Tensor, mod_params: Tensor, target_token_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        scale, gate = mod_params.chunk(2, dim=-1)
        scale = _select_modulation_rows(scale, target_token_mask)
        gate = _select_modulation_rows(gate, target_token_mask)
        return hidden_states * (1 + scale), gate

    def forward(
        self,
        hidden_states: Tensor,
        modulation: Tensor,
        rotary_emb: Tensor,
        target_token_mask: Tensor | None,
        layer_cache: QwenImage21KVLayerCache | None,
        kv_cache_mode: str | None,
        cache_write_slice: slice | None,
        segments: list[tuple[int, int, bool]] | None,
        key_valid: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor:
        mod1, mod2 = modulation.chunk(2, dim=-1)
        modulated, gate1 = self._modulate(self.img_norm1(hidden_states), mod1, target_token_mask)
        attention_output = self.attn(
            modulated,
            rotary_emb,
            layer_cache,
            kv_cache_mode,
            cache_write_slice,
            segments,
            key_valid,
            attention_mask,
        )
        hidden_states = hidden_states + gate1.tanh() * attention_output
        modulated, gate2 = self._modulate(self.img_norm2(hidden_states), mod2, target_token_mask)
        hidden_states = hidden_states + gate2.tanh() * self.img_mlp(modulated)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


class QwenImage21Rope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.freqs = self._make_freqs()

    @staticmethod
    def rope_params(index: Tensor, dim: int, theta: int = 10000) -> Tensor:
        freqs = torch.outer(
            index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))
        )
        return torch.polar(torch.ones_like(freqs), freqs)

    def _make_freqs(self) -> list[Tensor]:
        pos_index = torch.arange(8192)
        neg_index = torch.arange(1024).flip(0) * -1 - 1
        return [
            torch.cat(
                [
                    self.rope_params(pos_index, dim, self.theta),
                    self.rope_params(neg_index, dim, self.theta),
                ],
                dim=0,
            )
            for dim in self.axes_dim
        ]

    def forward(
        self, img_shapes: list[tuple[int, int, int]], image_pad_mask: Tensor, device: torch.device
    ) -> Tensor:
        self.freqs = [freq.to(device) for freq in self.freqs]
        frame_index: list[int] = []
        image_height_index: list[int] = []
        image_width_index: list[int] = []
        cursor, position = 0, 0
        total_len = image_pad_mask.shape[-1]
        is_image_token = image_pad_mask.tolist()
        for _, height, width in img_shapes:
            block_start = is_image_token.index(True, cursor)
            text_len = block_start - cursor
            frame_index.extend(range(position, position + text_len))
            position += text_len
            cursor = block_start + height * width
            frame_index.extend([position] * (height * width))
            position += max(height, width)
            image_height_index.extend(
                h for h in range(-(height - height // 2), height // 2) for _ in range(width)
            )
            image_width_index.extend(
                w for _ in range(height) for w in range(-(width - width // 2), width // 2)
            )
        if cursor < total_len:
            frame_index.extend(range(position, position + total_len - cursor))
        frame = torch.tensor(frame_index, dtype=torch.long, device=device)
        height_positions = frame.clone()
        width_positions = frame.clone()
        height_positions[image_pad_mask] = torch.tensor(
            image_height_index, dtype=torch.long, device=device
        )
        width_positions[image_pad_mask] = torch.tensor(
            image_width_index, dtype=torch.long, device=device
        )
        return torch.cat(
            [
                self.freqs[0][frame],
                self.freqs[1][height_positions],
                self.freqs[2][width_positions],
            ],
            dim=-1,
        )


class QwenImage21Transformer2DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = {name: value for name, value in config.items() if not name.startswith("_")}
        super().__init__(cfg)
        patch_size = int(cfg.get("patch_size", 1))
        in_channels = int(cfg.get("in_channels", 64))
        out_channels = int(cfg.get("out_channels") or in_channels)
        num_layers = int(cfg.get("num_layers", 32))
        attention_head_dim = int(cfg.get("attention_head_dim", 128))
        num_attention_heads = int(cfg.get("num_attention_heads", 32))
        context_in_dim = int(cfg.get("context_in_dim", 4096))
        mlp_ratio = int(cfg.get("mlp_ratio", 3))
        axes_dims_rope = list(cfg.get("axes_dims_rope", (16, 56, 56)))
        eps = float(cfg.get("eps", 1e-6))
        self.causal_condition = bool(cfg.get("causal_condition", True))
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = QwenImage21Rope(10000, axes_dims_rope)
        self.time_text_embed = QwenImage21TimestepProjEmbeddings(self.inner_dim)
        self.txt_in = QwenImage21TextProjection(context_in_dim, self.inner_dim, eps)
        self.img_in = nn.Linear(in_channels * patch_size * patch_size, self.inner_dim, bias=False)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False)
        )
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = QwenImage21AdaLayerNormContinuous(self.inner_dim, self.inner_dim, eps)
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * out_channels, bias=False
        )

    def init_non_persistent_buffers(self) -> None:
        half = self.time_text_embed.time_proj.timestep_dim // 2
        self.time_text_embed.time_proj.freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        )
        self.pos_embed.freqs = self.pos_embed._make_freqs()

    @staticmethod
    def build_token_metadata(
        image_pad_mask: Tensor, img_shapes: list[tuple[int, int, int]]
    ) -> tuple[Tensor, Tensor]:
        image_positions = image_pad_mask.nonzero(as_tuple=True)[0]
        block_lengths = [math.prod(shape) for shape in img_shapes]
        if sum(block_lengths) != image_positions.numel():
            described_tokens = sum(block_lengths)
            raise ValueError(
                f"img_shapes accounts for {described_tokens} image tokens but image_pad_mask marks "
                f"{image_positions.numel()}."
            )
        image_ids = torch.full_like(image_pad_mask, -1, dtype=torch.long)
        block_ids = torch.repeat_interleave(
            torch.arange(len(block_lengths), device=image_pad_mask.device),
            torch.tensor(block_lengths, device=image_pad_mask.device),
        )
        image_ids[image_positions] = block_ids
        target_token_mask = torch.zeros_like(image_pad_mask)
        target_token_mask[image_positions[-block_lengths[-1] :]] = True
        return image_ids, target_token_mask

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: Tensor,
        encoder_hidden_states_mask: Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        kv_cache: QwenImage21KVCache | None = None,
        kv_cache_mode: str | None = None,
        return_dict: bool | None = None,
    ) -> Tensor:
        del attention_kwargs, return_dict
        if any(len(shapes) != 1 for shapes in img_shapes):
            raise ValueError("Qwen-Image 2.1 condition images are not supported")
        if kv_cache is not None and not self.causal_condition:
            raise ValueError("kv_cache requires causal_condition=True")
        if kv_cache is not None and kv_cache_mode not in ("extract", "cached"):
            raise ValueError(
                "kv_cache_mode must be 'extract' or 'cached' when kv_cache is provided"
            )
        if kv_cache is None and kv_cache_mode is not None:
            raise ValueError("kv_cache_mode was provided without a kv_cache")

        batch_size = hidden_states.shape[0]
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        repeats = torch.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        target_tokens = math.prod(img_shapes[0][-1])
        joint_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                encoder_hidden_states.new_zeros(
                    batch_size, target_tokens // 4, encoder_hidden_states.shape[2]
                ),
            ],
            dim=1,
        )
        joint_hidden_states = joint_hidden_states.repeat_interleave(repeats, dim=1)
        joint_hidden_states[:, image_pad_mask] = hidden_states
        rotary_emb = self.pos_embed(img_shapes[0], image_pad_mask, hidden_states.device)
        image_ids, target_token_mask = self.build_token_metadata(image_pad_mask, img_shapes[0])

        timestep = timestep.to(hidden_states.dtype)
        if self.causal_condition:
            timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
            modulation_mask: Tensor | None = target_token_mask
        else:
            modulation_mask = None
        temb = self.time_text_embed(timestep, hidden_states)
        modulation = self.modulation(temb)

        joint_key_valid = None
        if encoder_hidden_states_mask is not None:
            joint_key_valid = torch.ones(
                batch_size, image_pad_mask.shape[0], dtype=torch.bool, device=hidden_states.device
            )
            text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
            vlm_text_positions = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
            joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[
                :, vlm_text_positions
            ]
        prefix_len = int((~target_token_mask).sum())
        if kv_cache_mode == "cached":
            joint_hidden_states = joint_hidden_states[:, prefix_len:]
            rotary_emb = rotary_emb[prefix_len:]
            if modulation_mask is not None:
                modulation_mask = modulation_mask[prefix_len:]
            attention_mask = None if joint_key_valid is None else joint_key_valid[:, None, None, :]
            cache_write_slice = None
            block_segments = None
            block_key_valid = None
        else:
            attention_mask = None
            block_segments = _qwenimage21_prefix_segments(image_ids, prefix_len)
            cache_write_slice = slice(0, prefix_len) if kv_cache_mode == "extract" else None
            block_key_valid = joint_key_valid

        for index, block in enumerate(self.transformer_blocks):
            layer_cache = kv_cache.get_layer(index) if kv_cache is not None else None
            joint_hidden_states = block(
                joint_hidden_states,
                modulation,
                rotary_emb,
                modulation_mask,
                layer_cache,
                kv_cache_mode,
                cache_write_slice,
                block_segments,
                block_key_valid,
                attention_mask,
            )
        joint_hidden_states = self.norm_out(joint_hidden_states, temb, modulation_mask)
        return self.proj_out(joint_hidden_states)
