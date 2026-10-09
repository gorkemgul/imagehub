"""ChatGLM3-6B text encoder used by Kolors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

CHATGLM_DEFAULTS: dict[str, Any] = {
    "num_layers": 28,
    "padded_vocab_size": 65024,
    "hidden_size": 4096,
    "ffn_hidden_size": 13696,
    "kv_channels": 128,
    "num_attention_heads": 32,
    "seq_length": 2048,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
    "layernorm_epsilon": 1e-5,
    "rmsnorm": True,
    "apply_residual_connection_post_layernorm": False,
    "post_layer_norm": True,
    "add_bias_linear": False,
    "add_qkv_bias": False,
    "multi_query_attention": False,
    "multi_query_group_num": 1,
    "fp32_residual_connection": False,
    "pre_seq_len": None,
}


class ChatGLMRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


def _norm(config: Mapping[str, Any]) -> nn.Module:
    if config["rmsnorm"]:
        return ChatGLMRMSNorm(config["hidden_size"], config["layernorm_epsilon"])
    return nn.LayerNorm(config["hidden_size"], eps=config["layernorm_epsilon"])


class ChatGLMEmbedding(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config["padded_vocab_size"], config["hidden_size"])
        self.fp32_residual_connection = config["fp32_residual_connection"]

    def forward(self, input_ids: Tensor) -> Tensor:
        embeddings = self.word_embeddings(input_ids).transpose(0, 1).contiguous()
        return embeddings.float() if self.fp32_residual_connection else embeddings


class ChatGLMRotaryEmbedding(nn.Module):
    """Rotary cache over half the head dimension, as ``[position, dim / 2, (cos, sin)]``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("inv_freq", self._frequencies(dim))

    @staticmethod
    def _frequencies(dim: int) -> Tensor:
        return 1.0 / (10000 ** (torch.arange(0, dim, 2).to(dtype=None) / dim))

    def forward(self, max_seq_len: int) -> Tensor:
        inv_freq = cast(Tensor, self.inv_freq)
        dtype, device = inv_freq.dtype, inv_freq.device
        theta = 1.0 / (
            10000 ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device) / self.dim)
        )
        seq_idx = torch.arange(max_seq_len, dtype=torch.float, device=device)
        idx_theta = torch.outer(seq_idx, theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        if dtype in (torch.float16, torch.bfloat16):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache


def _apply_rotary_pos_emb(x: Tensor, rope_cache: Tensor) -> Tensor:
    # x: [sq, b, np, hn]
    sq, np_ = x.size(0), x.size(2)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np_, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


class ChatGLMSelfAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.heads = config["num_attention_heads"]
        self.head_dim = config["kv_channels"]
        self.projection_size = self.head_dim * self.heads
        self.multi_query = config["multi_query_attention"]
        self.groups = config["multi_query_group_num"]
        qkv_size = 3 * self.projection_size
        if self.multi_query:
            qkv_size = self.projection_size + 2 * self.head_dim * self.groups
        self.query_key_value = nn.Linear(
            config["hidden_size"],
            qkv_size,
            bias=config["add_bias_linear"] or config["add_qkv_bias"],
        )
        self.dense = nn.Linear(
            self.projection_size, config["hidden_size"], bias=config["add_bias_linear"]
        )

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None, rotary_pos_emb: Tensor
    ) -> Tensor:
        mixed = self.query_key_value(hidden_states)
        if self.multi_query:
            query, key, value = mixed.split(
                [
                    self.heads * self.head_dim,
                    self.groups * self.head_dim,
                    self.groups * self.head_dim,
                ],
                dim=-1,
            )
            query = query.view(query.size()[:-1] + (self.heads, self.head_dim))
            key = key.view(key.size()[:-1] + (self.groups, self.head_dim))
            value = value.view(value.size()[:-1] + (self.groups, self.head_dim))
        else:
            mixed = mixed.view(mixed.size()[:-1] + (self.heads, 3 * self.head_dim))
            query, key, value = torch.split(mixed, mixed.size(-1) // 3, dim=-1)
        query = _apply_rotary_pos_emb(query, rotary_pos_emb)
        key = _apply_rotary_pos_emb(key, rotary_pos_emb)
        if self.multi_query:
            repeats = self.heads // self.groups
            key = key.unsqueeze(-2).expand(-1, -1, -1, repeats, -1)
            key = key.contiguous().view(key.size()[:2] + (self.heads, self.head_dim))
            value = value.unsqueeze(-2).expand(-1, -1, -1, repeats, -1)
            value = value.contiguous().view(value.size()[:2] + (self.heads, self.head_dim))

        query, key, value = (tensor.permute(1, 2, 0, 3) for tensor in (query, key, value))
        if attention_mask is None:
            context = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        else:
            context = F.scaled_dot_product_attention(query, key, value, ~attention_mask)
        context = context.permute(2, 0, 1, 3)
        context = context.reshape(*context.size()[:-2], self.projection_size)
        return self.dense(context)


class ChatGLMMLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        bias = config["add_bias_linear"]
        self.dense_h_to_4h = nn.Linear(
            config["hidden_size"], config["ffn_hidden_size"] * 2, bias=bias
        )
        self.dense_4h_to_h = nn.Linear(config["ffn_hidden_size"], config["hidden_size"], bias=bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, up = torch.chunk(self.dense_h_to_4h(hidden_states), 2, dim=-1)
        return self.dense_4h_to_h(F.silu(gate) * up)


class ChatGLMBlock(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.post_layernorm_residual = config["apply_residual_connection_post_layernorm"]
        self.input_layernorm = _norm(config)
        self.self_attention = ChatGLMSelfAttention(config)
        self.post_attention_layernorm = _norm(config)
        self.mlp = ChatGLMMLP(config)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None, rotary_pos_emb: Tensor
    ) -> Tensor:
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output = self.self_attention(layernorm_output, attention_mask, rotary_pos_emb)
        residual = layernorm_output if self.post_layernorm_residual else hidden_states
        layernorm_input = residual + attention_output
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        mlp_output = self.mlp(layernorm_output)
        residual = layernorm_output if self.post_layernorm_residual else layernorm_input
        return residual + mlp_output


class ChatGLMTransformer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(ChatGLMBlock(config) for _ in range(config["num_layers"]))
        if config["post_layer_norm"]:
            self.final_layernorm = _norm(config)


class ChatGLMModel(NativeModel):
    """ChatGLM3 encoder returning the two states Kolors consumes.

    ``forward`` returns ``(prompt_embeds, pooled)``: the input of the last block as
    ``[batch, sequence, hidden]`` and the last position of the last block's output as
    ``[batch, hidden]``, both before the final layer norm.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(CHATGLM_DEFAULTS | dict(config))
        if self.config["pre_seq_len"] is not None:
            raise ValueError("Unsupported ChatGLM prefix encoder")
        self.embedding = ChatGLMEmbedding(self.config)
        self.rotary_pos_emb = ChatGLMRotaryEmbedding(self.config["kv_channels"] // 2)
        self.encoder = ChatGLMTransformer(self.config)
        self.output_layer = nn.Linear(
            self.config["hidden_size"], self.config["padded_vocab_size"], bias=False
        )

    def init_non_persistent_buffers(self) -> None:
        # inv_freq is stored in the checkpoint; the cache itself is derived on demand.
        return

    @staticmethod
    def _full_attention_mask(batch_size: int, seq_length: int, padding_mask: Tensor) -> Tensor:
        padding = padding_mask.long()
        mask = torch.ones(batch_size, seq_length, seq_length, device=padding.device)
        mask.tril_()
        mask = mask * padding.unsqueeze(1)
        mask -= padding.unsqueeze(-1) - 1
        return (mask < 0.5).bool().unsqueeze_(1)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, seq_length = input_ids.shape
        hidden_states = self.embedding(input_ids)
        full_attention_mask = None
        if attention_mask is not None and not attention_mask.all():
            full_attention_mask = self._full_attention_mask(batch_size, seq_length, attention_mask)

        rotary_pos_emb = self.rotary_pos_emb(self.config["seq_length"])
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        layers = self.encoder.layers
        penultimate = hidden_states
        for index, layer in enumerate(layers):
            if index == len(layers) - 1:
                penultimate = hidden_states
            hidden_states = layer(hidden_states, full_attention_mask, rotary_pos_emb)
        return penultimate.permute(1, 0, 2).clone(), hidden_states[-1, :, :].clone()
