"""BERT encoder used as the first HunyuanDiT text encoder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

BERT_DEFAULTS: dict[str, Any] = {
    "vocab_size": 30522,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "layer_norm_eps": 1e-12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
}


class BertEmbeddings(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config["vocab_size"], config["hidden_size"], padding_idx=config["pad_token_id"]
        )
        self.position_embeddings = nn.Embedding(
            config["max_position_embeddings"], config["hidden_size"]
        )
        self.token_type_embeddings = nn.Embedding(config["type_vocab_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, sequence_length = input_ids.shape
        position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        token_type_ids = torch.zeros(
            (batch_size, sequence_length), dtype=torch.long, device=input_ids.device
        )
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)


class BertSelfAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        heads = config["num_attention_heads"]
        if hidden_size % heads:
            raise ValueError("BERT hidden_size must be divisible by num_attention_heads")
        self.num_attention_heads = heads
        self.attention_head_size = hidden_size // heads
        self.all_head_size = heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)
        query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        attention_output = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=self.scaling,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        return attention_output.reshape(*input_shape, -1).contiguous()


class BertSelfOutput(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class BertAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
        self_output = self.self(hidden_states, attention_mask)
        return self.output(self_output, hidden_states)


class BertIntermediate(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["intermediate_size"])

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.gelu(self.dense(hidden_states))


class BertOutput(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.dense = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class BertLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class BertEncoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.layer = nn.ModuleList(BertLayer(config) for _ in range(config["num_hidden_layers"]))

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.activation(self.dense(hidden_states[:, 0]))


class BertModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(BERT_DEFAULTS | dict(config))
        if self.config["hidden_act"] != "gelu":
            raise ValueError(f"Unsupported BERT activation {self.config['hidden_act']!r}")
        if self.config["position_embedding_type"] != "absolute":
            raise ValueError(
                f"Unsupported BERT position_embedding_type "
                f"{self.config['position_embedding_type']!r}"
            )
        self.embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)
        self.pooler = BertPooler(self.config)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden_states = self.embeddings(input_ids)
        extended_mask = None
        if attention_mask is not None:
            mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
            if not bool(mask.all()):
                sequence_length = input_ids.shape[1]
                extended_mask = mask[:, None, None, :].expand(-1, 1, sequence_length, -1)
        return self.encoder(hidden_states, extended_mask)
