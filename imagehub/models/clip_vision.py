"""CLIP vision transformer with projection and published transformers tensor names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

CLIP_VISION_DEFAULTS: dict[str, Any] = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_channels": 3,
    "image_size": 224,
    "patch_size": 32,
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
    "attention_dropout": 0.0,
    "projection_dim": 512,
}


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = config["hidden_size"]
        patch = config["patch_size"]
        positions = (config["image_size"] // patch) ** 2 + 1
        self.class_embedding = nn.Parameter(torch.empty(dim))
        self.patch_embedding = nn.Conv2d(
            config["num_channels"], dim, kernel_size=patch, stride=patch, bias=False
        )
        self.position_embedding = nn.Embedding(positions, dim)

    def forward(self, pixel_values: Tensor) -> Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_ids = torch.arange(embeddings.shape[1], device=embeddings.device)[None]
        return embeddings + self.position_embedding(position_ids)


class CLIPVisionAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.heads = config["num_attention_heads"]
        dim = config["hidden_size"]
        self.head_dim = dim // self.heads
        self.scale = self.head_dim**-0.5
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.heads, self.head_dim)
        queries = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            queries, keys, values, dropout_p=0.0, scale=self.scale, is_causal=False
        )
        output = output.transpose(1, 2).contiguous()
        return self.out_proj(output.reshape(*input_shape, -1).contiguous())


class CLIPVisionMLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.activation = config["hidden_act"]
        self.fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"])

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        if self.activation == "gelu":
            hidden_states = F.gelu(hidden_states)
        else:
            hidden_states = hidden_states * torch.sigmoid(1.702 * hidden_states)
        return self.fc2(hidden_states)


class CLIPVisionEncoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = config["hidden_size"]
        eps = config["layer_norm_eps"]
        self.self_attn = CLIPVisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(dim, eps=eps)
        self.mlp = CLIPVisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(dim, eps=eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.layer_norm2(hidden_states))
        return residual + hidden_states


class CLIPVisionEncoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            CLIPVisionEncoderLayer(config) for _ in range(config["num_hidden_layers"])
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        dim = config["hidden_size"]
        eps = config["layer_norm_eps"]
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(dim, eps=eps)
        self.encoder = CLIPVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(dim, eps=eps)

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        hidden_states = self.pre_layrnorm(self.embeddings(pixel_values))
        last_hidden_state = self.encoder(hidden_states)
        pooled_output = self.post_layernorm(last_hidden_state[:, 0, :])
        return last_hidden_state, pooled_output


class CLIPVisionModelWithProjection(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(CLIP_VISION_DEFAULTS | dict(config))
        if self.config["hidden_act"] not in ("quick_gelu", "gelu"):
            raise ValueError(f"Unsupported CLIP activation {self.config['hidden_act']!r}")
        self.vision_model = CLIPVisionTransformer(self.config)
        self.visual_projection = nn.Linear(
            self.config["hidden_size"], self.config["projection_dim"], bias=False
        )

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        state = dict(state)
        state.pop("vision_model.embeddings.position_ids", None)
        return state

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        last_hidden_state, pooled_output = self.vision_model(pixel_values)
        return self.visual_projection(pooled_output), last_hidden_state
