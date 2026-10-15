"""HiDream-O1 pixel-generation model components."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from imagehub.checkpoints import NativeModel
from imagehub.models.qwen3_vl_text import Qwen3VLTextModel


class HiDreamO1PatchEmbed(nn.Module):
    """Two-stage projection of flattened RGB pixel patches."""

    def __init__(self, patch_size: int, hidden_size: int, in_channels: int = 3) -> None:
        super().__init__()
        bottleneck = hidden_size // 4
        self.proj1 = nn.Linear(patch_size * patch_size * in_channels, bottleneck, bias=False)
        self.proj2 = nn.Linear(bottleneck, hidden_size, bias=True)

    def forward(self, patches: Tensor) -> Tensor:
        return self.proj2(self.proj1(patches))


class HiDreamO1TimestepEmbedder(nn.Module):
    """Sinusoidal flow timestep embedding followed by a two-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(timestep: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=timestep.device)
            / half
        )
        args = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat((args.cos(), args.sin()), dim=-1)
        if dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding

    def forward(self, timestep: Tensor) -> Tensor:
        frequencies = self.timestep_embedding(timestep * 1000, self.frequency_embedding_size)
        return self.mlp(frequencies.to(dtype=next(self.mlp.parameters()).dtype))


class HiDreamO1FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.linear(hidden_states)


class _VisionAttentionLayout(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)


class _VisionMLPLayout(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size)


class _VisionBlockLayout(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = _VisionAttentionLayout(hidden_size)
        self.mlp = _VisionMLPLayout(hidden_size, intermediate_size)


class _VisionMergerLayout(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, *, postshuffle_norm: bool) -> None:
        super().__init__()
        merged_size = hidden_size * 4
        self.norm = nn.LayerNorm(merged_size if postshuffle_norm else hidden_size)
        self.linear_fc1 = nn.Linear(merged_size, merged_size)
        self.linear_fc2 = nn.Linear(merged_size, output_size)


class HiDreamO1VisionLayout(nn.Module):
    """Published Qwen3-VL vision tensor layout used for strict checkpoint loading.

    The text-to-image path does not execute this encoder. Its inference implementation is reserved
    for the optional image-conditioning path.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(
            config.get("in_channels", 3),
            hidden_size,
            kernel_size=(config["temporal_patch_size"], config["patch_size"], config["patch_size"]),
        )
        self.pos_embed = nn.Embedding(config["num_position_embeddings"], hidden_size)
        self.blocks = nn.ModuleList(
            _VisionBlockLayout(hidden_size, config["intermediate_size"])
            for _ in range(config["depth"])
        )
        output_size = config["out_hidden_size"]
        self.merger = _VisionMergerLayout(hidden_size, output_size, postshuffle_norm=False)
        self.deepstack_merger_list = nn.ModuleList(
            _VisionMergerLayout(hidden_size, output_size, postshuffle_norm=True)
            for _ in config["deepstack_visual_indexes"]
        )


class HiDreamO1Core(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        text_config = config["text_config"]
        hidden_size = text_config["hidden_size"]
        patch_size = config.get("pixel_patch_size", 32)
        self.visual = HiDreamO1VisionLayout(config["vision_config"])
        self.language_model = Qwen3VLTextModel(text_config)
        self.t_embedder1 = HiDreamO1TimestepEmbedder(hidden_size)
        self.x_embedder = HiDreamO1PatchEmbed(patch_size, hidden_size)
        self.final_layer2 = HiDreamO1FinalLayer(hidden_size, patch_size)


class HiDreamO1Model(NativeModel):
    """Qwen3-VL backbone extended with direct pixel prediction."""

    clone_shards = False

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.model = HiDreamO1Core(config)
        text_config = config["text_config"]
        self.lm_head = nn.Linear(text_config["hidden_size"], text_config["vocab_size"], bias=False)
        self.tms_token_id = config.get("tms_token_id", 151673)

    def init_non_persistent_buffers(self) -> None:
        self.model.language_model.init_non_persistent_buffers()

    def _decode(
        self, inputs_embeds: Tensor, position_ids: Tensor, attention_mask: Tensor
    ) -> Tensor:
        language_model = self.model.language_model
        position_embeddings = language_model.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer in language_model.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask)
        return language_model.norm(hidden_states)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        vinputs: Tensor,
        timestep: Tensor,
        token_types: Tensor,
    ) -> Tensor:
        """Predict clean RGB patches for a text-to-image denoising step."""
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        timestep_embedding = self.model.t_embedder1(timestep.to(inputs_embeds.device))
        timestep_mask = (input_ids == self.tms_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = torch.where(
            timestep_mask, timestep_embedding.unsqueeze(1).expand_as(inputs_embeds), inputs_embeds
        )
        pixel_embeds = self.model.x_embedder(vinputs.to(inputs_embeds.device)).to(
            inputs_embeds.dtype
        )
        inputs_embeds = torch.cat((inputs_embeds, pixel_embeds), dim=1)

        batch_size, sequence_length = inputs_embeds.shape[:2]
        token_types = token_types.to(inputs_embeds.device)
        if token_types.ndim == 1:
            token_types = token_types.unsqueeze(0)
        if token_types.shape[0] == 1 and batch_size > 1:
            token_types = token_types.expand(batch_size, -1)
        minimum = torch.finfo(inputs_embeds.dtype).min
        causal = torch.full(
            (sequence_length, sequence_length),
            minimum,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        ).triu(diagonal=1)
        masks = []
        for index in range(batch_size):
            mask = causal.clone()
            mask[token_types[index].bool(), :] = 0
            masks.append(mask)
        attention_mask = torch.stack(masks).unsqueeze(1)
        hidden_states = self._decode(inputs_embeds, position_ids, attention_mask)
        return self.model.final_layer2(hidden_states)
