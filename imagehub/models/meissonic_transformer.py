"""Native Meissonic masked-token transformer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from imagehub.checkpoints import NativeModel
from imagehub.models.flux_transformer import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    FluxPosEmbed,
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
)
from imagehub.models.layers import timestep_embedding
from imagehub.models.uvit_2d import (
    CondEmbed,
    ConvEmbed,
    ConvMlmLayer,
    Downsample,
    RMSNorm,
    Upsample,
)

MEISSONIC_DEFAULTS: dict[str, Any] = {
    "patch_size": 1,
    "in_channels": 64,
    "num_layers": 19,
    "num_single_layers": 38,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "guidance_embeds": False,
    "axes_dims_rope": (16, 56, 56),
    "vocab_size": 8256,
    "codebook_size": 8192,
    "downsample": False,
    "upsample": False,
}


class SimpleUVitBlock(nn.Module):
    def __init__(self, channels: int, *, downsample: bool, upsample: bool) -> None:
        super().__init__()
        self.downsample = Downsample(channels, 1e-6, True, False) if downsample else None
        self.upsample = Upsample(channels, 1e-6, True, False) if upsample else None

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)
        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)
        return hidden_states


class MeissonicTransformer2DModel(NativeModel):
    """Meissonic's convolutional masked-token stem around FLUX-style transformer blocks."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        unknown = sorted(
            key for key in config if not key.startswith("_") and key not in MEISSONIC_DEFAULTS
        )
        if unknown:
            raise ValueError(f"Unsupported Meissonic transformer setting {unknown[0]}")
        cfg = MEISSONIC_DEFAULTS | {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        if sum(cfg["axes_dims_rope"]) != cfg["attention_head_dim"]:
            raise ValueError("Unsupported Meissonic transformer rotary dimensions")
        super().__init__(cfg)
        inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        self.inner_dim = inner_dim
        self.pos_embed = FluxPosEmbed(10000, list(cfg["axes_dims_rope"]))
        embedding_class = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if cfg["guidance_embeds"]
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = embedding_class(inner_dim, cfg["pooled_projection_dim"])
        self.context_embedder = nn.Linear(cfg["joint_attention_dim"], inner_dim)
        self.transformer_blocks = nn.ModuleList(
            FluxTransformerBlock(inner_dim, cfg["num_attention_heads"], cfg["attention_head_dim"])
            for _ in range(cfg["num_layers"])
        )
        self.single_transformer_blocks = nn.ModuleList(
            FluxSingleTransformerBlock(
                inner_dim, cfg["num_attention_heads"], cfg["attention_head_dim"]
            )
            for _ in range(cfg["num_single_layers"])
        )
        self.embed = ConvEmbed(inner_dim, inner_dim, cfg["vocab_size"], True, 1e-6, False)
        self.mlm_layer = ConvMlmLayer(inner_dim, inner_dim, False, True, 1e-6, cfg["codebook_size"])
        self.cond_embed = CondEmbed(1280 + cfg["pooled_projection_dim"], inner_dim, False)
        self.encoder_proj_layer_norm = RMSNorm(inner_dim, 1e-6, True)
        self.project_to_hidden_norm = RMSNorm(inner_dim, 1e-6, True)
        self.project_to_hidden = nn.Linear(inner_dim, inner_dim, bias=False)
        self.project_from_hidden_norm = RMSNorm(inner_dim, 1e-6, True)
        self.project_from_hidden = nn.Linear(inner_dim, inner_dim, bias=False)
        self.down_block = SimpleUVitBlock(inner_dim, downsample=cfg["downsample"], upsample=False)
        self.up_block = SimpleUVitBlock(inner_dim, downsample=False, upsample=cfg["upsample"])

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        pooled_projections: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        micro_conds: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        """Return codebook logits with shape `(batch, codebook, height, width)`."""
        micro = timestep_embedding(
            micro_conds.flatten(), 256, flip_sin_to_cos=True, shift=0
        ).reshape(hidden_states.shape[0], -1)
        pooled = torch.cat([pooled_projections, micro], dim=1)
        pooled = self.cond_embed(pooled.to(dtype=self.context_embedder.weight.dtype))

        hidden_states = self.down_block(self.embed(hidden_states))
        batch, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        hidden_states = self.project_to_hidden(self.project_to_hidden_norm(hidden_states))
        encoder_hidden_states = self.encoder_proj_layer_norm(
            self.context_embedder(encoder_hidden_states)
        )
        pooled = pooled.to(encoder_hidden_states.dtype)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is None:
            temb = self.time_text_embed(timestep, pooled)
        else:
            temb = self.time_text_embed(timestep, guidance.to(hidden_states.dtype) * 1000, pooled)
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        rotary_emb = self.pos_embed(torch.cat((txt_ids, img_ids), dim=0))
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states, temb, rotary_emb
            )
        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states, temb, rotary_emb
            )
        hidden_states = self.project_from_hidden(self.project_from_hidden_norm(hidden_states))
        hidden_states = hidden_states.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return self.mlm_layer(self.up_block(hidden_states))
