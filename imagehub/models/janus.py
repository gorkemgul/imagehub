"""Janus-Pro image-generation model and published-checkpoint key conversion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from imagehub.checkpoints import NativeModel
from imagehub.models.janus_vq import JanusVQVAE, JanusVQVAEAlignerMLP, JanusVQVAEHead
from imagehub.models.llama import LlamaModel

_EXCLUDED_PREFIXES = (
    "model.vision_model.",
    "model.aligner.",
    "lm_head.",
    "model.vqmodel.encoder.",
    "model.vqmodel.quant_conv.",
)


class JanusModel(NativeModel):
    """The generation-only subset of a transformers-format Janus model."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        text_config = self.config.get("text_config", self.config)
        vq_config = dict(self.config["vq_config"])
        vision_config = self.config.get("vision_config")
        if vision_config is not None:
            vq_config["num_patches"] = vision_config["image_size"] // vision_config["patch_size"]
        self.language_model = LlamaModel(text_config)
        self.vqmodel = JanusVQVAE(vq_config)
        self.generation_embeddings = nn.Embedding(
            vq_config["num_embeddings"], vq_config["embed_dim"]
        )
        self.generation_aligner = JanusVQVAEAlignerMLP(vq_config)
        self.generation_head = JanusVQVAEHead(vq_config)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        prepared: dict[str, Tensor] = {}
        for name, tensor in state.items():
            if name.startswith(_EXCLUDED_PREFIXES):
                continue
            prepared[name.removeprefix("model.")] = tensor
        return prepared

    def init_non_persistent_buffers(self) -> None:
        self.language_model.init_non_persistent_buffers()

    def prepare_embeddings_for_image_generation(self, image_tokens: Tensor) -> Tensor:
        return self.generation_aligner(self.generation_embeddings(image_tokens))

    def decode_image_tokens(self, image_tokens: Tensor) -> Tensor:
        return self.vqmodel.decode(image_tokens).permute(0, 2, 3, 1)
