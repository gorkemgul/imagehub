"""Multilingual CLIP text encoder used by Kandinsky 2.1.

The published checkpoint wraps ``transformers.XLMRobertaModel`` and adds a linear
projection from ``transformerDimensions`` to ``numDims``.  The output tuple is
``(projected_text_embeds, last_hidden_state)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from imagehub.checkpoints import NativeModel
from imagehub.models.xlm_roberta import XLMRobertaModel

MULTILINGUAL_CLIP_DEFAULTS: dict[str, Any] = {
    "transformerDimensions": 1024,
    "numDims": 768,
}


class MultilingualCLIP(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(MULTILINGUAL_CLIP_DEFAULTS | dict(config))
        self.transformer = XLMRobertaModel(self.config)
        self.LinearTransformation = nn.Linear(
            in_features=self.config["transformerDimensions"],
            out_features=self.config["numDims"],
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        embs = self.transformer(input_ids, attention_mask=attention_mask)
        mask = attention_mask.to(embs.dtype).unsqueeze(2)
        pooled = (embs * mask).sum(dim=1) / attention_mask.sum(dim=1).to(embs.dtype)[:, None]
        return self.LinearTransformation(pooled), embs

    def init_non_persistent_buffers(self) -> None:
        self.transformer.init_non_persistent_buffers()

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        state = dict(state)
        state.pop("transformer.embeddings.position_ids", None)
        return state
