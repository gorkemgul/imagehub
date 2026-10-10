"""GLM-Image prompt tokenization for the autoregressive stage and the glyph encoder."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

from imagehub.tokenization_glm import GlmTokenizer, _split_added

_PATCH_FACTOR = 32


class GlmImageTokenizer(GlmTokenizer):
    """The GLM-4 byte-level BPE with GLM-Image's added tokens and no prefix tokens."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        super().__init__(directory)
        config = json.loads((Path(directory) / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.grid_bos_token = config["grid_bos_token"]
        self.grid_eos_token = config["grid_eos_token"]
        bos_token = config["bos_token"]
        self.bos_token = bos_token["content"] if isinstance(bos_token, dict) else bos_token
        self.prefix_ids = []
        # The configuration lists added tokens that tokenizer.json does not (image, think, box).
        for index, token in config.get("added_tokens_decoder", {}).items():
            self._added.setdefault(token["content"], int(index))

    def tokenize(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece, added_id in _split_added(text, self._added):
            if added_id is not None:
                ids.append(added_id)
            else:
                ids.extend(self._ordinary_ids(piece))
        return ids


def target_shape(height: int, width: int) -> tuple[int, int, int, int]:
    """Token grid of the full-size image and of the low-resolution preview that precedes it."""
    height = (height // _PATCH_FACTOR) * _PATCH_FACTOR
    width = (width // _PATCH_FACTOR) * _PATCH_FACTOR
    token_h = height // _PATCH_FACTOR
    token_w = width // _PATCH_FACTOR
    ratio = token_h / token_w
    preview_h = int(math.sqrt(ratio) * (_PATCH_FACTOR // 2))
    preview_w = int(math.sqrt(1 / ratio) * (_PATCH_FACTOR // 2))
    return token_h, token_w, preview_h, preview_w


def build_prior_prompt(
    tokenizer: GlmImageTokenizer, prompt: str, height: int, width: int
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Return the AR model's input ids (1, length) and the (height, width) grids of the target."""
    token_h, token_w, preview_h, preview_w = target_shape(height, width)
    bos, eos = tokenizer.grid_bos_token, tokenizer.grid_eos_token
    text = (
        f"{prompt}{bos}{token_h} {token_w}{eos}{bos}{preview_h} {preview_w}{eos}"
        f"{tokenizer.bos_token}"
    )
    ids = torch.tensor([tokenizer.tokenize(text)], dtype=torch.long)
    return ids, [(token_h, token_w), (preview_h, preview_w)]
