"""AuraFlow prompt tokenizer: the Llama-style BPE (byte fallback) read from ``tokenizer.json``.

It reuses the BPE of :class:`GemmaTokenizer` and differs in the way the reference builds the
backend: the first text segment (before any added token) gets a leading ``▁``, unknown
characters without byte fallback are dropped, and the special tokens added around the content
come from the ``post_processor`` of ``tokenizer.json`` (the tokenizer config flags are ignored
when that file exists).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import torch

from imagehub.tokenization_gemma import GemmaTokenizer

_SPACE = "▁"


def _template_ids(config: dict[str, Any] | None) -> tuple[list[int], list[int]]:
    """Ids before and after the sequence of a ``TemplateProcessing`` single template."""
    if config is None:
        return [], []
    if config.get("type") != "TemplateProcessing":
        raise ValueError(f"Unsupported tokenizer post-processor: {config.get('type')}")
    before: list[int] = []
    after: list[int] = []
    target = before
    for item in config["single"]:
        if "Sequence" in item:
            target = after
            continue
        target.extend(int(i) for i in config["special_tokens"][item["SpecialToken"]["id"]]["ids"])
    return before, after


class AuraFlowTokenizer(GemmaTokenizer):
    """Tokenizer of the ``LlamaTokenizer`` folder used by the AuraFlow text encoder."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        super().__init__(directory)
        data = json.loads((Path(directory) / "tokenizer.json").read_text(encoding="utf-8"))
        self._prefix_ids, self._suffix_ids = _template_ids(data.get("post_processor"))
        # The reference rebuilds the model with byte fallback, fused unknowns and no unk token.
        self._byte_fallback = True
        self._fuse_unk = True
        self._unk_token = None

    def _segment_ids(self, text: str, first: bool = False) -> list[int]:
        symbols = text.replace(" ", _SPACE)
        if first and symbols and not symbols.startswith(_SPACE):
            symbols = _SPACE + symbols
        return [self.encoder[token] for token in self._bpe(symbols) if token in self.encoder]

    def _content_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        start = 0
        if self._added_pattern is not None:
            for match in self._added_pattern.finditer(text):
                ids.extend(self._segment_ids(text[start : match.start()], first=start == 0))
                ids.append(self._added[match.group()])
                start = match.end()
        ids.extend(self._segment_ids(text[start:], first=start == 0))
        return ids

    def tokenize(self, text: str, max_length: int | None = None) -> list[int]:
        """Return the ids of ``encode(text[, truncation, max_length])``."""
        ids = self._content_ids(text)
        if max_length is not None:
            room = max_length - len(self._prefix_ids) - len(self._suffix_ids)
            ids = ids[: max(room, 0)]
        return [*self._prefix_ids, *ids, *self._suffix_ids]

    def __call__(
        self,
        texts: str | list[str],
        max_length: int = 256,
        padding_side: Literal["left", "right"] = "right",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(texts, max_length=max_length, padding_side=padding_side)
