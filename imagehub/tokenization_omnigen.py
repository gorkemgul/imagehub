"""Native tokenizer for OmniGen: Llama-style BPE with byte fallback and ``rstrip`` added tokens.

Differences from :class:`~imagehub.tokenization_gemma.GemmaTokenizer`: the first piece of a text
gets a prefix ``▁`` (unless it already starts with a space), later pieces after an added token do
not, and an added token flagged ``rstrip`` swallows the whitespace that follows it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from imagehub.tokenization import _WHITESPACE
from imagehub.tokenization_gemma import _SPACE, GemmaTokenizer


class OmniGenTokenizer(GemmaTokenizer):
    def __init__(self, directory: str | os.PathLike[str]) -> None:
        super().__init__(directory)
        data = json.loads((Path(directory) / "tokenizer.json").read_text(encoding="utf-8"))
        self._rstrip = {
            token["content"] for token in data.get("added_tokens", []) if token.get("rstrip")
        }

    def _piece_ids(self, text: str, first: bool) -> list[int]:
        if not text:
            return []
        text = text.replace(" ", _SPACE)
        if first and not text.startswith(_SPACE):
            text = _SPACE + text
        return self._segment_ids(text)

    def _content_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        start = 0
        if self._added_pattern is not None:
            for match in self._added_pattern.finditer(text):
                if match.start() < start:
                    continue
                ids.extend(self._piece_ids(text[start : match.start()], start == 0))
                ids.append(self._added[match.group()])
                start = match.end()
                if match.group() in self._rstrip:
                    while start < len(text) and text[start] in _WHITESPACE:
                        start += 1
        ids.extend(self._piece_ids(text[start:], start == 0))
        return ids
