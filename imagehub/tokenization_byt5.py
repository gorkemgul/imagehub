"""Native ByT5 tokenizer: UTF-8 bytes shifted past the special ids, with added tokens."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ByT5Tokenizer:
    """Byte-level tokenizer reading ``tokenizer_config.json`` from a snapshot directory."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        config = json.loads((Path(directory) / "tokenizer_config.json").read_text(encoding="utf-8"))
        decoder: dict[str, dict[str, Any]] = config["added_tokens_decoder"]
        self.pad_token_id = self._token_id(config.get("pad_token", "<pad>"), decoder, 0)
        self.eos_token_id = self._token_id(config.get("eos_token", "</s>"), decoder, 1)
        self.unk_token_id = self._token_id(config.get("unk_token", "<unk>"), decoder, 2)
        self.offset = 3
        self._added: dict[str, tuple[int, bool, bool]] = {
            entry["content"]: (
                int(index),
                bool(entry.get("lstrip", False)),
                bool(entry.get("rstrip", False)),
            )
            for index, entry in decoder.items()
        }

    @staticmethod
    def _token_id(token: Any, decoder: dict[str, dict[str, Any]], default: int) -> int:
        content = token["content"] if isinstance(token, dict) else token
        for index, entry in decoder.items():
            if entry["content"] == content:
                return int(index)
        return default

    def _split(self, text: str) -> list[tuple[str, int | None]]:
        pieces: list[tuple[str, int | None]] = []
        position = 0
        pending_rstrip = False
        while True:
            best: tuple[int, int, str] | None = None
            for token in self._added:
                found = text.find(token, position)
                if found >= 0 and (
                    best is None or found < best[0] or (found == best[0] and len(token) > best[1])
                ):
                    best = (found, len(token), token)
            segment_end = len(text) if best is None else best[0]
            segment = text[position:segment_end]
            if pending_rstrip:
                segment = segment.lstrip()
            if best is None:
                if segment:
                    pieces.append((segment, None))
                return pieces
            index, lstrip, rstrip = self._added[best[2]]
            if lstrip:
                segment = segment.rstrip()
            if segment:
                pieces.append((segment, None))
            pieces.append((best[2], index))
            pending_rstrip = rstrip
            position = best[0] + best[1]

    def tokenize(self, text: str, max_length: int | None = None) -> list[int]:
        """Return ids ending in ``</s>``, truncated to ``max_length`` with ``</s>`` kept."""
        ids: list[int] = []
        for piece, added in self._split(text):
            if added is not None:
                ids.append(added)
            else:
                ids.extend(byte + self.offset for byte in piece.encode("utf-8"))
        if max_length is not None and len(ids) > max_length - 1:
            ids = ids[: max_length - 1]
        if not ids or ids[-1] != self.eos_token_id:
            ids.append(self.eos_token_id)
        return ids
