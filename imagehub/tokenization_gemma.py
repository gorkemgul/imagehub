"""Native SentencePiece-style BPE tokenizer for Gemma ``tokenizer.json`` files."""

from __future__ import annotations

import heapq
import json
import os
import re
from pathlib import Path
from typing import Literal

import torch

_SPACE = "▁"


def _content(value: object) -> str | None:
    if isinstance(value, dict):
        value = value.get("content")
    return value if isinstance(value, str) else None


class GemmaTokenizer:
    """BPE with byte fallback, a space-to-``▁`` normalizer and no pre-tokenizer splitting."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        path = Path(directory)
        data = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
        model = data["model"]
        self.encoder: dict[str, int] = model["vocab"]
        self.ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(model["merges"]):
            pair = tuple(merge) if isinstance(merge, list) else tuple(merge.split(" ", 1))
            self.ranks.setdefault((pair[0], pair[1]), rank)
        self._byte_fallback = bool(model.get("byte_fallback", False))
        self._fuse_unk = bool(model.get("fuse_unk", False))
        self._unk_token = model.get("unk_token")

        self._added: dict[str, int] = {}
        for token in data.get("added_tokens", []):
            self._added[token["content"]] = token["id"]
        self._added_pattern = (
            re.compile("|".join(re.escape(t) for t in sorted(self._added, key=lambda t: -len(t))))
            if self._added
            else None
        )

        config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.add_bos_token = bool(config.get("add_bos_token", True))
        self.add_eos_token = bool(config.get("add_eos_token", False))
        self.bos_id = self._token_id(_content(config.get("bos_token")) or "<bos>")
        self.eos_id = self._token_id(_content(config.get("eos_token")) or "<eos>")
        self.pad_id = self._token_id(_content(config.get("pad_token")) or "<pad>")

    def _token_id(self, token: str) -> int:
        return self._added[token] if token in self._added else self.encoder[token]

    def _initial_symbols(self, text: str) -> list[str]:
        symbols: list[str] = []
        previous_unk = False
        for character in text:
            if character in self.encoder:
                symbols.append(character)
                previous_unk = False
                continue
            if self._byte_fallback:
                fallback = [f"<0x{byte:02X}>" for byte in character.encode("utf-8")]
                if all(token in self.encoder for token in fallback):
                    symbols.extend(fallback)
                    previous_unk = False
                    continue
            if self._unk_token is None:
                continue
            if self._fuse_unk and previous_unk:
                symbols[-1] += character
            else:
                symbols.append(character)
            previous_unk = True
        return symbols

    def _bpe(self, text: str) -> list[str]:
        """Merge symbols by rank, leftmost pair first among equal ranks."""
        words = self._initial_symbols(text)
        count = len(words)
        if count < 2:
            return words
        previous = list(range(-1, count - 1))
        following = list(range(1, count + 1))
        following[-1] = -1
        alive = [True] * count
        heap: list[tuple[int, int]] = []

        def push(position: int) -> None:
            neighbour = following[position]
            if neighbour < 0:
                return
            rank = self.ranks.get((words[position], words[neighbour]))
            if rank is not None:
                heapq.heappush(heap, (rank, position))

        for position in range(count - 1):
            push(position)
        while heap:
            rank, position = heapq.heappop(heap)
            neighbour = following[position]
            if not alive[position] or neighbour < 0:
                continue
            if self.ranks.get((words[position], words[neighbour])) != rank:
                continue
            words[position] += words[neighbour]
            alive[neighbour] = False
            following[position] = following[neighbour]
            if following[position] >= 0:
                previous[following[position]] = position
            if previous[position] >= 0:
                push(previous[position])
            push(position)
        return [word for word, keep in zip(words, alive, strict=True) if keep]

    def _segment_ids(self, text: str) -> list[int]:
        if not text:
            return []
        ids: list[int] = []
        for token in self._bpe(text.replace(" ", _SPACE)):
            if token in self.encoder:
                ids.append(self.encoder[token])
            elif self._unk_token is not None:
                ids.append(self.encoder[self._unk_token])
        return ids

    def _content_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        start = 0
        if self._added_pattern is not None:
            for match in self._added_pattern.finditer(text):
                ids.extend(self._segment_ids(text[start : match.start()]))
                ids.append(self._added[match.group()])
                start = match.end()
        ids.extend(self._segment_ids(text[start:]))
        return ids

    def tokenize(self, text: str, max_length: int | None = None) -> list[int]:
        """Return the ids of ``encode(text, add_special_tokens=True[, truncation])``."""
        ids = self._content_ids(text)
        if max_length is not None:
            room = max_length - int(self.add_bos_token) - int(self.add_eos_token)
            ids = ids[: max(room, 0)]
        if self.add_bos_token:
            ids = [self.bos_id, *ids]
        if self.add_eos_token:
            ids = [*ids, self.eos_id]
        return ids

    def __call__(
        self,
        texts: str | list[str],
        max_length: int = 300,
        padding_side: Literal["left", "right"] = "left",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Truncate and pad to ``max_length``; ``left`` is the reference default, the
        Sana pipeline switches the reference to ``right``."""
        if isinstance(texts, str):
            texts = [texts]
        rows: list[list[int]] = []
        masks: list[list[bool]] = []
        for text in texts:
            ids = self.tokenize(text, max_length)
            padding = max_length - len(ids)
            if padding_side == "left":
                rows.append([self.pad_id] * padding + ids)
                masks.append([False] * padding + [True] * len(ids))
            else:
                rows.append(ids + [self.pad_id] * padding)
                masks.append([True] * len(ids) + [False] * padding)
        return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)
