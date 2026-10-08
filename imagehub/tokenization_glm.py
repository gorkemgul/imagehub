"""Native byte-level BPE tokenizer for GLM-4 tokenizer files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeAlias

import torch

from imagehub.tokenization import _WHITESPACE, _is_letter, _is_number, bytes_to_unicode

_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
_AddedPiece: TypeAlias = tuple[str, int | None]


def _split_added(text: str, tokens: dict[str, int]) -> list[_AddedPiece]:
    pieces: list[_AddedPiece] = []
    start = 0
    while start < len(text):
        matches = [
            (text.find(token, start), -len(token), token)
            for token in tokens
            if text.find(token, start) >= 0
        ]
        if not matches:
            pieces.append((text[start:], None))
            break
        position, _, token = min(matches)
        if position > start:
            pieces.append((text[start:position], None))
        pieces.append((token, tokens[token]))
        start = position + len(token)
    if not text:
        return [("", None)]
    return pieces


def _pretokenize(text: str) -> list[str]:
    pieces: list[str] = []
    index = 0
    while index < len(text):
        contraction = next(
            (
                value
                for value in _CONTRACTIONS
                if text[index : index + len(value)].casefold() == value
            ),
            None,
        )
        if contraction is not None:
            pieces.append(text[index : index + len(contraction)])
            index += len(contraction)
            continue

        letter_start = index
        if not _is_letter(text[index]):
            if (
                text[index] in "\r\n"
                or _is_number(text[index])
                or index + 1 >= len(text)
                or not _is_letter(text[index + 1])
            ):
                letter_start = -1
            else:
                letter_start = index + 1
        if letter_start >= 0:
            end = letter_start + 1
            while end < len(text) and _is_letter(text[end]):
                end += 1
            pieces.append(text[index:end])
            index = end
            continue

        if _is_number(text[index]):
            end = index + 1
            while end < len(text) and end - index < 3 and _is_number(text[end]):
                end += 1
            pieces.append(text[index:end])
            index = end
            continue

        punctuation_start = index + (text[index] == " ")
        if punctuation_start < len(text) and not (
            text[punctuation_start] in _WHITESPACE
            or _is_letter(text[punctuation_start])
            or _is_number(text[punctuation_start])
        ):
            end = punctuation_start + 1
            while end < len(text) and not (
                text[end] in _WHITESPACE or _is_letter(text[end]) or _is_number(text[end])
            ):
                end += 1
            while end < len(text) and text[end] in "\r\n":
                end += 1
            pieces.append(text[index:end])
            index = end
            continue

        if text[index] in _WHITESPACE:
            end = index
            last_newline = -1
            while end < len(text) and text[end] in _WHITESPACE:
                if text[end] in "\r\n":
                    last_newline = end
                end += 1
            if last_newline >= 0:
                end = last_newline + 1
            elif end < len(text) and end - index > 1:
                end -= 1
            pieces.append(text[index:end])
            index = end
            continue

        raise ValueError(f"GLM pre-tokenizer could not split character {text[index]!r}")
    return pieces


class GlmTokenizer:
    """GLM-4 byte-level BPE tokenizer with CogView4 pipeline padding."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        path = Path(directory)
        data = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
        model = data["model"]
        self.encoder: dict[str, int] = model["vocab"]
        merges = [
            tuple(merge) if isinstance(merge, list) else tuple(merge.split())
            for merge in model["merges"]
        ]
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.ignore_merges = model.get("ignore_merges", False)
        self.byte_encoder = bytes_to_unicode()
        self._cache: dict[str, list[str]] = {}
        self._added = {token["content"]: token["id"] for token in data.get("added_tokens", [])}

        config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
        pad_token = config["pad_token"]
        if isinstance(pad_token, dict):
            pad_token = pad_token["content"]
        self.pad_id = self._added[pad_token]
        self.prefix_ids = [self._added["[gMASK]"], self._added["<sop>"]]

    def _bpe(self, token: str) -> list[str]:
        if self.ignore_merges and token in self.encoder:
            return [token]
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = list(token)
        while len(word) > 1:
            pairs = {(word[index], word[index + 1]) for index in range(len(word) - 1)}
            best = min(pairs, key=lambda pair: self.ranks.get(pair, float("inf")))
            if best not in self.ranks:
                break
            merged: list[str] = []
            index = 0
            while index < len(word):
                if index + 1 < len(word) and (word[index], word[index + 1]) == best:
                    merged.append(word[index] + word[index + 1])
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = merged
        self._cache[token] = word
        return word

    def _ordinary_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _pretokenize(text):
            encoded = "".join(self.byte_encoder[byte] for byte in piece.encode("utf-8"))
            ids.extend(self.encoder[token] for token in self._bpe(encoded))
        return ids

    def tokenize(self, text: str) -> list[int]:
        """Return ids with the GLM `[gMASK]`, `<sop>` prefix."""
        ids = list(self.prefix_ids)
        for piece, added_id in _split_added(text, self._added):
            if added_id is not None:
                ids.append(added_id)
            else:
                ids.extend(self._ordinary_ids(piece))
        return ids

    def __call__(
        self, texts: str | list[str], max_length: int = 1024
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize, left-pad to the longest row, then pad to a multiple of 16."""
        if isinstance(texts, str):
            texts = [texts]
        tokenized = [self.tokenize(text)[:max_length] for text in texts]
        width = max(len(ids) for ids in tokenized)
        width += (-width) % 16
        rows: list[list[int]] = []
        masks: list[list[bool]] = []
        for ids in tokenized:
            padding = width - len(ids)
            rows.append([self.pad_id] * padding + ids)
            masks.append([False] * padding + [True] * len(ids))
        return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)
