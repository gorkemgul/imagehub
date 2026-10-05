"""Native byte-level BPE tokenizer for Qwen2 tokenizer files."""

from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path
from typing import TypeAlias

import torch

from imagehub.tokenization import _WHITESPACE, _is_letter, _is_number, bytes_to_unicode

_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
_AddedPiece: TypeAlias = tuple[str, int | None]


def _split_added(text: str, tokens: dict[str, int]) -> list[_AddedPiece]:
    if not tokens:
        return [(text, None)]
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
    i = 0
    while i < len(text):
        contraction = next(
            (value for value in _CONTRACTIONS if text[i : i + len(value)].casefold() == value),
            None,
        )
        if contraction is not None:
            pieces.append(text[i : i + len(contraction)])
            i += len(contraction)
            continue

        letter_start = i
        if not _is_letter(text[i]):
            if (
                text[i] in "\r\n"
                or _is_number(text[i])
                or i + 1 >= len(text)
                or not _is_letter(text[i + 1])
            ):
                letter_start = -1
            else:
                letter_start = i + 1
        if letter_start >= 0:
            end = letter_start + 1
            while end < len(text) and _is_letter(text[end]):
                end += 1
            pieces.append(text[i:end])
            i = end
            continue

        if _is_number(text[i]):
            pieces.append(text[i])
            i += 1
            continue

        punctuation_start = i
        if text[i] == " ":
            punctuation_start += 1
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
            pieces.append(text[i:end])
            i = end
            continue

        if text[i] in _WHITESPACE:
            end = i
            last_newline = -1
            while end < len(text) and text[end] in _WHITESPACE:
                if text[end] in "\r\n":
                    last_newline = end
                end += 1
            if last_newline >= 0:
                end = last_newline + 1
            elif end < len(text) and end - i > 1:
                end -= 1
            pieces.append(text[i:end])
            i = end
            continue

        raise ValueError(f"Qwen2 pre-tokenizer could not split character {text[i]!r}")
    return pieces


class Qwen2Tokenizer:
    """Qwen2 NFC-normalized byte-level BPE tokenizer."""

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
        self.byte_encoder = bytes_to_unicode()
        self._cache: dict[str, list[str]] = {}

        raw_added: dict[str, int] = {}
        normalized_added: dict[str, int] = {}
        for token in data.get("added_tokens", []):
            destination = normalized_added if token.get("normalized", False) else raw_added
            destination[token["content"]] = token["id"]
        self._raw_added = raw_added
        self._normalized_added = normalized_added

        config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
        pad_token = config.get("pad_token", "<|endoftext|>")
        if isinstance(pad_token, dict):
            pad_token = pad_token["content"]
        self.pad_id = (
            self._raw_added[pad_token] if pad_token in self._raw_added else self.encoder[pad_token]
        )

    def _bpe(self, token: str) -> list[str]:
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
        """Return the same ids as Qwen2 ``encode(..., add_special_tokens=True)``."""
        ids: list[int] = []
        for raw, raw_id in _split_added(text, self._raw_added):
            if raw_id is not None:
                ids.append(raw_id)
                continue
            normalized = unicodedata.normalize("NFC", raw)
            for piece, normalized_id in _split_added(normalized, self._normalized_added):
                if normalized_id is not None:
                    ids.append(normalized_id)
                else:
                    ids.extend(self._ordinary_ids(piece))
        return ids

    def __call__(
        self, texts: str | list[str], max_length: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and return max-length padded ids and boolean attention masks."""
        if isinstance(texts, str):
            texts = [texts]
        rows: list[list[int]] = []
        masks: list[list[bool]] = []
        for text in texts:
            ids = self.tokenize(text)[:max_length]
            padding = max_length - len(ids)
            rows.append(ids + [self.pad_id] * padding)
            masks.append([True] * len(ids) + [False] * padding)
        return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

    def chat_prompt(self, prompt: str) -> str:
        """Render the pipeline's single-user-message chat template."""
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
