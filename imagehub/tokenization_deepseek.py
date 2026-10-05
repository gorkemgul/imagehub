"""Native DeepSeek byte-level BPE tokenizer used by Janus-Pro."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import torch

from imagehub.tokenization import _WHITESPACE, bytes_to_unicode


def _added_pattern(tokens: dict[str, int]) -> re.Pattern[str] | None:
    if not tokens:
        return None
    alternatives = sorted(tokens, key=lambda token: (-len(token), token))
    return re.compile("|".join(re.escape(token) for token in alternatives))


def _split_added(
    text: str, tokens: dict[str, int], pattern: re.Pattern[str] | None
) -> list[tuple[str, int | None]]:
    if pattern is None:
        return [(text, None)]
    pieces: list[tuple[str, int | None]] = []
    start = 0
    for match in pattern.finditer(text):
        if match.start() > start:
            pieces.append((text[start : match.start()], None))
        token = match.group()
        pieces.append((token, tokens[token]))
        start = match.end()
    if start < len(text) or not text:
        pieces.append((text[start:], None))
    return pieces


def _python_regex(pattern: str) -> re.Pattern[str]:
    whitespace = "[" + re.escape("".join(_WHITESPACE)) + "]"
    translated = pattern.replace(r"\s", whitespace)
    if translated.endswith("$"):
        translated = translated[:-1] + r"\Z"
    return re.compile(translated)


def _split_isolated(texts: list[str], pattern: re.Pattern[str]) -> list[str]:
    result: list[str] = []
    for text in texts:
        start = 0
        for match in pattern.finditer(text):
            if match.start() > start:
                result.append(text[start : match.start()])
            if match.end() > match.start():
                result.append(match.group())
            start = match.end()
        if start < len(text):
            result.append(text[start:])
    return result


def _split_digits(texts: list[str]) -> list[str]:
    result: list[str] = []
    for text in texts:
        start = 0
        for index, character in enumerate(text):
            if unicodedata.category(character).startswith("N"):
                if index > start:
                    result.append(text[start:index])
                result.append(character)
                start = index + 1
        if start < len(text):
            result.append(text[start:])
    return result


class DeepSeekTokenizer:
    """DeepSeek LLM tokenizer with Janus image-prompt batching."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        path = Path(directory)
        data: dict[str, Any] = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
        model = data["model"]
        self.encoder: dict[str, int] = model["vocab"]
        merges = [
            tuple(merge) if isinstance(merge, list) else tuple(merge.split())
            for merge in model["merges"]
        ]
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self._cache: dict[str, list[str]] = {}

        self._raw_added: dict[str, int] = {}
        self._normalized_added: dict[str, int] = {}
        for token in data.get("added_tokens", []):
            added_destination = (
                self._normalized_added if token.get("normalized", False) else self._raw_added
            )
            added_destination[token["content"]] = token["id"]
        self._raw_pattern = _added_pattern(self._raw_added)
        self._normalized_pattern = _added_pattern(self._normalized_added)

        normalizer = data.get("normalizer")
        if normalizer != {"type": "Sequence", "normalizers": []}:
            raise ValueError("DeepSeek tokenizer requires an empty normalizer sequence")

        pre_tokenizer = data["pre_tokenizer"]
        if pre_tokenizer.get("type") != "Sequence":
            raise ValueError("DeepSeek tokenizer requires a pre-tokenizer sequence")
        self._pre_tokenizers: list[tuple[str, re.Pattern[str] | None]] = []
        for item in pre_tokenizer["pretokenizers"]:
            kind = item["type"]
            if kind == "Split":
                if item.get("behavior") != "Isolated" or item.get("invert", False):
                    raise ValueError(
                        "DeepSeek tokenizer only supports non-inverted Isolated splits"
                    )
                self._pre_tokenizers.append((kind, _python_regex(item["pattern"]["Regex"])))
            elif kind in {"Digits", "ByteLevel"}:
                self._pre_tokenizers.append((kind, None))
            else:
                raise ValueError(f"Unsupported DeepSeek pre-tokenizer {kind!r}")

        config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
        pad_token = config["pad_token"]
        if isinstance(pad_token, dict):
            pad_token = pad_token["content"]
        self.pad_id = self._token_id(pad_token)

        single = data["post_processor"]["single"]
        special = data["post_processor"]["special_tokens"]
        self._prefix_ids: list[int] = []
        self._suffix_ids: list[int] = []
        post_destination = self._prefix_ids
        for item in single:
            if "Sequence" in item:
                post_destination = self._suffix_ids
            else:
                name = item["SpecialToken"]["id"]
                post_destination.extend(special[name]["ids"])

    def _token_id(self, token: str) -> int:
        if token in self._raw_added:
            return self._raw_added[token]
        if token in self._normalized_added:
            return self._normalized_added[token]
        return self.encoder[token]

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

    def _pretokenize(self, text: str) -> list[str]:
        pieces = [text]
        for kind, pattern in self._pre_tokenizers:
            if kind == "Split":
                assert pattern is not None
                pieces = _split_isolated(pieces, pattern)
            elif kind == "Digits":
                pieces = _split_digits(pieces)
            else:
                return [
                    "".join(self.byte_encoder[byte] for byte in piece.encode("utf-8"))
                    for piece in pieces
                ]
        raise ValueError("DeepSeek pre-tokenizer has no ByteLevel stage")

    def _ordinary_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self._pretokenize(text):
            ids.extend(self.encoder[token] for token in self._bpe(piece))
        return ids

    def tokenize(self, text: str) -> list[int]:
        """Return ids matching reference ``encode`` special-token handling."""
        ids = list(self._prefix_ids)
        for raw, raw_id in _split_added(text, self._raw_added, self._raw_pattern):
            if raw_id is not None:
                ids.append(raw_id)
                continue
            for piece, normalized_id in _split_added(
                raw, self._normalized_added, self._normalized_pattern
            ):
                if normalized_id is not None:
                    ids.append(normalized_id)
                else:
                    ids.extend(self._ordinary_ids(piece))
        ids.extend(self._suffix_ids)
        return ids

    def __call__(self, texts: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and dynamically left-pad a batch like ``JanusProcessor``."""
        if isinstance(texts, str):
            texts = [texts]
        tokenized = [self.tokenize(text) for text in texts]
        width = max((len(ids) for ids in tokenized), default=0)
        rows: list[list[int]] = []
        masks: list[list[bool]] = []
        for ids in tokenized:
            padding = width - len(ids)
            rows.append([self.pad_id] * padding + ids)
            masks.append([False] * padding + [True] * len(ids))
        return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

    def image_prompt(self, prompt: str) -> str:
        """Render one user turn and append the Janus begin-of-image token."""
        return f"<|User|>: {prompt.strip()}\n\n<|Assistant|>:<begin_of_image>"
