"""Native byte-level BPE tokenizer of the Mistral/Ministral "Tekken" vocabulary (``tokenizer.json``).

The reference (``TokenizersBackend``) applies no normalizer, splits the text with an isolated regex
split, maps the pieces to bytes-as-unicode and merges them; a piece that is already a vocabulary entry
skips the merges (``ignore_merges``). Added tokens are matched in the raw text before any of this, and
the post-processor prepends the beginning-of-sequence token. Python's ``re`` has no ``\\p{..}``
classes, so they are expanded into explicit code point ranges from ``unicodedata``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from imagehub.tokenization import _WHITESPACE, bytes_to_unicode
from imagehub.tokenization_deepseek import _added_pattern, _split_added, _split_isolated


@lru_cache(maxsize=1)
def _codepoints_by_category() -> dict[str, list[int]]:
    table: dict[str, list[int]] = {}
    for codepoint in range(sys.maxunicode + 1):
        table.setdefault(unicodedata.category(chr(codepoint)), []).append(codepoint)
    return table


def _ranges(codepoints: list[int]) -> str:
    parts: list[str] = []
    start = previous = None
    for codepoint in sorted(codepoints):
        if start is None:
            start = previous = codepoint
        elif codepoint == previous + 1:  # type: ignore[operator]
            previous = codepoint
        else:
            parts.append(_range(start, previous))  # type: ignore[arg-type]
            start = previous = codepoint
    if start is not None:
        parts.append(_range(start, previous))  # type: ignore[arg-type]
    return "".join(parts)


def _range(start: int, end: int) -> str:
    if start == end:
        return f"\\U{start:08x}"
    return f"\\U{start:08x}-\\U{end:08x}"


@lru_cache(maxsize=None)
def _property_body(name: str) -> str:
    table = _codepoints_by_category()
    members = [codepoint for category, points in table.items() if category.startswith(name) for codepoint in points]
    if not members:
        raise ValueError(f"Unknown Unicode property {name!r}")
    return _ranges(members)


_WHITESPACE_BODY = "".join(f"\\U{ord(ch):08x}" for ch in sorted(_WHITESPACE))


def python_regex(pattern: str) -> re.Pattern[str]:
    """Translate ``\\p{..}``, ``\\s`` and ``\\S`` of a tokenizers pattern into ``re`` syntax."""
    out: list[str] = []
    in_class = False
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\" and index + 1 < len(pattern):
            escape = pattern[index + 1]
            if escape == "p" and pattern[index + 2 : index + 3] == "{":
                end = pattern.index("}", index)
                body = _property_body(pattern[index + 3 : end])
                out.append(body if in_class else f"[{body}]")
                index = end + 1
                continue
            if escape == "s":
                out.append(_WHITESPACE_BODY if in_class else f"[{_WHITESPACE_BODY}]")
                index += 2
                continue
            if escape == "S":
                if in_class:
                    raise ValueError("\\S inside a character class is not supported")
                out.append(f"[^{_WHITESPACE_BODY}]")
                index += 2
                continue
            out.append(pattern[index : index + 2])
            index += 2
            continue
        if char == "[" and not in_class:
            in_class = True
        elif char == "]" and in_class:
            in_class = False
        out.append(char)
        index += 1
    return re.compile("".join(out))


class MistralTokenizer:
    """Byte-level BPE tokenizer with the Mistral ``tokenizer.json`` layout."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        path = Path(directory)
        data: dict[str, Any] = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
        config: dict[str, Any] = json.loads(
            (path / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        if data.get("normalizer") is not None:
            raise ValueError("Mistral tokenizer requires an empty normalizer")
        model = data["model"]
        if model.get("type") != "BPE" or model.get("byte_fallback") or model.get("dropout"):
            raise ValueError("Mistral tokenizer requires a plain byte-level BPE model")
        self.ignore_merges = bool(model.get("ignore_merges", False))
        self.encoder: dict[str, int] = model["vocab"]
        self.ranks = {
            (tuple(merge) if isinstance(merge, list) else tuple(merge.split())): rank
            for rank, merge in enumerate(model["merges"])
        }
        self.byte_encoder = bytes_to_unicode()
        self._cache: dict[str, list[str]] = {}

        self._added: dict[str, int] = {}
        for token in data.get("added_tokens", []):
            if token["normalized"] or token["lstrip"] or token["rstrip"] or token["single_word"]:
                raise ValueError("Mistral tokenizer supports plain added tokens only")
            self._added[token["content"]] = token["id"]
        self._added_pattern = _added_pattern(self._added)

        pre = data["pre_tokenizer"]
        stages = pre["pretokenizers"] if pre["type"] == "Sequence" else [pre]
        self._split: re.Pattern[str] | None = None
        self._byte_level = False
        for stage in stages:
            if stage["type"] == "Split":
                if stage["behavior"] != "Isolated" or stage.get("invert") or self._byte_level:
                    raise ValueError("Mistral tokenizer only supports a leading isolated split")
                self._split = python_regex(stage["pattern"]["Regex"])
            elif stage["type"] == "ByteLevel":
                if stage.get("add_prefix_space") or stage.get("use_regex"):
                    raise ValueError("Unsupported ByteLevel pre-tokenizer options")
                self._byte_level = True
            else:
                raise ValueError(f"Unsupported pre-tokenizer {stage['type']!r}")
        if not self._byte_level:
            raise ValueError("Mistral tokenizer requires a ByteLevel pre-tokenizer")

        processor = data["post_processor"]
        if processor["type"] != "TemplateProcessing":
            raise ValueError("Unsupported Mistral post-processor")
        self._prefix_ids: list[int] = []
        self._suffix_ids: list[int] = []
        destination = self._prefix_ids
        for item in processor["single"]:
            if "Sequence" in item:
                destination = self._suffix_ids
            else:
                destination.extend(processor["special_tokens"][item["SpecialToken"]["id"]]["ids"])
        self.max_length = int(config["model_max_length"])
        self.bos_id = self._added[config["bos_token"]]

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
        pieces = _split_isolated([text], self._split) if self._split is not None else [text]
        ids: list[int] = []
        for piece in pieces:
            mapped = "".join(self.byte_encoder[byte] for byte in piece.encode("utf-8"))
            if self.ignore_merges and mapped in self.encoder:
                ids.append(self.encoder[mapped])
            else:
                ids.extend(self.encoder[token] for token in self._bpe(mapped))
        return ids

    def tokenize(self, text: str) -> list[int]:
        """Ids with the added tokens extracted, the template applied and no truncation."""
        ids = list(self._prefix_ids)
        for piece, added_id in _split_added(text, self._added, self._added_pattern):
            if added_id is not None:
                ids.append(added_id)
            elif piece:
                ids.extend(self._ordinary_ids(piece))
        ids.extend(self._suffix_ids)
        return ids

    def __call__(self, text: str, max_length: int | None = None) -> list[int]:
        """Encode like ``tokenizer(text, add_special_tokens=True, truncation=True)``."""
        limit = self.max_length if max_length is None else max_length
        ids = self.tokenize(text)
        extra = len(self._prefix_ids) + len(self._suffix_ids)
        if len(ids) > limit:
            body = ids[len(self._prefix_ids) : len(ids) - len(self._suffix_ids)]
            ids = self._prefix_ids + body[: max(limit - extra, 0)] + self._suffix_ids
        return ids
