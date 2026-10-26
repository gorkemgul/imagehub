"""Native XLM-Roberta fast tokenizer read from ``tokenizer.json`` files.

The layout is the one produced by Hugging Face ``tokenizers`` for XLM-RoBERTa:
a SentencePiece precompiled normalizer, a ``WhitespaceSplit`` + ``Metaspace``
pre-tokenizer, an ``Unigram`` model, and a ``TemplateProcessing`` post-processor
that wraps the sequence with ``<s>`` and ``</s>``.  Special tokens are extracted
from the raw text before normalization; everything else is normalized,
pre-tokenized and encoded with the Unigram model.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal, cast

import torch

from imagehub.tokenization_t5 import _PrecompiledNormalizer, _Unigram


def _content(value: object) -> str | None:
    if isinstance(value, dict):
        value = value.get("content")
    return value if isinstance(value, str) else None


class XLMRobertaTokenizer:
    """Tokenizer loaded from a Hugging Face XLM-Roberta ``tokenizer`` directory."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        path = Path(directory)
        config: dict[str, Any] = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
        if config["model"]["type"] != "Unigram":
            raise ValueError("XLMRobertaTokenizer requires a Unigram model")
        self.model = _Unigram(config["model"])
        self._normalizers = self._load_normalizers(config.get("normalizer"))

        tokenizer_config: dict[str, Any] = {}
        tokenizer_config_path = path / "tokenizer_config.json"
        if tokenizer_config_path.is_file():
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        special_map: dict[str, Any] = {}
        special_map_path = path / "special_tokens_map.json"
        if special_map_path.is_file():
            special_map = json.loads(special_map_path.read_text(encoding="utf-8"))
        # special_tokens_map overrides tokenizer_config for special-token flags.
        token_flags = {
            **_special_flags(tokenizer_config.get("added_tokens_decoder", {})),
            **_special_flags(special_map),
        }

        added = config.get("added_tokens", [])
        self.added_ids = {str(item["content"]): int(item["id"]) for item in added}
        raw: list[str] = []
        normalized: list[str] = []
        for item in added:
            token = str(item["content"])
            is_normalized = token_flags.get(token, bool(item.get("normalized", False)))
            (normalized if is_normalized else raw).append(token)
        self._raw_added = self._added_pattern(raw)
        self._normalized_added = self._added_pattern(normalized)

        def token_id(name: str) -> int:
            token = _content(tokenizer_config.get(name)) or _content(special_map.get(name))
            if token is None:
                raise ValueError(f"tokenizer has no configured {name}")
            if token in self.added_ids:
                return self.added_ids[token]
            return self.model.byte_ids[token]

        self.bos_id = token_id("bos_token")
        self.eos_id = token_id("eos_token")
        self.pad_id = token_id("pad_token")
        self.unk_id = token_id("unk_token")
        self.max_length = int(tokenizer_config.get("model_max_length", 512))
        self.replacement = "▁"

    @staticmethod
    def _load_normalizers(config: dict[str, Any] | None) -> list[tuple[str, Any]]:
        if config is None:
            return []
        configs = config.get("normalizers", []) if config["type"] == "Sequence" else [config]
        normalizers: list[tuple[str, Any]] = []
        for item in configs:
            kind = item["type"]
            if kind == "Precompiled":
                normalizers.append((kind, _PrecompiledNormalizer(item["precompiled_charsmap"])))
            elif kind == "Replace":
                pattern = item["pattern"]
                if "Regex" in pattern:
                    pattern = re.compile(pattern["Regex"])
                else:
                    pattern = str(pattern["String"])
                normalizers.append((kind, (pattern, str(item["content"]))))
            elif kind in {"NFC", "NFD", "NFKC", "NFKD", "Lowercase", "Strip"}:
                normalizers.append((kind, item))
            else:
                raise ValueError(f"unsupported tokenizer normalizer: {kind}")
        return normalizers

    @staticmethod
    def _added_pattern(tokens: list[str]) -> re.Pattern[str] | None:
        if not tokens:
            return None
        ordered = sorted(tokens, key=len, reverse=True)
        return re.compile("(" + "|".join(re.escape(token) for token in ordered) + ")")

    def _normalize(self, text: str) -> str:
        for kind, value in self._normalizers:
            if kind == "Precompiled":
                text = value.normalize(text)
            elif kind == "Replace":
                pattern, replacement = value
                text = (
                    pattern.sub(replacement, text)
                    if hasattr(pattern, "sub")
                    else text.replace(pattern, replacement)
                )
            elif kind in {"NFC", "NFD", "NFKC", "NFKD"}:
                form = cast(Literal["NFC", "NFD", "NFKC", "NFKD"], kind)
                text = unicodedata.normalize(form, text)
            elif kind == "Lowercase":
                text = text.lower()
            elif kind == "Strip":
                text = text.strip()
        return text

    def _encode_plain(self, text: str) -> list[int]:
        """Encode one normalized segment with the WhitespaceSplit + Metaspace pre-tokenizer."""
        if not text:
            return []
        ids: list[int] = []
        for piece in re.split(r"\s+", text.strip()):
            if piece:
                ids.extend(self.model.encode(self.replacement + piece))
        return ids

    def _encode_segment(self, text: str) -> list[int]:
        if self._normalized_added is None:
            return self._encode_plain(text)
        ids: list[int] = []
        for index, piece in enumerate(self._normalized_added.split(text)):
            if index % 2:
                ids.append(self.added_ids[piece])
            else:
                ids.extend(self._encode_plain(piece))
        return ids

    def tokenize(self, text: str) -> list[int]:
        """Content token ids without ``<s>`` / ``</s>`` or padding."""
        pieces = self._raw_added.split(text) if self._raw_added is not None else [text]
        ids: list[int] = []
        for index, piece in enumerate(pieces):
            if index % 2:
                ids.append(self.added_ids[piece])
            else:
                ids.extend(self._encode_segment(self._normalize(piece)))
        return ids

    def __call__(
        self, text: str, max_length: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``<s>`` + content (truncated) + ``</s>``, padded with ``<pad>``."""
        if max_length is None:
            max_length = self.max_length
        content = self.tokenize(text)[: max_length - 2]
        ids = [self.bos_id, *content, self.eos_id]
        padding = max_length - len(ids)
        mask = [True] * len(ids) + [False] * padding
        ids = ids + [self.pad_id] * padding
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def _special_flags(config: object) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    if isinstance(config, dict):
        for value in config.values():
            if isinstance(value, dict):
                content = value.get("content")
                if isinstance(content, str):
                    flags[content] = bool(value.get("normalized", False))
    return flags
