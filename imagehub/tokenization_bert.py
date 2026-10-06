"""Native BERT WordPiece tokenizer."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import torch


def _content(value: object, default: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return default


def _is_chinese(character: str) -> bool:
    code_point = ord(character)
    return (
        0x4E00 <= code_point <= 0x9FFF
        or 0x3400 <= code_point <= 0x4DBF
        or 0x20000 <= code_point <= 0x2A6DF
        or 0x2A700 <= code_point <= 0x2B73F
        or 0x2B740 <= code_point <= 0x2B81F
        or 0x2B820 <= code_point <= 0x2CEAF
        or 0xF900 <= code_point <= 0xFAFF
        or 0x2F800 <= code_point <= 0x2FA1F
    )


def _is_control(character: str) -> bool:
    if character in "\t\n\r":
        return False
    return unicodedata.category(character) in {"Cc", "Cf"}


def _is_punctuation(character: str) -> bool:
    code_point = ord(character)
    return (
        33 <= code_point <= 47
        or 58 <= code_point <= 64
        or 91 <= code_point <= 96
        or 123 <= code_point <= 126
        or unicodedata.category(character).startswith("P")
    )


class BertTokenizer:
    """BERT basic tokenizer and WordPiece model loaded from ``vocab.txt``."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        path = Path(directory)
        tokens = (path / "vocab.txt").read_text(encoding="utf-8").splitlines()
        self.vocab: dict[str, int] = {}
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        config: dict[str, Any] = json.loads(
            (path / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        self.do_lower_case = bool(config.get("do_lower_case", True))
        self.tokenize_chinese_chars = bool(config.get("tokenize_chinese_chars", True))
        strip_accents = config.get("strip_accents")
        self.strip_accents = self.do_lower_case if strip_accents is None else bool(strip_accents)

        self.unk_token = _content(config.get("unk_token"), "[UNK]")
        self.cls_token = _content(config.get("cls_token"), "[CLS]")
        self.sep_token = _content(config.get("sep_token"), "[SEP]")
        self.pad_token = _content(config.get("pad_token"), "[PAD]")
        self.unk_id = self.vocab[self.unk_token]
        self.cls_id = self.vocab[self.cls_token]
        self.sep_id = self.vocab[self.sep_token]
        self.pad_id = self.vocab[self.pad_token]

        decoder = config.get("added_tokens_decoder", {})
        raw_tokens = [
            str(item["content"])
            for item in decoder.values()
            if isinstance(item, dict) and not item.get("normalized", False)
        ]
        if not raw_tokens:
            raw_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        raw_tokens = [token for token in raw_tokens if token in self.vocab]
        self._raw_added = re.compile(
            "("
            + "|".join(re.escape(token) for token in sorted(raw_tokens, key=len, reverse=True))
            + ")"
        )

    def _normalize(self, text: str) -> str:
        output: list[str] = []
        for character in text:
            if character == "\0" or character == "\ufffd" or _is_control(character):
                continue
            if character.isspace():
                output.append(" ")
            elif self.tokenize_chinese_chars and _is_chinese(character):
                output.extend((" ", character, " "))
            else:
                output.append(character)
        text = "".join(output)
        if self.do_lower_case:
            # Per character: str.lower() maps a word-final capital sigma to ς, the reference to σ.
            text = "".join(character.lower() for character in text)
        if self.strip_accents:
            text = "".join(
                character
                for character in unicodedata.normalize("NFD", text)
                if unicodedata.category(character) != "Mn"
            )
        return text

    @staticmethod
    def _basic_tokens(text: str) -> list[str]:
        tokens: list[str] = []
        current: list[str] = []
        for character in text:
            if character.isspace():
                if current:
                    tokens.append("".join(current))
                    current.clear()
            elif _is_punctuation(character):
                if current:
                    tokens.append("".join(current))
                    current.clear()
                tokens.append(character)
            else:
                current.append(character)
        if current:
            tokens.append("".join(current))
        return tokens

    def _wordpiece(self, token: str) -> list[int]:
        if len(token) > 100:
            return [self.unk_id]
        output: list[int] = []
        start = 0
        while start < len(token):
            end = len(token)
            token_id: int | None = None
            while start < end:
                piece = token[start:end]
                if start:
                    piece = "##" + piece
                token_id = self.vocab.get(piece)
                if token_id is not None:
                    break
                end -= 1
            if token_id is None:
                return [self.unk_id]
            output.append(token_id)
            start = end
        return output

    def encode(self, text: str) -> list[int]:
        ids = [self.cls_id]
        for index, piece in enumerate(self._raw_added.split(text)):
            if index % 2:
                ids.append(self.vocab[piece])
                continue
            for token in self._basic_tokens(self._normalize(piece)):
                ids.extend(self._wordpiece(token))
        ids.append(self.sep_id)
        return ids

    def tokenize(self, text: str) -> list[int]:
        return self.encode(text)

    def __call__(
        self, texts: list[str] | tuple[str, ...], max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows: list[list[int]] = []
        masks: list[list[bool]] = []
        for text in texts:
            ids = self.encode(text)
            if len(ids) > max_length:
                ids = ids[: max_length - 1] + [self.sep_id]
            padding = max_length - len(ids)
            rows.append(ids + [self.pad_id] * padding)
            masks.append([True] * len(ids) + [False] * padding)
        return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)
