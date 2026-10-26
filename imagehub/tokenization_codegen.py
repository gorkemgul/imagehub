"""CodeGen byte-level BPE tokenizer used by Phi-1.5 and Show-o."""

from __future__ import annotations

import json
import os
from pathlib import Path

from imagehub.tokenization import _WHITESPACE, _is_letter, _is_number, bytes_to_unicode

_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _split_added(text: str, added: dict[str, int]) -> list[tuple[str, int | None]]:
    result: list[tuple[str, int | None]] = []
    start = 0
    while start < len(text):
        matches = [
            (text.find(token, start), -len(token), token)
            for token in added
            if text.find(token, start) >= 0
        ]
        if not matches:
            result.append((text[start:], None))
            break
        position, _, token = min(matches)
        if position > start:
            result.append((text[start:position], None))
        result.append((token, added[token]))
        start = position + len(token)
    return result


def _split(text: str) -> list[str]:
    pieces: list[str] = []
    index = 0
    while index < len(text):
        contraction = next(
            (c for c in _CONTRACTIONS if text[index : index + len(c)].casefold() == c), None
        )
        if contraction is not None:
            pieces.append(text[index : index + len(contraction)])
            index += len(contraction)
            continue
        content = index + (text[index] == " ")
        if content < len(text) and _is_letter(text[content]):
            end = content + 1
            while end < len(text) and _is_letter(text[end]):
                end += 1
            pieces.append(text[index:end])
            index = end
            continue
        if content < len(text) and _is_number(text[content]):
            end = content + 1
            while end < len(text) and _is_number(text[end]):
                end += 1
            pieces.append(text[index:end])
            index = end
            continue
        if content < len(text) and text[content] not in _WHITESPACE:
            end = content + 1
            while end < len(text) and not (
                text[end] in _WHITESPACE or _is_letter(text[end]) or _is_number(text[end])
            ):
                end += 1
            pieces.append(text[index:end])
            index = end
            continue
        end = index + 1
        while end < len(text) and text[end] in _WHITESPACE:
            end += 1
        if end < len(text) and end - index > 1:
            end -= 1
        pieces.append(text[index:end])
        index = end
    return pieces


class CodeGenTokenizer:
    def __init__(self, directory: str | os.PathLike[str]) -> None:
        path = Path(directory)
        self.encoder: dict[str, int] = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        added = json.loads((path / "added_tokens.json").read_text(encoding="utf-8"))
        self.added: dict[str, int] = added
        self.encoder.update(added)
        merges = [
            tuple(line.split())
            for line in (path / "merges.txt").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self._cache: dict[str, list[str]] = {}
        self.eos_id = self.encoder["<|endoftext|>"]
        self.pad_id = 50295
        self.special_ids = {
            token: 50296 + index
            for index, token in enumerate(
                (
                    "<|soi|>",
                    "<|eoi|>",
                    "<|sov|>",
                    "<|eov|>",
                    "<|t2i|>",
                    "<|mmu|>",
                    "<|t2v|>",
                    "<|v2v|>",
                    "<|lvg|>",
                )
            )
        }

    def _bpe(self, token: str) -> list[str]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = list(token)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
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

    def tokenize(self, text: str) -> list[int]:
        ids: list[int] = []
        for segment, added_id in _split_added(text, self.added):
            if added_id is not None:
                ids.append(added_id)
                continue
            for piece in _split(segment):
                encoded = "".join(self.byte_encoder[byte] for byte in piece.encode("utf-8"))
                ids.extend(self.encoder[token] for token in self._bpe(encoded))
        return ids
