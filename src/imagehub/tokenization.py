"""Native CLIP byte-level BPE tokenizer matching transformers' ``CLIPTokenizer``.

The reference first splits the raw text on exact-case ``<|startoftext|>`` / ``<|endoftext|>``
(case variants are ordinary text). Every other segment is normalized (NFC, Unicode whitespace
runs collapsed to one space, per-character lowercase) and split with the pattern
``'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+``.
Python's ``re`` has no ``\\p{..}`` classes, so the split is an explicit scanner.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

# Unicode White_Space property (what the reference Rust regex ``\s`` matches).
_WHITESPACE = frozenset("\t\n\x0b\x0c\r \x85\xa0                　")
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
BOS, EOS = "<|startoftext|>", "<|endoftext|>"
_SPECIAL_SPLIT = re.compile(f"({re.escape(BOS)}|{re.escape(EOS)})")


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 reversible byte -> printable unicode character table."""
    keep = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = keep[:]
    extra = 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(keep, map(chr, chars), strict=True))


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


def normalize(text: str) -> str:
    """NFC, collapse whitespace runs to one space, then lowercase each character.

    Per-character lowercasing matches the reference (no context-sensitive final sigma).
    """
    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    for ch in text:
        if ch in _WHITESPACE:
            if not out or out[-1] != " ":
                out.append(" ")
        else:
            out.append(ch)
    return "".join(ch.lower() for ch in out)


def pretokenize(text: str) -> list[str]:
    """Split like the reference regex: contractions | letters+ | one digit | other+.

    Special tokens are not handled here; ``CLIPTokenizer.tokenize`` splits them off the raw text.
    """
    pieces: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _WHITESPACE:
            i += 1
            continue
        contraction = next((c for c in _CONTRACTIONS if text.startswith(c, i)), None)
        if contraction is not None:
            pieces.append(contraction)
            i += len(contraction)
            continue
        j = i + 1
        if _is_letter(ch):
            while j < n and _is_letter(text[j]):
                j += 1
        elif not _is_number(ch):
            while j < n and not (
                text[j] in _WHITESPACE or _is_letter(text[j]) or _is_number(text[j])
            ):
                j += 1
        pieces.append(text[i:j])
        i = j
    return pieces


class CLIPTokenizer:
    def __init__(self, directory: str | os.PathLike[str], *, max_length: int = 77) -> None:
        path = Path(directory)
        self.encoder: dict[str, int] = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        lines = (path / "merges.txt").read_text(encoding="utf-8").splitlines()
        merges = [tuple(line.split()) for line in lines if line and not line.startswith("#version")]
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.bos_id, self.eos_id = self.encoder[BOS], self.encoder[EOS]
        self.pad_id = self.eos_id
        self.max_length = max_length
        self._cache: dict[str, list[str]] = {}

    def _bpe(self, token: str) -> list[str]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            pairs = {(word[k], word[k + 1]) for k in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            merged: list[str] = []
            k = 0
            while k < len(word):
                if k < len(word) - 1 and (word[k], word[k + 1]) == best:
                    merged.append(word[k] + word[k + 1])
                    k += 2
                else:
                    merged.append(word[k])
                    k += 1
            word = merged
        self._cache[token] = word
        return word

    def tokenize(self, text: str) -> list[int]:
        """Content token ids without BOS/EOS, padding or truncation."""
        ids: list[int] = []
        for k, segment in enumerate(_SPECIAL_SPLIT.split(text)):
            if k % 2:  # odd items are the captured exact-case specials
                ids.append(self.encoder[segment])
                continue
            for piece in pretokenize(normalize(segment)):
                mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
                ids.extend(self.encoder[symbol] for symbol in self._bpe(mapped))
        return ids

    def __call__(self, text: str) -> list[int]:
        """BOS + content (truncated) + EOS, padded with EOS to ``max_length``."""
        content = self.tokenize(text)[: self.max_length - 2]
        ids = [self.bos_id, *content, self.eos_id]
        return ids + [self.pad_id] * (self.max_length - len(ids))
