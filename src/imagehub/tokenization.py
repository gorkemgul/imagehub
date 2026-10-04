"""Native CLIP byte-level BPE tokenizer matching transformers' ``CLIPTokenizer``.

Each special token carries a ``normalized`` flag read from the tokenizer config. The reference
first extracts ``normalized=False`` specials from the raw text by exact (case-sensitive) match.
The remaining segments are normalized (NFC, Unicode whitespace runs collapsed to one space,
per-character lowercase); ``normalized=True`` specials are then extracted from the normalized
text by exact match (so they also match after lowercasing, e.g. ``<|StartOfText|>``). Everything
else is split with the pattern
``<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+``
(leftmost-first alternation), followed by the GPT-2 ByteLevel regex. The latter is the identity
on every piece except a special literal that was not extracted as an added token (for example
``<|ENDOFTEXT|>`` lowercased): it is captured whole and then re-split into ``<|``, the name and
``|>``, so ``|>`` never merges with following punctuation. Python's ``re`` has no ``\\p{..}``
classes, so the split is an explicit scanner.
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
_SPECIAL_KEYS = ("unk_token", "bos_token", "eos_token", "pad_token")


def _split_on(text: str, tokens: list[str]) -> list[str]:
    """Split ``text`` on exact occurrences of ``tokens``; odd items of the result are the tokens."""
    if not tokens:
        return [text]
    return re.split("(" + "|".join(re.escape(t) for t in tokens) + ")", text)


def special_token_flags(directory: str | os.PathLike[str]) -> dict[str, bool]:
    """Map ``<|startoftext|>`` / ``<|endoftext|>`` to their ``normalized`` flag.

    Mirrors transformers: the tokens are registered in the order unk, bos, eos, pad from
    ``tokenizer_config.json`` (``special_tokens_map.json`` as fallback); when several keys share
    the same content the last registration wins. A plain-string value registers with
    ``normalized=False``, a dict value with its ``"normalized"`` field. Without any config both
    specials are ``normalized=False``.
    """
    path = Path(directory)
    config: dict[str, object] = {}
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        if (path / name).is_file():
            config = json.loads((path / name).read_text(encoding="utf-8"))
            break
    flags: dict[str, bool] = {}
    for key in _SPECIAL_KEYS:
        value = config.get(key)
        if isinstance(value, str):
            flags[value] = False
        elif isinstance(value, dict) and isinstance(value.get("content"), str):
            flags[value["content"]] = bool(value.get("normalized", False))
    return {BOS: flags.get(BOS, False), EOS: flags.get(EOS, False)}


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
    """Split like the reference Split(regex) + ByteLevel pre-tokenizer, scanning left to right.

    At each position (after skipping whitespace) the first matching alternative wins: a literal
    ``<|startoftext|>`` / ``<|endoftext|>`` (emitted as ``<|``, name, ``|>``, as ByteLevel would
    re-split it), a contraction, a letter run, a single number, or a greedy run of characters
    that are neither whitespace, letters nor numbers (it does not stop at ``<|``).
    Extraction of special tokens is done by ``CLIPTokenizer.tokenize`` beforehand.
    """
    pieces: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _WHITESPACE:
            i += 1
            continue
        literal = next((s for s in (BOS, EOS) if text.startswith(s, i)), None)
        if literal is not None:
            pieces.extend(("<|", literal[2:-2], "|>"))
            i += len(literal)
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
        self.special_normalized = special_token_flags(path)
        self._raw_specials = [t for t, norm in self.special_normalized.items() if not norm]
        self._normalized_specials = [t for t, norm in self.special_normalized.items() if norm]
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
        """Content token ids without BOS/EOS, padding or truncation.

        Specials with ``normalized=False`` are split off the raw text (exact case); each remaining
        segment is normalized, then split on ``normalized=True`` specials, then pretokenized + BPE.
        """
        ids: list[int] = []
        for k, raw in enumerate(_split_on(text, self._raw_specials)):
            if k % 2:
                ids.append(self.encoder[raw])
                continue
            for j, segment in enumerate(_split_on(normalize(raw), self._normalized_specials)):
                if j % 2:
                    ids.append(self.encoder[segment])
                    continue
                for piece in pretokenize(segment):
                    mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
                    ids.extend(self.encoder[symbol] for symbol in self._bpe(mapped))
        return ids

    def __call__(self, text: str) -> list[int]:
        """BOS + content (truncated) + EOS, padded with EOS to ``max_length``."""
        content = self.tokenize(text)[: self.max_length - 2]
        ids = [self.bos_id, *content, self.eos_id]
        return ids + [self.pad_id] * (self.max_length - len(ids))
