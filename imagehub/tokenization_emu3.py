"""Dependency-free byte-pair tokenizer for Emu3's published tiktoken vocabulary."""

from __future__ import annotations

import base64
import re
import unicodedata
from pathlib import Path

_PATTERN = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w]?[^\W\d_]+|\d|"
    r" ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_SPECIAL_START = 151643
_BASE_SPECIALS = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    *(f"<|extra_{index}|>" for index in range(205)),
)


class Emu3Tokenizer:
    """Emu3 byte BPE, including its image-control and 32,768 visual tokens."""

    def __init__(self, directory: str | Path) -> None:
        root = Path(directory)
        self.ranks = {
            base64.b64decode(token): int(rank)
            for line in (root / "emu3.tiktoken").read_bytes().splitlines()
            if line
            for token, rank in (line.split(),)
        }
        vision = tuple(
            line.strip()
            for line in (root / "emu3_vision_tokens.txt").read_text().splitlines()
            if line.strip()
        )
        self.special = {
            token: index
            for index, token in enumerate(_BASE_SPECIALS + vision, start=_SPECIAL_START)
        }
        self.decoder: dict[int, bytes | str] = {value: key for key, value in self.ranks.items()}
        self.decoder.update({value: key for key, value in self.special.items()})
        self.bos_id = self.special["<|extra_203|>"]
        self.eos_id = self.special["<|extra_204|>"]
        self.pad_id = self.special["<|endoftext|>"]
        self.img_id = self.special["<|image token|>"]
        self.boi_id = self.special["<|image start|>"]
        self.eoi_id = self.special["<|image end|>"]
        self.eol_id = self.special["<|extra_200|>"]
        self.eof_id = self.special["<|extra_201|>"]
        self.visual_start = self.special["<|visual token 000000|>"]
        self.visual_end = self.special["<|visual token 032767|>"]
        self._special_pattern = re.compile(r"(<\|.*?\|>)")

    def _bpe(self, piece: bytes) -> list[int]:
        parts = [bytes((byte,)) for byte in piece]
        while len(parts) > 1:
            candidates = [
                self.ranks.get(left + right) for left, right in zip(parts, parts[1:], strict=False)
            ]
            available = [rank for rank in candidates if rank is not None]
            if not available:
                break
            selected = min(available)
            merged: list[bytes] = []
            index = 0
            while index < len(parts):
                if (
                    index + 1 < len(parts)
                    and self.ranks.get(parts[index] + parts[index + 1]) == selected
                ):
                    merged.append(parts[index] + parts[index + 1])
                    index += 2
                else:
                    merged.append(parts[index])
                    index += 1
            parts = merged
        return [self.ranks[part] for part in parts]

    def encode(self, text: str) -> list[int]:
        result: list[int] = []
        for chunk in self._special_pattern.split(unicodedata.normalize("NFC", text)):
            if not chunk:
                continue
            if chunk in self.special:
                result.append(self.special[chunk])
                continue
            for piece in _PATTERN.findall(chunk):
                result.extend(self._bpe(piece.encode()))
        return result

    def decode(self, ids: list[int]) -> str:
        result, pending = "", b""
        for token_id in ids:
            token = self.decoder[token_id]
            if isinstance(token, bytes):
                pending += token
            else:
                result += pending.decode(errors="replace") + token
                pending = b""
        return result + pending.decode(errors="replace")
