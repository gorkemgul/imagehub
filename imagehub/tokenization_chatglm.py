"""Native ChatGLM3 SentencePiece (BPE) tokenizer used by Kolors."""

from __future__ import annotations

import heapq
import os
from pathlib import Path

import torch

from imagehub.tokenization_t5 import _load_sentencepiece, _PrecompiledNormalizer

_SPACE = "▁"
_UNUSED = 5
_PREFIX_TOKENS = ("[gMASK]", "sop")
_EXTRA_TOKENS = (
    "[MASK]",
    "[gMASK]",
    "[sMASK]",
    "sop",
    "eop",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
)


def _character_length(first_byte: int) -> int:
    if first_byte < 0x80:
        return 1
    if first_byte < 0xE0:
        return 2
    if first_byte < 0xF0:
        return 3
    return 4


class ChatGLMTokenizer:
    """Tokenizer loaded from a ChatGLM3 ``tokenizer.model`` (file or directory).

    Pieces follow the SentencePiece BPE algorithm (greedy highest-score merges with
    byte fallback). The role and mask markers are appended after the model vocabulary
    and are never produced from text. Batches are left padded and carry the
    ``[gMASK] sop`` prefix, exactly like the reference tokenizer.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        model_path = Path(path)
        if model_path.is_dir():
            model_path = model_path / "tokenizer.model"
        model = _load_sentencepiece(model_path)
        if model.trainer_spec.model_type != 2:
            raise ValueError("ChatGLMTokenizer requires a SentencePiece BPE model")
        self.pieces = [piece for piece, _, _ in model.pieces]
        self.scores = [score for _, score, _ in model.pieces]
        self.types = [kind for _, _, kind in model.pieces]
        self.unk_id = model.trainer_spec.unk_id
        self.pad_id = self.unk_id
        self.eos_id = model.trainer_spec.eos_id
        self.byte_fallback = model.trainer_spec.byte_fallback
        self.piece_ids = {piece: index for index, piece in enumerate(self.pieces)}
        self.vocab_size = len(self.pieces) + len(_EXTRA_TOKENS)
        self.special_ids = {
            token: len(self.pieces) + index for index, token in enumerate(_EXTRA_TOKENS)
        }
        self.prefix_ids = [self.special_ids[token] for token in _PREFIX_TOKENS]

        spec = model.normalizer_spec
        self._charsmap = (
            _PrecompiledNormalizer(spec.precompiled_charsmap) if spec.precompiled_charsmap else None
        )
        self._add_dummy_prefix = spec.add_dummy_prefix
        self._remove_extra_whitespaces = spec.remove_extra_whitespaces
        self._escape_whitespaces = spec.escape_whitespaces
        self._merge_pieces = {
            piece: index
            for index, piece in enumerate(self.pieces)
            if self.types[index] in (1, 4, _UNUSED)
        }

    def _normalize(self, text: str) -> str:
        if self._charsmap is not None:
            text = self._charsmap.normalize(text)
        if self._remove_extra_whitespaces:
            text = " ".join(part for part in text.split(" ") if part)
        if not text:
            return ""
        if self._add_dummy_prefix:
            text = " " + text
        space = " "
        if self._escape_whitespaces:
            text = text.replace(" ", _SPACE)
            space = _SPACE
        if self._remove_extra_whitespaces:
            text = text.rstrip(space)
        return text

    def _merge(self, text: str, splits: dict[str, tuple[str, str]]) -> list[str]:
        symbols = list(text)
        count = len(symbols)
        if count == 0:
            return []
        previous = list(range(-1, count - 1))
        following = list(range(1, count + 1))
        following[-1] = -1
        agenda: list[tuple[float, int, int, int]] = []

        def push(left: int, right: int) -> None:
            if left < 0 or right < 0:
                return
            merged = symbols[left] + symbols[right]
            index = self._merge_pieces.get(merged)
            if index is not None:
                heapq.heappush(agenda, (-self.scores[index], left, right, len(merged)))

        for index in range(count - 1):
            push(index, index + 1)
        while agenda:
            _, left, right, size = heapq.heappop(agenda)
            if not symbols[left] or not symbols[right]:
                continue
            if len(symbols[left]) + len(symbols[right]) != size:
                continue
            splits[symbols[left] + symbols[right]] = (symbols[left], symbols[right])
            symbols[left] += symbols[right]
            symbols[right] = ""
            following[left] = following[right]
            if following[right] >= 0:
                previous[following[right]] = left
            push(previous[left], left)
            push(left, following[left])
        return [symbol for symbol in symbols if symbol]

    def _resegment(self, piece: str, splits: dict[str, tuple[str, str]], output: list[str]) -> None:
        index = self.piece_ids.get(piece)
        if index is None or self.types[index] != _UNUSED or piece not in splits:
            output.append(piece)
            return
        left, right = splits[piece]
        self._resegment(left, splits, output)
        self._resegment(right, splits, output)

    def _piece_to_ids(self, piece: str) -> list[int]:
        if piece in self.special_ids:
            return [self.special_ids[piece]]
        index = self.piece_ids.get(piece)
        if index is not None:
            return [index]
        if self.byte_fallback:
            return [self.piece_ids.get(f"<0x{byte:02X}>", self.unk_id) for byte in piece.encode()]
        return [self.unk_id]

    def _text_ids(self, text: str) -> list[int]:
        splits: dict[str, tuple[str, str]] = {}
        pieces: list[str] = []
        for piece in self._merge(self._normalize(text), splits):
            self._resegment(piece, splits, pieces)
        if not self.byte_fallback:
            pieces = self._fuse_unknown(pieces)
        return [identifier for piece in pieces for identifier in self._piece_to_ids(piece)]

    def _fuse_unknown(self, pieces: list[str]) -> list[str]:
        """Join each run of out-of-vocabulary characters into one unknown piece."""
        fused: list[str] = []
        previous_unknown = False
        for piece in pieces:
            unknown = piece not in self.piece_ids
            if unknown and previous_unknown:
                fused[-1] += piece
            else:
                fused.append(piece)
            previous_unknown = unknown
        return fused

    def tokenize(self, text: str) -> list[int]:
        """Token ids of ``text`` including the ``[gMASK] sop`` prefix."""
        return self.prefix_ids + self._text_ids(text)

    def __call__(
        self, texts: list[str] | tuple[str, ...], max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Left-padded ``(input_ids, attention_mask, position_ids)`` of width ``max_length``."""
        rows: list[list[int]] = []
        masks: list[list[bool]] = []
        positions: list[list[int]] = []
        for text in texts:
            ids = self.tokenize(text)[:max_length]
            padding = max_length - len(ids)
            rows.append([self.pad_id] * padding + ids)
            masks.append([False] * padding + [True] * len(ids))
            positions.append([0] * padding + list(range(len(ids))))
        return (
            torch.tensor(rows, dtype=torch.long),
            torch.tensor(masks, dtype=torch.bool),
            torch.tensor(positions, dtype=torch.long),
        )
