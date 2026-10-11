"""Native byte-level BPE tokenizer for Llama 3 ``tokenizer.json`` files."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from imagehub.tokenization import bytes_to_unicode
from imagehub.tokenization_qwen import _pretokenize, _split_added

_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


class Llama3Tokenizer:
    """Llama 3 tokenizer (``add_special_tokens=True``), right-padded with the EOS token."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        path = Path(directory)
        data = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
        model = data["model"]
        if model["type"] != "BPE" or data.get("normalizer") is not None:
            raise ValueError("Llama 3 tokenizer requires a BPE model without a normalizer")
        pre_tokenizer = data["pre_tokenizer"]
        splits = [
            item for item in pre_tokenizer.get("pretokenizers", []) if item["type"] == "Split"
        ]
        if (
            pre_tokenizer["type"] != "Sequence"
            or len(splits) != 1
            or splits[0]["pattern"].get("Regex") != _SPLIT_PATTERN
            or splits[0]["behavior"] != "Isolated"
            or splits[0]["invert"]
            or pre_tokenizer["pretokenizers"][-1]["type"] != "ByteLevel"
        ):
            raise ValueError("Unsupported Llama 3 pre-tokenizer")
        self.encoder: dict[str, int] = model["vocab"]
        self.ignore_merges = bool(model.get("ignore_merges", False))
        merges = [
            tuple(merge) if isinstance(merge, list) else tuple(merge.split())
            for merge in model["merges"]
        ]
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self._cache: dict[str, list[str]] = {}
        self._added = {token["content"]: token["id"] for token in data.get("added_tokens", [])}

        self._prefix_ids: list[int] = []
        post_processor = data.get("post_processor") or {}
        for processor in post_processor.get("processors", [post_processor]):
            if processor.get("type") != "TemplateProcessing":
                continue
            before_sequence = True
            for item in processor["single"]:
                if "Sequence" in item:
                    before_sequence = False
                elif not before_sequence:
                    raise ValueError("Unsupported Llama 3 post-processor")
                else:
                    special = processor["special_tokens"][item["SpecialToken"]["id"]]
                    self._prefix_ids.extend(special["ids"])

        config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
        eos_token = config["eos_token"]
        if isinstance(eos_token, dict):
            eos_token = eos_token["content"]
        # The pipeline pads with the EOS token.
        self.pad_id = self._added.get(eos_token, self.encoder.get(eos_token))
        if self.pad_id is None:
            raise ValueError(f"EOS token {eos_token!r} is not in the vocabulary")
        self.model_max_length = int(config.get("model_max_length", 2**62))

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
        for piece in _pretokenize(text, digit_group=3):
            encoded = "".join(self.byte_encoder[byte] for byte in piece.encode("utf-8"))
            if self.ignore_merges and encoded in self.encoder:
                ids.append(self.encoder[encoded])
            else:
                ids.extend(self.encoder[token] for token in self._bpe(encoded))
        return ids

    def tokenize(self, text: str) -> list[int]:
        """Return the ids of ``encode(text, add_special_tokens=True)``."""
        ids = list(self._prefix_ids)
        for piece, added_id in _split_added(text, self._added):
            if added_id is not None:
                ids.append(added_id)
            elif piece:
                ids.extend(self._ordinary_ids(piece))
        return ids

    def __call__(
        self, texts: str | list[str], max_length: int = 128
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize, truncate and right-pad to ``max_length``; also return the attention mask."""
        if isinstance(texts, str):
            texts = [texts]
        max_length = min(max_length, self.model_max_length)
        rows: list[list[int]] = []
        masks: list[list[bool]] = []
        for text in texts:
            ids = self.tokenize(text)[:max_length]
            padding = max_length - len(ids)
            rows.append(ids + [self.pad_id] * padding)
            masks.append([True] * len(ids) + [False] * padding)
        return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)
