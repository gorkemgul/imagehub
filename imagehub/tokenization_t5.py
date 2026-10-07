"""Native ``tokenizers``-compatible T5 Unigram tokenizer."""

from __future__ import annotations

import base64
import json
import os
import re
import struct
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch


class _PrecompiledNormalizer:
    """SentencePiece precompiled normalizer backed by its Darts double-array trie."""

    def __init__(self, encoded: str | bytes) -> None:
        data = base64.b64decode(encoded) if isinstance(encoded, str) else encoded
        trie_size = struct.unpack_from("<I", data)[0]
        if trie_size % 4:
            raise ValueError("invalid precompiled normalizer trie")
        self.units = struct.unpack_from(f"<{trie_size // 4}I", data, 4)
        self.normalized = data[4 + trie_size :]

    @staticmethod
    def _offset(unit: int) -> int:
        return (unit >> 10) << ((unit & (1 << 9)) >> 6)

    def _longest_prefix(self, source: bytes, start: int) -> tuple[int, int] | None:
        if not self.units:
            return None
        node = self._offset(self.units[0])
        best: tuple[int, int] | None = None
        position = start
        while position < len(source):
            byte = source[position]
            node ^= byte
            if node >= len(self.units):
                break
            unit = self.units[node]
            if unit & ((1 << 31) | 0xFF) != byte:
                break
            node ^= self._offset(unit)
            position += 1
            if unit & (1 << 8):
                value = self.units[node] & ((1 << 31) - 1)
                best = position, value
        return best

    def normalize(self, text: str) -> str:
        source = text.encode("utf-8")
        output = bytearray()
        position = 0
        while position < len(source):
            match = self._longest_prefix(source, position)
            if match is not None:
                position, value = match
                end = self.normalized.find(b"\0", value)
                if end < 0:
                    raise ValueError("invalid precompiled normalizer string table")
                output.extend(self.normalized[value:end])
                continue
            byte = source[position]
            width = 1 if byte < 0x80 else 2 if byte < 0xE0 else 3 if byte < 0xF0 else 4
            output.extend(source[position : position + width])
            position += width
        return output.decode("utf-8")


def _read_varint(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while position < len(data):
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, position
        shift += 7
        if shift >= 70:
            break
    raise ValueError("invalid protobuf varint")


def _protobuf_fields(data: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    position = 0
    while position < len(data):
        tag, position = _read_varint(data, position)
        field_number, wire_type = tag >> 3, tag & 7
        value: int | bytes
        if wire_type == 0:
            value, position = _read_varint(data, position)
        elif wire_type == 1:
            end = position + 8
            value, position = data[position:end], end
        elif wire_type == 2:
            size, position = _read_varint(data, position)
            end = position + size
            value, position = data[position:end], end
        elif wire_type == 5:
            end = position + 4
            value, position = data[position:end], end
        else:
            raise ValueError(f"unsupported protobuf wire type: {wire_type}")
        if position > len(data):
            raise ValueError("truncated protobuf field")
        yield field_number, wire_type, value


def _signed_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


@dataclass(frozen=True)
class _TrainerSpec:
    model_type: int = 1
    unk_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = -1
    byte_fallback: bool = False


@dataclass(frozen=True)
class _NormalizerSpec:
    name: str = ""
    precompiled_charsmap: bytes = b""
    add_dummy_prefix: bool = True
    remove_extra_whitespaces: bool = True
    escape_whitespaces: bool = True


@dataclass(frozen=True)
class _SentencePieceModel:
    pieces: list[tuple[str, float, int]]
    trainer_spec: _TrainerSpec
    normalizer_spec: _NormalizerSpec


def _parse_trainer_spec(data: bytes) -> _TrainerSpec:
    values: dict[str, Any] = {
        "model_type": 1,
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": -1,
        "byte_fallback": False,
    }
    fields = {
        3: "model_type",
        35: "byte_fallback",
        40: "unk_id",
        41: "bos_id",
        42: "eos_id",
        43: "pad_id",
    }
    for field_number, wire_type, value in _protobuf_fields(data):
        name = fields.get(field_number)
        if name is None or wire_type != 0 or not isinstance(value, int):
            continue
        values[name] = bool(value) if name == "byte_fallback" else _signed_int32(value)
    return _TrainerSpec(**values)


def _parse_normalizer_spec(data: bytes) -> _NormalizerSpec:
    values: dict[str, Any] = {
        "name": "",
        "precompiled_charsmap": b"",
        "add_dummy_prefix": True,
        "remove_extra_whitespaces": True,
        "escape_whitespaces": True,
    }
    fields = {
        1: "name",
        2: "precompiled_charsmap",
        3: "add_dummy_prefix",
        4: "remove_extra_whitespaces",
        5: "escape_whitespaces",
    }
    for field_number, wire_type, value in _protobuf_fields(data):
        name = fields.get(field_number)
        if name is None:
            continue
        if field_number == 1 and wire_type == 2 and isinstance(value, bytes):
            values[name] = value.decode("utf-8")
        elif field_number == 2 and wire_type == 2 and isinstance(value, bytes):
            values[name] = value
        elif field_number >= 3 and wire_type == 0 and isinstance(value, int):
            values[name] = bool(value)
    return _NormalizerSpec(**values)


def _parse_piece(data: bytes) -> tuple[str, float, int]:
    piece = ""
    score = 0.0
    piece_type = 1
    for field_number, wire_type, value in _protobuf_fields(data):
        if field_number == 1 and wire_type == 2 and isinstance(value, bytes):
            piece = value.decode("utf-8")
        elif field_number == 2 and wire_type == 5 and isinstance(value, bytes):
            score = struct.unpack("<f", value)[0]
        elif field_number == 3 and wire_type == 0 and isinstance(value, int):
            piece_type = value
    return piece, score, piece_type


def _load_sentencepiece(path: Path) -> _SentencePieceModel:
    pieces: list[tuple[str, float, int]] = []
    trainer_spec = _TrainerSpec()
    normalizer_spec = _NormalizerSpec()
    for field_number, wire_type, value in _protobuf_fields(path.read_bytes()):
        if wire_type != 2 or not isinstance(value, bytes):
            continue
        if field_number == 1:
            pieces.append(_parse_piece(value))
        elif field_number == 2:
            trainer_spec = _parse_trainer_spec(value)
        elif field_number == 3:
            normalizer_spec = _parse_normalizer_spec(value)
    if not pieces:
        raise ValueError("SentencePiece model has no pieces")
    return _SentencePieceModel(pieces, trainer_spec, normalizer_spec)


class _Unigram:
    def __init__(self, config: dict[str, Any]) -> None:
        vocab = config["vocab"]
        self.pieces = [str(entry[0]) for entry in vocab]
        self.scores = [float(entry[1]) for entry in vocab]
        self.unk_id = int(config["unk_id"])
        self.unk_score = min(self.scores) - 10.0
        self.byte_fallback = bool(config.get("byte_fallback", False))
        self.byte_ids = {piece: index for index, piece in enumerate(self.pieces)}
        self.trie: dict[object, Any] = {}
        for token_id, piece in enumerate(self.pieces):
            node = self.trie
            for character in piece:
                node = node.setdefault(character, {})
            node[None] = token_id

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        size = len(text)
        best = [float("-inf")] * (size + 1)
        previous = [-1] * (size + 1)
        token_ids = [self.unk_id] * (size + 1)
        best[0] = 0.0
        for start in range(size):
            if best[start] == float("-inf"):
                continue
            node = self.trie
            has_single = False
            for end in range(start, size):
                child = node.get(text[end])
                if child is None:
                    break
                node = child
                token_id = node.get(None)
                if token_id is None:
                    continue
                if end == start:
                    has_single = True
                score = best[start] + self.scores[token_id]
                if score > best[end + 1]:
                    best[end + 1] = score
                    previous[end + 1] = start
                    token_ids[end + 1] = token_id
            if not has_single:
                end = start + 1
                score = best[start] + self.unk_score
                if score > best[end]:
                    best[end] = score
                    previous[end] = start
                    token_ids[end] = self.unk_id

        nodes: list[tuple[int, str]] = []
        end = size
        while end > 0:
            start = previous[end]
            if start < 0:
                raise ValueError("Unigram vocabulary cannot encode input")
            nodes.append((token_ids[end], text[start:end]))
            end = start
        nodes.reverse()

        output: list[int] = []
        for token_id, piece in nodes:
            if token_id != self.unk_id:
                output.append(token_id)
            elif self.byte_fallback:
                output.extend(
                    self.byte_ids.get(f"<0x{byte:02X}>", self.unk_id)
                    for byte in piece.encode("utf-8")
                )
            elif not output or output[-1] != self.unk_id:
                output.append(self.unk_id)
        return output


def _content(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


class T5Tokenizer:
    """Tokenizer loaded from a Hugging Face T5 tokenizer directory."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        path = Path(directory)
        tokenizer_config: dict[str, Any] = {}
        tokenizer_config_path = path / "tokenizer_config.json"
        if tokenizer_config_path.is_file():
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))

        tokenizer_json_path = path / "tokenizer.json"
        if tokenizer_json_path.is_file():
            config: dict[str, Any] = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
        else:
            config = self._config_from_sentencepiece(path / "spiece.model", tokenizer_config)
        if config["model"]["type"] != "Unigram":
            raise ValueError("T5Tokenizer requires a Unigram model")
        self.model = _Unigram(config["model"])

        self._normalizers = self._load_normalizers(config.get("normalizer"))
        pre_tokenizer = config["pre_tokenizer"]
        if pre_tokenizer["type"] != "Metaspace":
            raise ValueError("T5Tokenizer requires a Metaspace pre-tokenizer")
        self.replacement = str(pre_tokenizer["replacement"])
        self.add_prefix_space = bool(pre_tokenizer.get("add_prefix_space", True))

        added = config.get("added_tokens", [])
        self.added_ids = {str(item["content"]): int(item["id"]) for item in added}
        raw = [str(item["content"]) for item in added if not item.get("normalized", False)]
        normalized = [str(item["content"]) for item in added if item.get("normalized", False)]
        self._raw_added = self._added_pattern(raw)
        self._normalized_added = self._added_pattern(normalized)

        self.post_processor = self._load_post_processor(config.get("post_processor"))
        pad_token = _content(tokenizer_config.get("pad_token")) or "<pad>"
        pad_id = self.added_ids.get(pad_token, self.model.byte_ids.get(pad_token))
        if pad_id is None:
            raise ValueError("tokenizer has no configured padding token")
        self.pad_id = pad_id

    @staticmethod
    def _config_from_sentencepiece(
        model_path: Path, tokenizer_config: dict[str, Any]
    ) -> dict[str, Any]:
        model = _load_sentencepiece(model_path)
        if model.trainer_spec.model_type != 1:
            raise ValueError("T5Tokenizer requires a SentencePiece Unigram model")
        normalizers: list[dict[str, Any]] = []
        if model.normalizer_spec.precompiled_charsmap:
            normalizers.append(
                {
                    "type": "Precompiled",
                    "precompiled_charsmap": base64.b64encode(
                        model.normalizer_spec.precompiled_charsmap
                    ).decode("ascii"),
                }
            )
        add_prefix_space = bool(tokenizer_config.get("add_prefix_space", True))
        legacy = bool(tokenizer_config.get("legacy", True))
        prepend = add_prefix_space and legacy
        added_decoder = tokenizer_config.get("added_tokens_decoder", {})
        added_tokens = [
            {"id": int(token_id), **item}
            for token_id, item in added_decoder.items()
            if isinstance(item, dict)
        ]
        eos_id = model.trainer_spec.eos_id
        eos_token = _content(tokenizer_config.get("eos_token")) or "</s>"
        return {
            "model": {
                "type": "Unigram",
                "vocab": [(piece, score) for piece, score, _ in model.pieces],
                "unk_id": model.trainer_spec.unk_id,
                "byte_fallback": False,
            },
            "normalizer": {"type": "Sequence", "normalizers": normalizers},
            "pre_tokenizer": {
                "type": "Metaspace",
                "replacement": "▁",
                "add_prefix_space": prepend,
            },
            "added_tokens": added_tokens,
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": eos_token, "type_id": 0}},
                ],
                "special_tokens": {eos_token: {"ids": [eos_id], "tokens": [eos_token]}},
            },
        }

    @staticmethod
    def _added_pattern(tokens: list[str]) -> re.Pattern[str] | None:
        if not tokens:
            return None
        ordered = sorted(tokens, key=len, reverse=True)
        return re.compile("(" + "|".join(re.escape(token) for token in ordered) + ")")

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

    def _load_post_processor(self, config: dict[str, Any] | None) -> list[int | None]:
        if config is None:
            return [None]
        if config["type"] != "TemplateProcessing":
            raise ValueError("T5Tokenizer requires a TemplateProcessing post-processor")
        result: list[int | None] = []
        specials = config["special_tokens"]
        for item in config["single"]:
            if "Sequence" in item:
                result.append(None)
            else:
                token = item["SpecialToken"]["id"]
                result.extend(int(token_id) for token_id in specials[token]["ids"])
        return result

    def _encode_plain(self, text: str) -> list[int]:
        if not text:
            return []
        text = "".join(self.replacement if character.isspace() else character for character in text)
        if self.add_prefix_space and not text.startswith(self.replacement):
            text = self.replacement + text
        output: list[int] = []
        prefixed = text.startswith(self.replacement)
        for index, piece in enumerate(text.split(self.replacement)):
            if piece:
                prefix = self.replacement if prefixed or index > 0 else ""
                output.extend(self.model.encode(prefix + piece))
        return output

    def _encode_normalized_segment(self, text: str) -> list[int]:
        if self._normalized_added is None:
            return self._encode_plain(text)
        output: list[int] = []
        for index, piece in enumerate(self._normalized_added.split(text)):
            if index % 2:
                output.append(self.added_ids[piece])
            else:
                output.extend(self._encode_plain(piece))
        return output

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        pieces = self._raw_added.split(text) if self._raw_added is not None else [text]
        for index, piece in enumerate(pieces):
            if index % 2:
                ids.append(self.added_ids[piece])
            else:
                ids.extend(self._encode_normalized_segment(self._normalize(piece)))
        output: list[int] = []
        for token_id in self.post_processor:
            if token_id is None:
                output.extend(ids)
            else:
                output.append(token_id)
        return output

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
                special_count = sum(token_id is not None for token_id in self.post_processor)
                content_length = max_length - special_count
                content = ids[:content_length]
                suffix = [int(token_id) for token_id in self.post_processor if token_id is not None]
                ids = content + suffix
            padding = max_length - len(ids)
            rows.append(ids + [self.pad_id] * padding)
            masks.append([True] * len(ids) + [False] * padding)
        input_ids = torch.tensor(rows, dtype=torch.long)
        return input_ids, torch.tensor(masks, dtype=torch.bool)
