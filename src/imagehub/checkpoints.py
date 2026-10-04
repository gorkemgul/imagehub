"""Local, strict checkpoint loading without transformers, diffusers or safetensors."""

from __future__ import annotations

import ctypes
import json
import math
import os
import pickle
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor, nn

from imagehub.exceptions import CheckpointError

_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
_NAMES = {dtype: name for name, dtype in _DTYPES.items()}
WEIGHT_FILES = (
    "model.safetensors",
    "diffusion_pytorch_model.safetensors",
    "model.pt",
    "pytorch_model.bin",
    "diffusion_pytorch_model.bin",
)
_INDEX_FILES = (
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "diffusion_pytorch_model.bin.index.json",
)


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    file_path = Path(path)
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"Invalid JSON in {file_path}") from error
    if not isinstance(value, dict):
        raise CheckpointError(f"Expected a JSON object in {file_path}")
    return value


SNAPSHOT_FILE = "imagehub_snapshot.json"


def read_snapshot(directory: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Return the download manifest of a snapshot, or None when it has none."""
    path = Path(directory) / SNAPSHOT_FILE
    if not path.is_file():
        return None
    return read_json(path)


def load_safetensors(path: str | os.PathLike[str]) -> dict[str, Tensor]:
    """Read the documented safetensors format using byte buffers and torch.frombuffer."""
    result: dict[str, Tensor] = {}
    with Path(path).open("rb") as source:
        size = os.fstat(source.fileno()).st_size
        prefix = source.read(8)
        if len(prefix) != 8:
            raise CheckpointError("Truncated safetensors header")
        header_length = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_length <= min(100_000_000, size - 8):
            raise CheckpointError("Invalid safetensors header length")
        try:
            header = json.loads(source.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CheckpointError("Invalid safetensors header") from error
        if not isinstance(header, dict):
            raise CheckpointError("Invalid safetensors header")
        payload_start = 8 + header_length
        spans = []
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("dtype"), str):
                raise CheckpointError(f"Invalid tensor descriptor: {name}")
            dtype = _DTYPES.get(entry["dtype"])
            shape, offsets = entry.get("shape"), entry.get("data_offsets")
            if dtype is None or not isinstance(shape, list) or not isinstance(offsets, list):
                raise CheckpointError(f"Invalid tensor descriptor: {name}")
            if any(type(n) is not int or n < 0 for n in shape) or len(offsets) != 2:
                raise CheckpointError(f"Invalid tensor dimensions: {name}")
            start, end = offsets
            expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end - start != expected
                or payload_start + end > size
            ):
                raise CheckpointError(f"Invalid tensor offsets: {name}")
            spans.append((start, end, name, dtype, shape))
        cursor = 0
        for start, end, name, dtype, shape in sorted(spans):
            if start != cursor:
                raise CheckpointError("Safetensors payload has overlapping or missing data")
            source.seek(payload_start + start)
            buffer = bytearray(source.read(end - start))
            result[name] = (
                torch.frombuffer(buffer, dtype=dtype).reshape(shape)
                if buffer
                else torch.empty(shape, dtype=dtype)
            )
            cursor = end
        if payload_start + cursor != size:
            raise CheckpointError("Safetensors payload has unreferenced trailing data")
    return result


def save_safetensors(tensors: Mapping[str, Tensor], path: str | os.PathLike[str]) -> None:
    """Write tensors in the safetensors format (used for tests and converted snapshots)."""
    header: dict[str, Any] = {}
    payloads: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        if tensor.dtype not in _NAMES:
            raise TypeError(f"Unsupported dtype for safetensors: {tensor.dtype}")
        nbytes = tensor.numel() * tensor.element_size()
        raw = ctypes.string_at(tensor.data_ptr(), nbytes) if nbytes else b""
        header[name] = {
            "dtype": _NAMES[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        payloads.append(raw)
        offset += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with Path(path).open("wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for raw in payloads:
            out.write(raw)


def _load_pickle(source: Path) -> dict[str, Tensor]:
    try:
        state = torch.load(source, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        raise CheckpointError(
            f"{source} is not a tensor-only checkpoint and was not loaded: {error}"
        ) from error
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not all(
        isinstance(name, str) and isinstance(value, Tensor) for name, value in state.items()
    ):
        raise CheckpointError(f"{source} must contain a tensor-only state dictionary")
    return state


def load_weights(path: str | os.PathLike[str]) -> dict[str, Tensor]:
    """Read safetensors, tensor-only PT/BIN, or a (sharded) model directory."""
    source = Path(path)
    if source.is_dir():
        for name in _INDEX_FILES:
            index_path = source / name
            if index_path.is_file():
                weight_map = read_json(index_path).get("weight_map")
                if not isinstance(weight_map, dict) or not all(
                    isinstance(tensor_name, str) and isinstance(shard_name, str)
                    for tensor_name, shard_name in weight_map.items()
                ):
                    raise CheckpointError(f"Invalid weight_map in {index_path}")
                result: dict[str, Tensor] = {}
                for shard_name in sorted(set(weight_map.values())):
                    if (
                        not shard_name
                        or shard_name == "."
                        or Path(shard_name).is_absolute()
                        or ".." in Path(shard_name).parts
                    ):
                        raise CheckpointError(
                            f"Invalid checkpoint shard path {shard_name!r} in {index_path}"
                        )
                    shard = source / shard_name
                    if not shard.is_file():
                        raise CheckpointError(
                            f"Checkpoint shard {shard} listed in {index_path} is missing"
                        )
                    values = load_weights(shard)
                    if result.keys() & values.keys():
                        raise CheckpointError("Duplicate tensor names across checkpoint shards")
                    result.update(values)
                if set(result) != set(weight_map):
                    raise CheckpointError("Checkpoint index does not match its shards")
                return result
        for name in WEIGHT_FILES:
            if (source / name).is_file():
                return load_weights(source / name)
        raise CheckpointError(f"No supported checkpoint found in {source}")
    if not source.is_file():
        raise CheckpointError(f"No supported checkpoint found at {source}")
    if source.suffix == ".safetensors":
        return load_safetensors(source)
    return _load_pickle(source)


def _describe(missing: list[str], unexpected: list[str]) -> str:
    lines = ["Checkpoint does not match the model architecture."]
    if missing:
        lines.append(f"Missing {len(missing)} tensors: " + ", ".join(missing[:20]))
    if unexpected:
        lines.append(f"Unexpected {len(unexpected)} tensors: " + ", ".join(unexpected[:20]))
    return "\n".join(lines)


class NativeModel(nn.Module):
    """Base for native model definitions; checkpoints are always loaded strictly."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Override only for documented key conversion or constant buffers to drop."""
        return state

    def load_checkpoint(self, state: Mapping[str, Tensor]) -> None:
        prepared = self.prepare_state_dict(dict(state))
        expected = self.state_dict()
        missing = sorted(set(expected) - set(prepared))
        unexpected = sorted(set(prepared) - set(expected))
        if missing or unexpected:
            raise CheckpointError(_describe(missing, unexpected))
        mismatched = [
            f"{name}: checkpoint {tuple(prepared[name].shape)} != model {tuple(value.shape)}"
            for name, value in expected.items()
            if tuple(prepared[name].shape) != tuple(value.shape)
        ]
        if mismatched:
            raise CheckpointError(
                "Checkpoint tensor shapes do not match the model:\n  "
                + "\n  ".join(mismatched[:20])
            )
        self.load_state_dict(prepared, strict=True, assign=True)

    @classmethod
    def from_pretrained(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> Self:
        path = Path(directory)
        config = read_json(path / "config.json")
        with torch.device("meta"):
            model = cls(config)
        model.load_checkpoint(load_weights(path))
        model.to(device=device, dtype=dtype).eval()
        return model
