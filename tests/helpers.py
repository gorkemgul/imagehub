"""Shared test helpers: tiny tokenizer, checkpoint locations and difference metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from imagehub.tokenization import BOS, EOS, bytes_to_unicode

TINY_MERGES = (("h", "i</w>"), ("c", "a"), ("ca", "t</w>"))
TINY_VOCAB_SIZE = 517


def write_tiny_tokenizer(directory: str | os.PathLike[str]) -> Path:
    """Byte-level vocab (512 symbols) + 3 merges + BOS/EOS; ids: hi</w>=512, ca=513, cat</w>=514."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    chars = list(bytes_to_unicode().values())
    vocab = {c: i for i, c in enumerate(chars)} | {c + "</w>": 256 + i for i, c in enumerate(chars)}
    for a, b in TINY_MERGES:
        vocab[a + b] = len(vocab)
    vocab[BOS] = len(vocab)
    vocab[EOS] = len(vocab)
    assert len(vocab) == TINY_VOCAB_SIZE
    (path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    merges = "#version: 0.2\n" + "".join(f"{a} {b}\n" for a, b in TINY_MERGES)
    (path / "merges.txt").write_text(merges, encoding="utf-8")
    special = {"bos_token": BOS, "eos_token": EOS, "pad_token": EOS, "unk_token": EOS}
    (path / "special_tokens_map.json").write_text(json.dumps(special), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps(special | {"model_max_length": 77, "do_lower_case": True}), encoding="utf-8"
    )
    return path


def checkpoint_dir(version: int) -> Path:
    """Local sd-v1-<version> snapshot from IMAGEHUB_SD_V1_<version>, or skip the test."""
    value = os.environ.get(f"IMAGEHUB_SD_V1_{version}")
    if not value:
        pytest.skip(f"Set IMAGEHUB_SD_V1_{version} to a local sd-v1-{version} snapshot")
    return Path(value)


def max_diff(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    a = actual.detach().cpu().double()
    e = expected.detach().cpu().double()
    d = (a - e).abs()
    return {
        "max_abs": float(d.max()),
        "max_rel": float((d / e.abs().clamp_min(1e-12)).max()),
    }
