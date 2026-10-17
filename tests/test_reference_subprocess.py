"""Tests for isolated reference-process execution."""

from pathlib import Path

import torch
from helpers import reference_in_subprocess


def _save_reference_result(*, value: int, output: str | Path) -> None:
    torch.save(torch.tensor([value]), output)


def test_reference_in_subprocess_saves_and_loads_result(tmp_path):
    actual = reference_in_subprocess(
        "test_reference_subprocess._save_reference_result",
        {"value": 17},
        tmp_path / "result.pt",
    )
    assert torch.equal(actual, torch.tensor([17]))
