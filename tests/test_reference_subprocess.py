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


def _save_tensor(value, out_path):
    torch.save(torch.tensor(value), out_path)


def test_reference_in_subprocess_loads_saved_tensor(tmp_path):
    actual = reference_in_subprocess(
        "test_reference_subprocess._save_tensor", {"value": [1, 2, 3]}, tmp_path / "result.pt"
    )
    assert torch.equal(actual, torch.tensor([1, 2, 3]))


def test_every_reference_subprocess_call_names_an_importable_module():
    import ast
    import importlib.util
    from pathlib import Path

    tests = Path(__file__).parent
    paths = []
    for source in sorted(tests.rglob("*.py")):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "reference_in_subprocess"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                paths.append((source.name, node.args[0].value))
    assert paths
    missing = [
        (name, path)
        for name, path in paths
        if importlib.util.find_spec(path.rsplit(".", 1)[0]) is None
    ]
    assert not missing
