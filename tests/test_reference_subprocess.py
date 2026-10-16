import torch
from helpers import reference_in_subprocess


def _save_tensor(value, out_path):
    torch.save(torch.tensor(value), out_path)


def test_reference_in_subprocess_loads_saved_tensor(tmp_path):
    actual = reference_in_subprocess(
        "test_reference_subprocess._save_tensor", {"value": [1, 2, 3]}, tmp_path / "result.pt"
    )
    assert torch.equal(actual, torch.tensor([1, 2, 3]))
