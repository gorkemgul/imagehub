import json
import struct

import pytest
import torch
from torch import nn

from imagehub.checkpoints import (
    NativeModel,
    load_safetensors,
    load_weights,
    save_safetensors,
)
from imagehub.exceptions import CheckpointError


class TinyModel(NativeModel):
    def __init__(self, config):
        super().__init__({"width": 3} | dict(config))
        self.linear = nn.Linear(self.config["width"], 2)

    def prepare_state_dict(self, state):
        state = dict(state)
        state.pop("legacy.buffer", None)
        return state


def tensors():
    return {
        "f32": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "bf16": torch.ones(4, dtype=torch.bfloat16),
        "i64": torch.tensor([1, -2], dtype=torch.int64),
        "flag": torch.tensor([True, False]),
        "empty": torch.zeros(0, 5),
    }


def test_safetensors_roundtrip(tmp_path):
    path = tmp_path / "w.safetensors"
    save_safetensors(tensors(), path)
    restored = load_safetensors(path)
    assert set(restored) == set(tensors())
    for name, value in tensors().items():
        assert restored[name].dtype == value.dtype
        torch.testing.assert_close(restored[name], value, rtol=0, atol=0)


def _raw(header: dict, payload: bytes) -> bytes:
    blob = json.dumps(header).encode()
    return struct.pack("<Q", len(blob)) + blob + payload


@pytest.mark.parametrize(
    ("header", "payload", "message"),
    [
        ({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, b"\0" * 4, "offsets"),
        ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, b"\0" * 8, "trailing"),
        (
            {
                "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                "b": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            },
            b"\0" * 4,
            "overlapping",
        ),
        ({"a": {"dtype": "X9", "shape": [1], "data_offsets": [0, 4]}}, b"\0" * 4, "descriptor"),
    ],
)
def test_safetensors_rejects_corruption(tmp_path, header, payload, message):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(_raw(header, payload))
    with pytest.raises(CheckpointError, match=message):
        load_safetensors(path)


def test_safetensors_rejects_bad_header_length(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", 10**9) + b"{}")
    with pytest.raises(CheckpointError, match="header length"):
        load_safetensors(path)


def test_load_weights_directory_order_and_bin(tmp_path):
    torch.save({"x": torch.ones(1)}, tmp_path / "diffusion_pytorch_model.bin")
    assert torch.equal(load_weights(tmp_path)["x"], torch.ones(1))
    save_safetensors({"x": torch.zeros(1)}, tmp_path / "diffusion_pytorch_model.safetensors")
    assert torch.equal(load_weights(tmp_path)["x"], torch.zeros(1))
    with pytest.raises(CheckpointError, match="No supported checkpoint"):
        load_weights(tmp_path / "missing_dir_is_file_not_dir")


class Unsafe:
    pass


def test_load_weights_refuses_arbitrary_pickles(tmp_path):
    torch.save({"x": torch.ones(1), "obj": Unsafe()}, tmp_path / "model.pt")
    with pytest.raises(CheckpointError, match="tensor-only"):
        load_weights(tmp_path / "model.pt")


def test_strict_loading_reports_mismatches():
    model = TinyModel({})
    good = {"linear.weight": torch.zeros(2, 3), "linear.bias": torch.zeros(2)}
    model.load_checkpoint(good | {"legacy.buffer": torch.zeros(1)})
    with pytest.raises(CheckpointError, match="Missing 1 tensors: linear.bias"):
        model.load_checkpoint({"linear.weight": torch.zeros(2, 3)})
    with pytest.raises(CheckpointError, match="Unexpected 1 tensors: extra"):
        model.load_checkpoint(good | {"extra": torch.zeros(1)})
    with pytest.raises(CheckpointError, match="shapes"):
        model.load_checkpoint(good | {"linear.bias": torch.zeros(3)})


def test_from_pretrained(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"width": 4, "_class_name": "Tiny"}))
    weights = {"linear.weight": torch.randn(2, 4), "linear.bias": torch.randn(2)}
    save_safetensors(weights, tmp_path / "model.safetensors")
    model = TinyModel.from_pretrained(tmp_path, dtype=torch.float64)
    assert not model.training
    assert model.linear.weight.dtype == torch.float64
    assert model.config["width"] == 4
    torch.testing.assert_close(model.linear.weight.float(), weights["linear.weight"])
