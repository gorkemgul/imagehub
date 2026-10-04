import json
import struct

import pytest
import torch
from torch import nn

from imagehub.checkpoints import (
    NativeModel,
    load_safetensors,
    load_weights,
    read_json,
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


def test_sharded_index_loads_and_merges(tmp_path):
    shard_one = {"alpha": torch.arange(4, dtype=torch.float32), "beta": torch.ones(2)}
    shard_two = {"gamma": torch.zeros(1, 3, dtype=torch.float64)}
    save_safetensors(shard_one, tmp_path / "model-00001-of-00002.safetensors")
    save_safetensors(shard_two, tmp_path / "model-00002-of-00002.safetensors")
    index = {
        "weight_map": {
            "alpha": "model-00001-of-00002.safetensors",
            "beta": "model-00001-of-00002.safetensors",
            "gamma": "model-00002-of-00002.safetensors",
        },
        "metadata": {"format": "pt"},
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    weights = load_weights(tmp_path)
    assert set(weights) == {"alpha", "beta", "gamma"}
    for name, value in shard_one.items():
        torch.testing.assert_close(weights[name], value, rtol=0, atol=0)
    torch.testing.assert_close(weights["gamma"], shard_two["gamma"], rtol=0, atol=0)


def test_sharded_index_follows_symlinked_blobs(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    shards = {
        "model-00001-of-00002.safetensors": {"one": torch.ones(2, 2)},
        "model-00002-of-00002.safetensors": {"two": torch.full((3,), 7.0, dtype=torch.float64)},
    }
    weight_map: dict[str, str] = {}
    for number, (filename, tensors) in enumerate(shards.items(), start=1):
        staging = snapshot / f"staging-{number}.safetensors"
        save_safetensors(tensors, staging)
        blob = blobs / f"blob{number}"
        blob.write_bytes(staging.read_bytes())
        staging.unlink()
        (snapshot / filename).symlink_to(blob)
        for name in tensors:
            weight_map[name] = filename
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    weights = load_weights(snapshot)
    for tensors in shards.values():
        for tensor_name, value in tensors.items():
            torch.testing.assert_close(weights[tensor_name], value, rtol=0, atol=0)


def test_sharded_index_rejects_escaping_shard_path(tmp_path):
    save_safetensors({"alpha": torch.zeros(1)}, tmp_path / "model-00001-of-00002.safetensors")
    index = {"weight_map": {"alpha": "../escape.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(CheckpointError, match="Invalid checkpoint shard path"):
        load_weights(tmp_path)


def test_sharded_index_rejects_duplicate_tensor_names(tmp_path):
    save_safetensors(
        {"shared": torch.zeros(1), "only_one": torch.zeros(1)}, tmp_path / "one.safetensors"
    )
    save_safetensors(
        {"shared": torch.ones(1), "only_two": torch.zeros(1)}, tmp_path / "two.safetensors"
    )
    index = {
        "weight_map": {
            "shared": "one.safetensors",
            "only_one": "one.safetensors",
            "only_two": "two.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(CheckpointError, match="Duplicate tensor names"):
        load_weights(tmp_path)


def test_sharded_index_must_match_shards(tmp_path):
    save_safetensors({"alpha": torch.zeros(1)}, tmp_path / "model.safetensors")
    index = {"weight_map": {"alpha": "model.safetensors", "ghost": "model.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(CheckpointError, match="index does not match"):
        load_weights(tmp_path)


@pytest.mark.parametrize("weight_map", [None, {"alpha": 1}])
def test_sharded_index_requires_valid_weight_map(tmp_path, weight_map):
    save_safetensors({"alpha": torch.zeros(1)}, tmp_path / "model.safetensors")
    index = {"weight_map": weight_map} if weight_map is not None else {}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(CheckpointError, match="weight_map"):
        load_weights(tmp_path)


@pytest.mark.parametrize(
    "header_bytes",
    [b"{not valid json  ", b"\xff\xfe\xfa\xfb\xfc\xfd\xfe\xff"],
)
def test_safetensors_rejects_invalid_header_bytes(tmp_path, header_bytes):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes)
    with pytest.raises(CheckpointError, match="Invalid safetensors header"):
        load_safetensors(path)


@pytest.mark.parametrize(
    "entry",
    [
        ["F32", [1], [0, 4]],
        {"dtype": 42, "shape": [1], "data_offsets": [0, 4]},
    ],
)
def test_safetensors_rejects_invalid_descriptor(tmp_path, entry):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(_raw({"a": entry}, b"\0" * 4))
    with pytest.raises(CheckpointError, match="Invalid tensor descriptor: a"):
        load_safetensors(path)


def test_truncated_pickle_raises_checkpoint_error(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"x": torch.ones(1)}, path)
    path.write_bytes(b"")
    with pytest.raises(CheckpointError, match="tensor-only"):
        load_weights(path)


def test_read_json_rejects_invalid_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not valid json")
    with pytest.raises(CheckpointError, match="Invalid JSON"):
        read_json(path)


def test_read_snapshot(tmp_path):
    from imagehub.checkpoints import SNAPSHOT_FILE, read_snapshot

    assert read_snapshot(tmp_path) is None
    (tmp_path / SNAPSHOT_FILE).write_text('{"model_id": "sd-v1-5"}')
    assert read_snapshot(tmp_path) == {"model_id": "sd-v1-5"}
    for bad in ("{not json", "[1, 2]"):
        (tmp_path / SNAPSHOT_FILE).write_text(bad)
        with pytest.raises(CheckpointError, match=SNAPSHOT_FILE):
            read_snapshot(tmp_path)


def test_pretrained_reexports_snapshot_helpers():
    from imagehub import checkpoints, pretrained

    assert pretrained.read_snapshot is checkpoints.read_snapshot
    assert pretrained.SNAPSHOT_FILE == checkpoints.SNAPSHOT_FILE
