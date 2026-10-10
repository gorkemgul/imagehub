"""Tensor-header check of every published SD 1.x and SDXL snapshot without loading weights.

For each component the model is built on the meta device from the snapshot's config.json, and
the checkpoint's tensor names and shapes are compared with the model's state dict after
``prepare_state_dict``. Safetensors files are read header-only; pickle files are opened with
``torch.load(map_location="meta")``, which creates meta tensors and never reads the storage.
"""

import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import checkpoint_dir, snapshot_dir

from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel, CLIPTextModelWithProjection
from imagehub.models.qwen3 import Qwen3Model
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.models.z_image_transformer import ZImageTransformer2DModel

pytestmark = pytest.mark.reference

COMPONENTS = {
    "text_encoder": CLIPTextModel,
    "unet": UNet2DConditionModel,
    "vae": AutoencoderKL,
}
SDXL_COMPONENTS = COMPONENTS | {"text_encoder_2": CLIPTextModelWithProjection}
WEIGHT_FILES = (
    "model.safetensors",
    "diffusion_pytorch_model.safetensors",
    "pytorch_model.bin",
    "diffusion_pytorch_model.bin",
)
# Overlay entries (a base snapshot with a replaced U-Net) check their headers in their own
# test files.
SDXL_MODEL_IDS = tuple(
    spec.id
    for spec in MODEL_CATALOG.values()
    if spec.family == "stable-diffusion-xl" and spec.base is None and spec.id != "sdxl-refiner-1.0"
)
SDXL_FIXTURES = {
    "playground-v2.5": "sdxl_tensor_shapes.json",
    "sdxl-base-1.0": "sdxl_tensor_shapes.json",
    "ssd-1b": "ssd1b_tensor_shapes.json",
    "sdxl-turbo": "sdxl_tensor_shapes.json",
}
Z_IMAGE_COMPONENTS = {
    "text_encoder": Qwen3Model,
    "transformer": ZImageTransformer2DModel,
    "vae": AutoencoderKL,
}


def checkpoint_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    path = next((directory / n for n in WEIGHT_FILES if (directory / n).is_file()), None)
    assert path is not None, f"no weights in {directory}"
    if path.suffix == ".safetensors":
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        return {n: tuple(e["shape"]) for n, e in header.items() if n != "__metadata__"}
    state = torch.load(path, map_location="meta", weights_only=True)
    assert all(t.device.type == "meta" for t in state.values())
    return {name: tuple(t.shape) for name, t in state.items()}


def _header_result(root, components, published=None):
    result = {}
    for component, cls in components.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        shapes = checkpoint_shapes(root / component)
        if published is not None:
            expected_published = {
                name: tuple(entry["shape"]) for name, entry in published[component].items()
            }
            assert shapes == expected_published
        prepared = model.prepare_state_dict(
            {n: torch.empty(s, device="meta") for n, s in shapes.items()}
        )
        expected = {n: tuple(t.shape) for n, t in model.state_dict().items()}
        got = {n: tuple(t.shape) for n, t in prepared.items()}
        result[component] = {
            "missing": sorted(expected.keys() - got.keys()),
            "extra": sorted(got.keys() - expected.keys()),
            "mismatch": sorted(
                f"{n}: checkpoint {got[n]} != model {expected[n]}"
                for n in expected.keys() & got.keys()
                if got[n] != expected[n]
            ),
            "tensors": len(got),
        }
    return result


def _assert_headers_match(result):
    for component, entry in result.items():
        assert not entry["missing"], (component, entry["missing"][:5])
        assert not entry["extra"], (component, entry["extra"][:5])
        assert not entry["mismatch"], (component, entry["mismatch"][:5])


def _sharded_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    index = next(directory.glob("*.safetensors.index.json"))
    weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    assert isinstance(weight_map, dict)
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in sorted(set(weight_map.values())):
        with (directory / shard).open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes |= {n: tuple(e["shape"]) for n, e in header.items() if n != "__metadata__"}
    assert set(shapes) == set(weight_map)
    return shapes


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_published_tensor_headers_match_models(version, record):
    result = _header_result(checkpoint_dir(version), COMPONENTS)
    record(f"header_sd-v1-{version}", **result)
    _assert_headers_match(result)


@pytest.mark.parametrize("model_id", SDXL_MODEL_IDS)
def test_sdxl_published_tensor_headers_match_models(model_id, record):
    fixture = Path(__file__).parents[1] / "fixtures" / SDXL_FIXTURES[model_id]
    published = json.loads(fixture.read_text(encoding="utf-8"))
    result = _header_result(snapshot_dir(model_id), SDXL_COMPONENTS, published)
    record(f"header_{model_id}", **result)
    _assert_headers_match(result)


@pytest.mark.parametrize("model_id", ["z-image-turbo", "z-image"])
def test_z_image_published_tensor_headers_match_models(model_id, record):
    root = snapshot_dir(model_id)
    result = {}
    for component, cls in Z_IMAGE_COMPONENTS.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        if component in ("text_encoder", "transformer"):
            shapes = _sharded_shapes(root / component)
        else:
            shapes = checkpoint_shapes(root / component)
        prepared = model.prepare_state_dict(
            {n: torch.empty(s, device="meta") for n, s in shapes.items()}
        )
        expected = {n: tuple(t.shape) for n, t in model.state_dict().items()}
        got = {n: tuple(t.shape) for n, t in prepared.items()}
        result[component] = {
            "missing": sorted(expected.keys() - got.keys()),
            "extra": sorted(got.keys() - expected.keys()),
            "mismatch": sorted(n for n in expected.keys() & got.keys() if expected[n] != got[n]),
            "tensors": len(shapes),
        }
    record(f"header_{model_id}", **result)
    _assert_headers_match(result)
