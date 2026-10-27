import json
from pathlib import Path

import pytest
import torch
from fixture_files import fixture_path

from imagehub.backends.hunyuan_dit import (
    HunyuanDiTBackend,
    get_add_time_ids,
    get_resize_crop_region_for_grid,
    map_to_standard_shapes,
)
from imagehub.catalog import model_info
from imagehub.checkpoints import read_json
from imagehub.models.hunyuan_dit_transformer import HunyuanDiT2DModel
from imagehub.types import T2IRequest


def test_catalog_defaults_and_backend_defaults():
    spec = model_info("hunyuan-dit-v1.1-distilled")
    assert spec.family == "hunyuan-dit"
    assert spec.config == {"defaults": {"steps": 25, "guidance": 5.0}}
    backend = HunyuanDiTBackend(Path("missing"), model_id=spec.id)
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert options["steps"] == 25
    assert options["guidance"] == 5.0
    assert options["width"] == 1024
    assert options["height"] == 1024
    assert options["use_resolution_binning"] is True
    family_options = HunyuanDiTBackend(Path("missing")).validate_options(T2IRequest("a cat"), {})
    assert (family_options["steps"], family_options["guidance"]) == (25, 5.0)


@pytest.mark.parametrize(
    ("model_id", "steps"),
    [("hunyuan-dit-v1.2", 50), ("hunyuan-dit-v1.2-distilled", 25)],
)
def test_v12_catalog_defaults(model_id, steps):
    backend = HunyuanDiTBackend(Path("missing"), model_id=model_id)
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert (options["steps"], options["guidance"]) == (steps, 5.0)
    assert (options["width"], options["height"]) == (1024, 1024)


@pytest.mark.parametrize("variant", ["", "_distilled"])
def test_v12_published_headers_load_on_meta(variant):
    fixtures = fixture_path("hunyuan_dit_v12_configs")
    config = read_json(fixtures / "transformer_config.json")
    assert config["use_style_cond_and_image_meta_size"] is False
    shapes = json.loads((fixtures / f"transformer{variant}_tensors.json").read_text())
    with torch.device("meta"):
        model = HunyuanDiT2DModel(config)
    assert "time_extra_emb.style_embedder.weight" not in shapes
    model.load_checkpoint(
        {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
    )
    assert len(shapes) == len(model.state_dict()) == 1544


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 1001, "use_resolution_binning": False}, ValueError, "multiple of 16"),
        ({"steps": 0}, ValueError, "steps"),
        ({"steps": 1001}, ValueError, "at most 1000"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"guidance_rescale": float("inf")}, ValueError, "guidance_rescale"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"seed": -1}, ValueError, "seed"),
        ({"use_resolution_binning": 1}, TypeError, "use_resolution_binning"),
        ({"latents": torch.zeros(1, 4, 64, 64)}, ValueError, "shape"),
    ],
)
def test_invalid_options_fail_before_loading(options, error, match):
    backend = HunyuanDiTBackend(Path("missing"))
    with pytest.raises(error, match=match):
        backend.validate_options(T2IRequest("a cat"), options)
    assert not backend.is_loaded


def test_resolution_binning_matches_reference_shapes():
    assert map_to_standard_shapes(1000, 800) == (1024, 768)
    assert map_to_standard_shapes(800, 1000) == (768, 1024)
    backend = HunyuanDiTBackend(Path("missing"))
    options = backend.validate_options(T2IRequest("a cat"), {"width": 1000, "height": 800})
    assert (options["width"], options["height"]) == (1024, 768)


def test_add_time_ids_match_reference_order():
    actual = get_add_time_ids(768, 1024, torch.float32, torch.device("cpu"), batch_size=2)
    assert actual.dtype == torch.float32
    assert actual.tolist() == [
        [1024.0, 1024.0, 768.0, 1024.0, 0.0, 0.0],
        [1024.0, 1024.0, 768.0, 1024.0, 0.0, 0.0],
    ]


def test_crop_region_matches_reference():
    assert get_resize_crop_region_for_grid((48, 64), 32) == ((4, 0), (28, 32))
    assert get_resize_crop_region_for_grid((64, 48), 32) == ((0, 4), (32, 28))
