from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from imagehub.catalog import model_info
from imagehub.models.ovis_image_transformer import OvisImageTransformer2DModel


def test_ovis_image_catalog_is_pinned() -> None:
    spec = model_info("ovis-image-7b")
    assert spec.family == "ovis-image"
    assert spec.revision == "41be1c5821a92c970d63d7eb595a2fd3fe32b22e"
    assert "transformer/diffusion_pytorch_model-00002-of-00002.safetensors" in spec.files


def test_ovis_image_published_transformer_header() -> None:
    fixture = Path("tests/fixtures/ovis_image_configs")
    config = json.loads((fixture / "transformer_config.json").read_text())
    expected: dict[str, list[int]] = {}
    for filename in ("transformer_00001_tensors.json", "transformer_00002_tensors.json"):
        expected.update(json.loads((fixture / filename).read_text()))
    with torch.device("meta"):
        model = OvisImageTransformer2DModel(config)
    actual = model.state_dict()
    assert set(actual) == set(expected)
    assert {name: list(value.shape) for name, value in actual.items()} == expected


def test_ovis_image_tiny_transformer_forward() -> None:
    config = {
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 4,
        "num_attention_heads": 2,
        "joint_attention_dim": 6,
        "axes_dims_rope": [2, 2, 0],
    }
    torch.manual_seed(0)
    model = OvisImageTransformer2DModel(config)
    output = model(
        torch.randn(2, 6, 4),
        torch.randn(2, 3, 6),
        torch.tensor([0.5, 0.75]),
        torch.zeros(6, 3),
        torch.zeros(3, 3),
    )
    assert output.shape == (2, 6, 4)
    assert torch.isfinite(output).all()


@pytest.mark.reference
def test_ovis_image_tiny_transformer_matches_diffusers() -> None:
    diffusers = pytest.importorskip("diffusers")
    config = {
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 4,
        "num_attention_heads": 2,
        "joint_attention_dim": 6,
        "axes_dims_rope": (2, 2, 0),
    }
    torch.manual_seed(1)
    reference = diffusers.OvisImageTransformer2DModel(**config).eval()
    native = OvisImageTransformer2DModel(config).eval()
    native.load_state_dict(reference.state_dict())
    torch.manual_seed(2)
    hidden, context = torch.randn(2, 6, 4), torch.randn(2, 3, 6)
    timestep = torch.tensor([0.5, 0.75])
    image_ids, text_ids = torch.zeros(6, 3), torch.zeros(3, 3)
    expected = reference(hidden, context, timestep, image_ids, text_ids, return_dict=False)[0]
    assert torch.equal(native(hidden, context, timestep, image_ids, text_ids), expected)
