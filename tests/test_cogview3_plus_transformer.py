import json
from pathlib import Path

import pytest
import torch

from imagehub.models.cogview3_plus_transformer import CogView3PlusTransformer2DModel

CONFIGS = Path(__file__).parent / "fixtures" / "cogview3_plus_configs"
TINY = {
    "patch_size": 2,
    "in_channels": 4,
    "out_channels": 4,
    "num_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "text_embed_dim": 12,
    "time_embed_dim": 16,
    "condition_dim": 4,
    "pos_embed_max_size": 8,
    "sample_size": 8,
}


def test_matches_published_tensor_layout():
    config = json.loads((CONFIGS / "transformer_config.json").read_text(encoding="utf-8"))
    expected = json.loads((CONFIGS / "transformer_tensors.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = CogView3PlusTransformer2DModel(config)
    assert {name: list(value.shape) for name, value in model.state_dict().items()} == expected
    assert len(expected) == 436


def test_position_buffer_survives_meta_construction():
    with torch.device("meta"):
        model = CogView3PlusTransformer2DModel(TINY)
    assert model.patch_embed.pos_embed.device.type == "meta"
    assert "pos_embed" not in model.state_dict()
    model.init_non_persistent_buffers()
    assert model.patch_embed.pos_embed.shape == (8, 8, 16)
    assert torch.isfinite(model.patch_embed.pos_embed).all()
    model.to(torch.float64)
    assert model.patch_embed.pos_embed.dtype == torch.float64


def test_forward_shape_and_odd_patch_grid():
    model = CogView3PlusTransformer2DModel(TINY).eval()
    sample = torch.randn(2, 4, 6, 10)
    out = model(
        sample,
        torch.randn(2, 5, 12),
        torch.tensor([3, 700]),
        torch.tensor([[64.0, 64.0]] * 2),
        torch.tensor([[64.0, 64.0]] * 2),
        torch.zeros(2, 2),
    )
    assert out.shape == sample.shape
    with pytest.raises(ValueError, match="divisible"):
        model(
            torch.randn(1, 4, 5, 6),
            torch.randn(1, 5, 12),
            torch.tensor([3]),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
        )


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        CogView3PlusTransformer2DModel(TINY | {"rope_axes_dim": [1, 1]})
