import json

import pytest
import torch
from fixture_files import fixture_path

from imagehub.models.lumina2_transformer import Lumina2Transformer2DModel

FIXTURES = fixture_path("lumina2_configs")
TINY = {
    "in_channels": 4,
    "hidden_size": 32,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "num_attention_heads": 4,
    "num_kv_heads": 2,
    "multiple_of": 16,
    "axes_dim_rope": [2, 2, 4],
    "axes_lens": [32, 16, 16],
    "cap_feat_dim": 12,
}


def test_matches_published_tensor_layout():
    config = json.loads((FIXTURES / "transformer_config.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURES / "transformer_tensors.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = Lumina2Transformer2DModel(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == expected


def test_tiny_forward_shape_and_mask():
    torch.manual_seed(0)
    model = Lumina2Transformer2DModel(TINY).eval()
    hidden = torch.randn(2, 4, 6, 10)
    text = torch.randn(2, 5, 12)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
    timestep = torch.tensor([0.25, 0.75])
    with torch.inference_mode():
        out = model(hidden, timestep, text, mask)
        full = model(hidden, timestep, text, torch.ones(2, 5, dtype=torch.long))
    assert out.shape == hidden.shape
    assert not torch.equal(out[1], full[1])


@pytest.mark.parametrize(
    "override",
    [
        {"scaling_factor": 2.0},
        {"axes_dim_rope": [2, 2]},
        {"axes_dim_rope": [4, 4, 4]},
        {"num_kv_heads": 3},
    ],
)
def test_unsupported_option_rejected(override):
    with pytest.raises(ValueError, match="Unsupported Lumina2 transformer setting"):
        Lumina2Transformer2DModel(TINY | override)


def test_unknown_option_rejected():
    with pytest.raises(ValueError, match="Unsupported Lumina2 transformer setting bogus"):
        Lumina2Transformer2DModel(TINY | {"bogus": 1})
