import json

import pytest
import torch
from fixture_files import fixture_path

from imagehub.models.lumina_next_transformer import LuminaNextDiT2DModel, rotary_frequencies

FIXTURES = fixture_path("lumina_next_configs")
TINY = {
    "in_channels": 4,
    "hidden_size": 32,
    "num_layers": 2,
    "num_attention_heads": 4,
    "num_kv_heads": 2,
    "multiple_of": 16,
    "cross_attention_dim": 24,
}


def test_matches_published_tensor_layout():
    config = json.loads((FIXTURES / "transformer_config.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURES / "transformer_tensors.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = LuminaNextDiT2DModel(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == expected
    assert len(actual) == 639


def test_tiny_forward_shape_and_text_mask():
    torch.manual_seed(0)
    model = LuminaNextDiT2DModel(TINY).eval()
    for layer in model.layers:
        layer.gate.data.normal_()
    hidden = torch.randn(2, 4, 6, 10)
    text = torch.randn(2, 5, 24)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
    timestep = torch.tensor([0.25, 0.75])
    rotary = rotary_frequencies(8, 16, 16)
    with torch.inference_mode():
        out = model(hidden, timestep, text, mask, rotary)
        full = model(hidden, timestep, text, torch.ones(2, 5, dtype=torch.long), rotary)
    assert out.shape == (2, 8, 6, 10)
    assert torch.equal(out[0], full[0])
    assert not torch.equal(out[1], full[1])


def test_zero_gate_ignores_the_token_order():
    torch.manual_seed(0)
    model = LuminaNextDiT2DModel(TINY).eval()
    hidden = torch.randn(1, 4, 4, 4)
    timestep = torch.tensor([0.5])
    mask = torch.ones(1, 3, dtype=torch.long)
    rotary = rotary_frequencies(8, 16, 16)
    with torch.inference_mode():
        a = model(hidden, timestep, torch.randn(1, 3, 24), mask, rotary)
        # The pooled caption still conditions the block norms, so keep the text's mean fixed.
        text = torch.randn(1, 3, 24)
        b = model(hidden, timestep, text, mask, rotary)
        c = model(hidden, timestep, text.flip(1), mask, rotary)
    assert not torch.equal(a, b)
    assert torch.allclose(b, c, atol=1e-5)


def test_rotary_table_shape_and_scaling():
    table = rotary_frequencies(16, 6, 5)
    assert table.shape == (6, 5, 8) and table.dtype == torch.complex64
    assert torch.allclose(table.abs(), torch.ones(()))
    linear = rotary_frequencies(16, 6, 5, linear_factor=2.0)
    assert torch.allclose(linear[1, 0].angle(), table[1, 0].angle() / 2, atol=1e-6)
    with pytest.raises(ValueError, match="divisible by 4"):
        rotary_frequencies(6)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"scaling_factor": 2.0}, "scaling_factor"),
        ({"num_kv_heads": 3}, "num_attention_heads"),
        ({"hidden_size": 36, "num_attention_heads": 4}, "divisible by 4"),
        ({"bogus": 1}, "bogus"),
    ],
)
def test_unsupported_option_rejected(override, match):
    with pytest.raises(ValueError, match=match):
        LuminaNextDiT2DModel(TINY | override)
