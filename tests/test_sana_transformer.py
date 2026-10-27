import json

import pytest
import torch
from fixture_files import fixture_path

from imagehub.models.sana_transformer import SanaTransformer2DModel

FIXTURES = fixture_path("sana_configs")
TINY = {
    "num_attention_heads": 2,
    "attention_head_dim": 8,
    "num_layers": 2,
    "num_cross_attention_heads": 2,
    "cross_attention_head_dim": 8,
    "cross_attention_dim": 16,
    "caption_channels": 12,
    "in_channels": 4,
    "out_channels": 4,
}


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("size", ["1600m", "600m"])
def test_matches_published_tensor_layout(size):
    config = _load(f"{size}/transformer_config.json")
    expected = _load(f"{size}_transformer_tensors.json")
    with torch.device("meta"):
        model = SanaTransformer2DModel(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == expected


def test_tiny_forward_shape_and_mask():
    torch.manual_seed(0)
    model = SanaTransformer2DModel(TINY).eval()
    hidden = torch.randn(2, 4, 6, 10)
    text = torch.randn(2, 5, 12)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
    with torch.inference_mode():
        out = model(hidden, text, torch.tensor([500.0, 250.0]), mask)
        unmasked = model(hidden, text, torch.tensor([500.0, 250.0]))
    assert out.shape == hidden.shape
    assert not torch.equal(out[1], unmasked[1])


@pytest.mark.parametrize(
    "override",
    [{"interpolation_scale": 2}, {"qk_norm": "rms_norm"}],
)
def test_unsupported_option_rejected(override):
    with pytest.raises(ValueError, match="Unsupported Sana transformer setting"):
        SanaTransformer2DModel(TINY | override)


@pytest.mark.parametrize("qk_norm", [None, "rms_norm_across_heads"])
def test_qk_norm_supported(qk_norm):
    config = TINY | {"qk_norm": qk_norm}
    model = SanaTransformer2DModel(config).eval()
    hidden = torch.randn(2, 4, 6, 10)
    text = torch.randn(2, 5, 12)
    with torch.inference_mode():
        out = model(hidden, text, torch.tensor([500.0, 250.0]))
    assert out.shape == hidden.shape


def test_guidance_embeds_supported():
    config = TINY | {"guidance_embeds": True}
    model = SanaTransformer2DModel(config).eval()
    hidden = torch.randn(1, 4, 6, 10)
    text = torch.randn(1, 5, 12)
    with torch.inference_mode():
        out = model(
            hidden,
            text,
            torch.tensor([0.5]),
            guidance=torch.tensor([0.45]),
        )
    assert out.shape == hidden.shape


def test_unknown_option_rejected():
    with pytest.raises(ValueError, match="Unsupported Sana transformer setting bogus"):
        SanaTransformer2DModel(TINY | {"bogus": 1})
