import json
from pathlib import Path

import pytest
import torch

from imagehub.models.ernie_image_transformer import ErnieImageTransformer2DModel

FIXTURES = Path(__file__).parent / "fixtures" / "ernie_image_configs"


def test_matches_published_tensor_layout_on_the_meta_device():
    config = json.loads((FIXTURES / "transformer" / "config.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURES / "transformer_tensors.json").read_text(encoding="utf-8"))
    index = json.loads(
        (FIXTURES / "transformer" / "diffusion_pytorch_model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )
    with torch.device("meta"):
        model = ErnieImageTransformer2DModel(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == expected
    assert sorted(actual) == sorted(index["weight_map"])
    assert (
        sum(value.numel() for value in model.parameters()) == index["metadata"]["total_size"] // 2
    )


def test_tiny_forward_shape_and_padding_independence():
    config = {
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_layers": 2,
        "ffn_hidden_size": 48,
        "in_channels": 8,
        "out_channels": 8,
        "text_in_dim": 24,
        "rope_axes_dim": [2, 2, 4],
    }
    torch.manual_seed(0)
    model = ErnieImageTransformer2DModel(config).eval()
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.05)
    latents = torch.randn(1, 8, 3, 5)
    text = torch.randn(1, 6, 24)
    timestep = torch.tensor([500.0])
    lengths = torch.tensor([6])
    out = model(latents, timestep, text, lengths)
    assert out.shape == latents.shape
    padded = torch.cat([text, torch.randn(1, 4, 24)], dim=1)
    again = model(latents, timestep, padded, lengths)
    torch.testing.assert_close(out, again, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "override",
    [{"qk_layernorm": False}, {"patch_size": 2}, {"rope_axes_dim": [2, 2, 2]}, {"hidden_size": 30}],
)
def test_unsupported_configs_are_rejected(override):
    config = {
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_layers": 1,
        "ffn_hidden_size": 48,
        "in_channels": 8,
        "out_channels": 8,
        "text_in_dim": 24,
        "rope_axes_dim": [2, 2, 4],
    } | override
    with pytest.raises(ValueError):
        ErnieImageTransformer2DModel(config)
