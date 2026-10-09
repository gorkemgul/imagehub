import json
from pathlib import Path

import pytest
import torch

from imagehub.models.auraflow_transformer import AuraFlowTransformer2DModel

FIXTURES = Path(__file__).parent / "fixtures" / "auraflow_configs"
TINY = {
    "sample_size": 8,
    "patch_size": 2,
    "in_channels": 4,
    "num_mmdit_layers": 1,
    "num_single_dit_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 12,
    "caption_projection_dim": 16,
    "out_channels": 4,
    "pos_embed_max_size": 64,
}


def test_matches_published_tensor_layout():
    config = json.loads((FIXTURES / "transformer_config.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURES / "transformer_tensors.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = AuraFlowTransformer2DModel(config)
    assert {name: list(value.shape) for name, value in model.state_dict().items()} == expected
    assert len(expected) == 331


def test_tiny_forward_shape_and_conditioning():
    torch.manual_seed(0)
    model = AuraFlowTransformer2DModel(TINY).eval()
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.1)
    hidden = torch.randn(2, 4, 6, 10)
    text = torch.randn(2, 5, 12)
    with torch.inference_mode():
        out = model(hidden, text, torch.tensor([0.5, 0.25]))
        other = model(hidden, text, torch.tensor([0.9, 0.25]))
    assert out.shape == hidden.shape
    assert not torch.equal(out[0], other[0])
    assert torch.equal(out[1], other[1])


def test_positional_embedding_is_centre_cropped():
    index = AuraFlowTransformer2DModel(TINY).pos_embed.selection_index(4, 8)
    # An 8x8 grid: a 2x4 patch window starts at row 3, column 2.
    assert index.tolist() == [(3 + r) * 8 + 2 + c for r in range(2) for c in range(4)]


@pytest.mark.parametrize("override", [{"unknown_option": 1}, {"patch_size": 2, "extra": None}])
def test_rejects_unknown_settings(override):
    with pytest.raises(ValueError, match="Unsupported AuraFlow transformer setting"):
        AuraFlowTransformer2DModel(TINY | override)


def test_rejects_latents_larger_than_the_positional_grid():
    model = AuraFlowTransformer2DModel(TINY).eval()
    with pytest.raises(ValueError, match="positional grid"):
        model(torch.zeros(1, 4, 18, 8), torch.zeros(1, 3, 12), torch.tensor([0.5]))
