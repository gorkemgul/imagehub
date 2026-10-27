import json

import pytest
import torch
from fixture_files import fixture_path

from imagehub.checkpoints import CheckpointError
from imagehub.models.hidream_transformer import HiDreamImageTransformer2DModel

CONFIGS = fixture_path("hidream_configs")

TINY = {
    "patch_size": 2,
    "in_channels": 4,
    "num_layers": 2,
    "num_single_layers": 2,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "caption_channels": [12, 20],
    "text_emb_dim": 10,
    "num_routed_experts": 4,
    "num_activated_experts": 2,
    "axes_dims_rope": [8, 4, 4],
    "max_resolution": [8, 8],
    "llama_layers": [0, 1, 2, 2],
}


def test_matches_hidream_transformer_tensor_layout():
    config = json.loads((CONFIGS / "transformer_config.json").read_text(encoding="utf-8"))
    expected = json.loads((CONFIGS / "transformer_tensors.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        state = HiDreamImageTransformer2DModel(config).state_dict()
    assert {name: list(value.shape) for name, value in state.items()} == expected
    assert len(state) == 1615


@pytest.mark.parametrize(("height", "width"), [(8, 8), (4, 8), (8, 4)])
def test_tiny_forward_shapes(height, width):
    torch.manual_seed(0)
    model = HiDreamImageTransformer2DModel(TINY)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    batch = 2
    llama = torch.randn(3, batch, 6, 20)
    t5 = torch.randn(batch, 5, 12)
    with torch.inference_mode():
        output = model(
            torch.randn(batch, 4, height, width),
            torch.tensor([10.0, 500.0]),
            t5,
            llama,
            torch.randn(batch, 10),
        )
    assert output.shape == (batch, 4, height, width)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "change",
    [
        {"unknown_option": 1},
        {"axes_dims_rope": [8, 4, 2]},
        {"patch_size": 0},
        {"num_layers": 0},
        {"num_activated_experts": 5},
        {"llama_layers": [0, 1]},
        {"caption_channels": [12]},
        {"out_channels": 3},
    ],
)
def test_rejects_unsupported_options(change):
    with pytest.raises(ValueError, match="HiDream transformer"):
        HiDreamImageTransformer2DModel(TINY | change)


def test_requires_published_options():
    config = dict(TINY)
    del config["llama_layers"]
    with pytest.raises(ValueError, match="Missing"):
        HiDreamImageTransformer2DModel(config)


def test_rejects_too_few_llama_layers_at_forward():
    model = HiDreamImageTransformer2DModel(TINY)
    with pytest.raises(ValueError, match="fewer layers"):
        model(
            torch.zeros(1, 4, 8, 8),
            torch.zeros(1),
            torch.zeros(1, 5, 12),
            torch.zeros(2, 1, 6, 20),
            torch.zeros(1, 10),
        )


def test_checkpoint_loading_is_strict():
    model = HiDreamImageTransformer2DModel(TINY)
    state = model.state_dict()
    state.pop("final_layer.linear.bias")
    with pytest.raises(CheckpointError):
        model.load_checkpoint(state)
