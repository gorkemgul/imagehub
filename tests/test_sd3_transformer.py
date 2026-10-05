import json
from pathlib import Path

import pytest
import torch

from imagehub.models.sd3_transformer import SD3Transformer2DModel

FIXTURES = Path(__file__).parent / "fixtures"
CONFIGS = FIXTURES / "sd35_configs"


def test_matches_sd35_large_tensor_names():
    config = json.loads((CONFIGS / "large_transformer_config.json").read_text(encoding="utf-8"))
    expected = json.loads(
        (FIXTURES / "sd35_large_transformer_tensors.json").read_text(encoding="utf-8")
    )
    with torch.device("meta"):
        model = SD3Transformer2DModel(config)
    assert sorted(model.state_dict()) == expected


TINY_CONFIG = {
    "sample_size": 8,
    "patch_size": 2,
    "in_channels": 4,
    "num_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 20,
    "caption_projection_dim": 16,
    "pooled_projection_dim": 12,
    "out_channels": 4,
    "pos_embed_max_size": 8,
    "dual_attention_layers": [],
    "qk_norm": "rms_norm",
}


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("qk_norm", None),
        ("qk_norm", "layer_norm"),
    ],
)
def test_rejects_unsupported_config_options(option, value):
    with pytest.raises(ValueError, match="Unsupported SD3 transformer option"):
        SD3Transformer2DModel(TINY_CONFIG | {option: value})


@pytest.mark.parametrize(
    "options",
    [
        {"joint_attention_kwargs": {"ip_adapter_image_embeds": torch.randn(1, 2)}},
        {"block_controlnet_hidden_states": [torch.randn(1, 2, 16)]},
        {"skip_layers": [0]},
    ],
)
def test_rejects_unsupported_forward_options(options):
    model = SD3Transformer2DModel(TINY_CONFIG)
    inputs = {
        "hidden_states": torch.randn(1, 4, 4, 4),
        "encoder_hidden_states": torch.randn(1, 3, 20),
        "pooled_projections": torch.randn(1, 12),
        "timestep": torch.tensor([1.5]),
    }
    with pytest.raises(ValueError, match="Unsupported SD3 transformer option"):
        model(**inputs, **options)
