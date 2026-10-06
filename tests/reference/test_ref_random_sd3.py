"""SD3 transformer structural parity with random weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.sd3_transformer import SD3Transformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

FIXTURES = Path(__file__).parents[1] / "fixtures"
CONFIGS = FIXTURES / "sd35_configs"

TINY = {
    "sample_size": 8,
    "patch_size": 2,
    "in_channels": 4,
    "num_layers": 3,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 20,
    "caption_projection_dim": 16,
    "pooled_projection_dim": 12,
    "out_channels": 4,
    "pos_embed_max_size": 8,
    "qk_norm": "rms_norm",
}
MEDIUM = TINY | {
    "num_layers": 1,
    "attention_head_dim": 64,
    "num_attention_heads": 8,
    "caption_projection_dim": 512,
    "dual_attention_layers": [],
}


@pytest.mark.parametrize(
    "config_path",
    [
        CONFIGS / "sd3_medium_transformer_config.json",
        CONFIGS / "medium_transformer_config.json",
        CONFIGS / "large_transformer_config.json",
    ],
    ids=["sd3-medium", "sd35-medium", "sd35-large"],
)
def test_sd35_published_layout(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = {name: value for name, value in config.items() if not name.startswith("_")}
    with torch.device("meta"):
        expected = diffusers.SD3Transformer2DModel(**config).state_dict()
        actual = SD3Transformer2DModel(config).state_dict()
    assert sorted(actual) == sorted(expected)
    assert {name: tuple(value.shape) for name, value in actual.items()} == {
        name: tuple(value.shape) for name, value in expected.items()
    }


@pytest.mark.parametrize(
    "config",
    [
        TINY | {"qk_norm": None, "dual_attention_layers": []},
        TINY | {"dual_attention_layers": []},
        TINY | {"dual_attention_layers": [0, 1]},
        MEDIUM,
    ],
    ids=["tiny-no-qk-norm", "tiny", "tiny-dual", "medium-width"],
)
def test_sd3_transformer_random_weights_exact(config):
    torch.manual_seed(0)
    reference = diffusers.SD3Transformer2DModel(**config).eval()
    model = SD3Transformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    hidden_states = torch.randn(2, config["in_channels"], 6, 8)
    encoder_hidden_states = torch.randn(2, 77 + 256, config["joint_attention_dim"])
    pooled_projections = torch.randn(2, config["pooled_projection_dim"])
    timestep = torch.tensor([1.25, 876.5])
    with torch.inference_mode():
        expected = reference(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
        ).sample
        actual = model(hidden_states, encoder_hidden_states, pooled_projections, timestep)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
