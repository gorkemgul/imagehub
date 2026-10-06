"""FLUX.2 transformer structural parity with random weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.flux2_transformer import Flux2Transformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

CONFIG = (
    Path(__file__).parents[1] / "fixtures" / "flux2_klein_4b_configs" / "transformer_config.json"
)
TENSORS = Path(__file__).parents[1] / "fixtures" / "flux2_klein_4b_transformer_tensors.json"

TINY = {
    "patch_size": 1,
    "in_channels": 8,
    "out_channels": 8,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "joint_attention_dim": 24,
    "timestep_guidance_channels": 32,
    "mlp_ratio": 3.0,
    "axes_dims_rope": [4, 4, 4, 4],
    "rope_theta": 2000,
    "eps": 1e-6,
    "guidance_embeds": False,
}
MEDIUM = TINY | {
    "num_layers": 2,
    "num_single_layers": 2,
    "attention_head_dim": 64,
    "num_attention_heads": 8,
    "joint_attention_dim": 96,
    "timestep_guidance_channels": 256,
    "axes_dims_rope": [16, 16, 16, 16],
    "guidance_embeds": True,
}


def _ids(batch_size: int, height: int, width: int, text_length: int):
    img_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(1)
    )
    txt_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(text_length)
    )
    return (
        img_ids.unsqueeze(0).expand(batch_size, -1, -1),
        txt_ids.unsqueeze(0).expand(batch_size, -1, -1),
    )


def test_flux2_klein_published_layout():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    published = json.loads(TENSORS.read_text(encoding="utf-8"))
    config = {name: value for name, value in config.items() if not name.startswith("_")}
    with torch.device("meta"):
        expected = diffusers.Flux2Transformer2DModel(**config).state_dict()
        actual = Flux2Transformer2DModel(config).state_dict()
    assert sorted(actual) == sorted(expected)
    assert {name: tuple(value.shape) for name, value in actual.items()} == {
        name: tuple(value.shape) for name, value in expected.items()
    }
    assert {name: list(value.shape) for name, value in actual.items()} == published


@pytest.mark.parametrize(
    ("config", "height", "width", "text_length"),
    [(TINY, 3, 5, 17), (MEDIUM, 2, 3, 512)],
    ids=["tiny", "medium-width-guidance"],
)
def test_flux2_transformer_random_weights_exact(config, height, width, text_length):
    torch.manual_seed(0)
    reference = diffusers.Flux2Transformer2DModel(**config).eval()
    model = Flux2Transformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    batch_size = 2
    hidden_states = torch.randn(batch_size, height * width, config["in_channels"])
    encoder_hidden_states = torch.randn(batch_size, text_length, config["joint_attention_dim"])
    timestep = torch.tensor([0.00125, 0.8765])
    guidance = torch.tensor([1.5, 4.0]) if config["guidance_embeds"] else None
    img_ids, txt_ids = _ids(batch_size, height, width, text_length)
    with torch.inference_mode():
        expected = reference(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        ).sample
        actual = model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            guidance=guidance,
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
