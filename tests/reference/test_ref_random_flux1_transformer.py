"""FLUX.1 transformer structural parity with random weights."""

import json

import pytest
import torch
from fixture_files import fixture_path

from imagehub.models.flux_transformer import FluxTransformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

CONFIGS = fixture_path("flux1_configs")
TENSORS = fixture_path("flux1_transformer_tensors.json")

TINY = {
    "patch_size": 1,
    "in_channels": 8,
    "out_channels": 8,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "joint_attention_dim": 24,
    "pooled_projection_dim": 20,
    "axes_dims_rope": [4, 6, 6],
}
MEDIUM = TINY | {
    "num_layers": 2,
    "num_single_layers": 2,
    "attention_head_dim": 64,
    "num_attention_heads": 8,
    "joint_attention_dim": 96,
    "pooled_projection_dim": 80,
    "axes_dims_rope": [16, 24, 24],
    "guidance_embeds": True,
}


def _ids(height: int, width: int, text_length: int):
    img_ids = torch.cartesian_prod(torch.arange(1), torch.arange(height), torch.arange(width))
    txt_ids = torch.zeros(text_length, 3)
    return img_ids, txt_ids


@pytest.mark.parametrize("variant", ["schnell", "dev", "lite"])
def test_flux1_published_layout(variant):
    config_path = (
        fixture_path("flux_lite_configs/transformer.json")
        if variant == "lite"
        else CONFIGS / f"{variant}_transformer_config.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tensor_path = (
        fixture_path("flux_lite_transformer_tensors.json") if variant == "lite" else TENSORS
    )
    published = json.loads(tensor_path.read_text(encoding="utf-8"))
    with torch.device("meta"):
        expected = diffusers.FluxTransformer2DModel(**config).state_dict()
        actual = FluxTransformer2DModel(config).state_dict()
    assert sorted(actual) == sorted(expected)
    assert {name: tuple(value.shape) for name, value in actual.items()} == {
        name: tuple(value.shape) for name, value in expected.items()
    }
    if variant in {"schnell", "lite"}:
        assert {name: list(value.shape) for name, value in actual.items()} == published


@pytest.mark.parametrize(
    ("config", "height", "width", "text_length"),
    [
        (TINY | {"guidance_embeds": False}, 3, 5, 17),
        (TINY | {"guidance_embeds": True}, 2, 7, 19),
        (MEDIUM, 2, 3, 256),
    ],
    ids=["tiny", "tiny-guidance", "medium-guidance"],
)
def test_flux1_transformer_random_weights_exact(config, height, width, text_length):
    torch.manual_seed(0)
    reference = diffusers.FluxTransformer2DModel(**config).eval()
    model = FluxTransformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    batch_size = 2
    hidden_states = torch.randn(batch_size, height * width, config["in_channels"])
    encoder_hidden_states = torch.randn(batch_size, text_length, config["joint_attention_dim"])
    pooled_projections = torch.randn(batch_size, config["pooled_projection_dim"])
    timestep = torch.tensor([0.00125, 0.8765])
    guidance = torch.tensor([1.5, 4.0]) if config["guidance_embeds"] else None
    img_ids, txt_ids = _ids(height, width, text_length)
    with torch.inference_mode():
        expected = reference(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        ).sample
        actual = model(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            guidance=guidance,
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
