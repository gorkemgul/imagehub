"""LongCat-Image transformer structural parity with random weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.longcat_image_transformer import LongCatImageTransformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

CONFIG = (
    Path(__file__).parents[1] / "fixtures" / "longcat_image_configs" / "transformer_config.json"
)
TENSORS = Path(__file__).parents[1] / "fixtures" / "longcat_image_transformer_tensors.json"

TINY = {
    "patch_size": 1,
    "in_channels": 8,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "joint_attention_dim": 24,
    "pooled_projection_dim": 24,
    "axes_dims_rope": [4, 6, 6],
}
MEDIUM = TINY | {
    "num_layers": 2,
    "num_single_layers": 2,
    "attention_head_dim": 32,
    "num_attention_heads": 4,
    "joint_attention_dim": 96,
    "pooled_projection_dim": 96,
    "axes_dims_rope": [8, 12, 12],
}


def _ids(height: int, width: int, text_length: int):
    txt_ids = torch.zeros(text_length, 3)
    txt_ids[:, 1] = torch.arange(text_length)
    txt_ids[:, 2] = torch.arange(text_length)
    img_ids = torch.zeros(height, width, 3)
    img_ids[..., 0] = 1
    img_ids[..., 1] += torch.arange(height)[:, None] + 512
    img_ids[..., 2] += torch.arange(width)[None, :] + 512
    return img_ids.reshape(height * width, 3), txt_ids


def test_longcat_image_published_layout():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    published = json.loads(TENSORS.read_text(encoding="utf-8"))
    with torch.device("meta"):
        expected = diffusers.LongCatImageTransformer2DModel(
            **{name: value for name, value in config.items() if name != "guidance_embeds"}
        ).state_dict()
        actual = LongCatImageTransformer2DModel(config).state_dict()
    assert {name: list(value.shape) for name, value in expected.items()} == published
    assert {name: list(value.shape) for name, value in actual.items()} == published


@pytest.mark.parametrize(
    ("config", "height", "width", "text_length"),
    [(TINY, 3, 5, 17), (MEDIUM, 2, 3, 511)],
    ids=["tiny-short", "medium-512-like"],
)
def test_longcat_image_transformer_random_weights_exact(config, height, width, text_length):
    torch.manual_seed(0)
    reference = diffusers.LongCatImageTransformer2DModel(**config).eval()
    model = LongCatImageTransformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    batch_size = 2
    hidden_states = torch.randn(batch_size, height * width, config["in_channels"])
    encoder_hidden_states = torch.randn(batch_size, text_length, config["joint_attention_dim"])
    timestep = torch.tensor([0.00125, 0.8765])
    img_ids, txt_ids = _ids(height, width, text_length)
    with torch.inference_mode():
        expected = reference(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
        ).sample
        actual = model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            guidance=None,
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
