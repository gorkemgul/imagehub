"""HunyuanImage 2.1 transformer structural parity with random weights."""

import json

import pytest
import torch
from fixture_files import fixture_path

from imagehub.models.hunyuanimage_transformer import HunyuanImageTransformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

FIXTURES = fixture_path("hunyuanimage_configs")
TINY = {
    "in_channels": 8,
    "out_channels": 8,
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "num_layers": 1,
    "num_single_layers": 1,
    "num_refiner_layers": 1,
    "patch_size": [1, 1],
    "text_embed_dim": 24,
    "text_embed_2_dim": 20,
    "rope_axes_dim": [8, 8],
}
MEDIUM = TINY | {
    "num_attention_heads": 4,
    "attention_head_dim": 32,
    "num_layers": 2,
    "num_single_layers": 3,
    "num_refiner_layers": 2,
    "text_embed_dim": 96,
    "rope_axes_dim": [16, 16],
}


def _reference(config):
    return diffusers.HunyuanImageTransformer2DModel(
        **{name: value for name, value in config.items() if name != "_class_name"}
    ).eval()


def _inputs(config, height, width, text_length, glyph_length, batch_size=2):
    mask = torch.zeros(batch_size, text_length, dtype=torch.int64)
    mask_2 = torch.zeros(batch_size, glyph_length, dtype=torch.int64)
    for row, (valid, valid_2) in enumerate(
        [(text_length - 3, glyph_length // 2), (5, 0)][:batch_size]
    ):
        mask[row, :valid] = 1
        mask_2[row, :valid_2] = 1
    return {
        "hidden_states": torch.randn(batch_size, config["in_channels"], height, width),
        "timestep": torch.tensor([1000.0, 523.25]),
        "encoder_hidden_states": torch.randn(batch_size, text_length, config["text_embed_dim"]),
        "encoder_attention_mask": mask,
        "encoder_hidden_states_2": torch.randn(
            batch_size, glyph_length, config["text_embed_2_dim"]
        ),
        "encoder_attention_mask_2": mask_2,
    }


def test_hunyuanimage_published_layout():
    config = json.loads((FIXTURES / "transformer_config.json").read_text(encoding="utf-8"))
    published = json.loads((FIXTURES / "transformer_tensors.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        expected = _reference(config).state_dict()
        actual = HunyuanImageTransformer2DModel(config).state_dict()
    assert {name: list(value.shape) for name, value in expected.items()} == published
    assert {name: list(value.shape) for name, value in actual.items()} == published


@pytest.mark.parametrize(
    ("config", "height", "width", "text_length", "glyph_length"),
    [(TINY, 3, 5, 17, 8), (MEDIUM, 4, 6, 41, 16)],
    ids=["tiny", "medium"],
)
def test_transformer_random_weights_exact(config, height, width, text_length, glyph_length):
    torch.manual_seed(0)
    reference = _reference(config)
    model = HunyuanImageTransformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())
    inputs = _inputs(config, height, width, text_length, glyph_length)
    with torch.inference_mode():
        expected = reference(**inputs, return_dict=False)[0]
        actual = model(**inputs)
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
