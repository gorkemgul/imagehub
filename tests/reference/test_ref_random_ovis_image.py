"""Ovis-Image module parity with random float32 weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.ovis_image_transformer import OvisImageTransformer2DModel
from imagehub.models.qwen3 import Qwen3Model

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

FIXTURES = Path(__file__).parents[1] / "fixtures" / "ovis_image_configs"
TINY = {
    "in_channels": 8,
    "out_channels": 8,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 4,
    "num_attention_heads": 2,
    "joint_attention_dim": 12,
    "axes_dims_rope": [2, 2, 0],
}
PUBLISHED_WIDTH = json.loads((FIXTURES / "transformer_config.json").read_text()) | {
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "joint_attention_dim": 32,
    "axes_dims_rope": [4, 6, 6],
}
TEXT = {
    "vocab_size": 37,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "use_sliding_window": False,
    "sliding_window": None,
}


@pytest.mark.parametrize("config", [TINY, PUBLISHED_WIDTH], ids=["tiny", "published-width"])
def test_ovis_image_transformer_random_weights_exact(config):
    torch.manual_seed(0)
    reference = diffusers.OvisImageTransformer2DModel(**config).eval()
    actual = OvisImageTransformer2DModel(config).eval()
    actual.load_checkpoint(reference.state_dict())
    hidden = torch.randn(2, 15, config["in_channels"])
    context = torch.randn(2, 7, config["joint_attention_dim"])
    timestep = torch.tensor([0.125, 0.875])
    image_ids = torch.zeros(15, 3)
    image_ids[:, 1] = torch.arange(15) // 3
    image_ids[:, 2] = torch.arange(15) % 3
    text_ids = torch.zeros(7, 3)
    text_ids[:, 1:] = torch.arange(7)[:, None]
    with torch.inference_mode():
        expected = reference(hidden, context, timestep, image_ids, text_ids, return_dict=False)[0]
        result = actual(hidden, context, timestep, image_ids, text_ids)
    assert torch.equal(result, expected)


def test_ovis_image_published_transformer_layout():
    config = json.loads((FIXTURES / "transformer_config.json").read_text())
    tensors = {}
    for name in ("transformer_00001_tensors.json", "transformer_00002_tensors.json"):
        tensors.update(json.loads((FIXTURES / name).read_text()))
    with torch.device("meta"):
        reference = diffusers.OvisImageTransformer2DModel(**config).state_dict()
        actual = OvisImageTransformer2DModel(config).state_dict()
    assert {name: list(value.shape) for name, value in reference.items()} == tensors
    assert {name: list(value.shape) for name, value in actual.items()} == tensors


def test_ovis_image_qwen3_text_encoder_random_weights_exact():
    torch.manual_seed(1)
    reference = transformers.Qwen3Model(transformers.Qwen3Config(**TEXT)).eval()
    actual = Qwen3Model(TEXT).eval()
    actual.load_checkpoint(reference.state_dict())
    ids = torch.tensor([[1, 2, 3, 4, 0], [4, 3, 2, 1, 0]])
    mask = ids.ne(0)
    with torch.inference_mode():
        expected = reference(ids, attention_mask=mask).last_hidden_state
        result = actual(ids, mask)
    assert torch.equal(result, expected)
