"""Lumina-Image 2.0 transformer exact parity with random weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.lumina2_transformer import Lumina2Transformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

FIXTURES = Path(__file__).parents[1] / "fixtures" / "lumina2_configs"
TINY = {
    "in_channels": 4,
    "hidden_size": 32,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "num_attention_heads": 4,
    "num_kv_heads": 2,
    "multiple_of": 16,
    "axes_dim_rope": [2, 2, 4],
    "axes_lens": [32, 16, 16],
    "cap_feat_dim": 12,
}


def _compare(config, shape, text_len, lengths, timesteps, record, name):
    torch.manual_seed(0)
    reference = diffusers.Lumina2Transformer2DModel(**config).eval()
    model = Lumina2Transformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())
    batch = len(lengths)
    hidden = torch.randn(batch, *shape)
    text = torch.randn(batch, text_len, config["cap_feat_dim"])
    mask = torch.zeros(batch, text_len, dtype=torch.long)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    timestep = torch.tensor(timesteps)
    with torch.inference_mode():
        expected = reference(hidden, timestep, text, mask, return_dict=False)[0]
        actual = model(hidden, timestep, text, mask)
    record(name, max_abs_diff=float((actual - expected).abs().max()))
    assert torch.equal(actual, expected)


def test_lumina2_tiny_exact(record):
    _compare(TINY, (4, 6, 10), 7, [7, 4], [0.1, 0.9], record, "lumina2_tiny")


def test_lumina2_tiny_equal_lengths_exact(record):
    _compare(TINY, (4, 8, 6), 5, [5, 5], [0.3, 0.6], record, "lumina2_tiny_equal")


def test_lumina2_published_config_exact(record):
    config = json.loads((FIXTURES / "transformer_config.json").read_text(encoding="utf-8"))
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    _compare(config, (16, 16, 24), 256, [256, 180], [0.2, 0.8], record, "lumina2_full")
