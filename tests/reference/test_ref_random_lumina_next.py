"""Lumina-Next transformer exact parity with random weights."""

import json
import math

import pytest
import torch
from fixture_files import fixture_path

from imagehub.models.lumina_next_transformer import LuminaNextDiT2DModel, rotary_frequencies

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

FIXTURES = fixture_path("lumina_next_configs")
TINY = {
    "in_channels": 4,
    "hidden_size": 32,
    "num_layers": 2,
    "num_attention_heads": 4,
    "num_kv_heads": 2,
    "multiple_of": 16,
    "cross_attention_dim": 24,
    "qk_norm": True,
    "learn_sigma": True,
    "patch_size": 2,
}


def published_config():
    config = json.loads((FIXTURES / "transformer_config.json").read_text(encoding="utf-8"))
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _random_reference(config):
    torch.manual_seed(0)
    reference = diffusers.LuminaNextDiT2DModel(**config).eval()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(0.0, 0.05)
        for layer in reference.layers:
            layer.gate.normal_(0.0, 1.0)
    return reference


def _inputs(config, shape, text_len, lengths):
    batch = len(lengths)
    hidden = torch.randn(batch, config["in_channels"], *shape)
    text = torch.randn(batch, text_len, config["cross_attention_dim"])
    mask = torch.zeros(batch, text_len, dtype=torch.long)
    for index, length in enumerate(lengths):
        mask[index, :length] = 1
    return hidden, text, mask


@pytest.mark.parametrize("linear_factor,ntk_factor", [(1.0, 1.0), (1.5, 1.0), (1.0, 2.0)])
def test_rotary_frequencies_exact(linear_factor, ntk_factor):
    expected = diffusers.models.embeddings.get_2d_rotary_pos_embed_lumina(
        64, 384, 384, linear_factor=linear_factor, ntk_factor=ntk_factor
    )
    actual = rotary_frequencies(64, linear_factor=linear_factor, ntk_factor=ntk_factor)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def _compare(config, shape, text_len, lengths, timesteps, proportional, factors, record, name):
    reference = _random_reference(config)
    model = LuminaNextDiT2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())
    hidden, text, mask = _inputs(config, shape, text_len, lengths)
    timestep = torch.tensor(timesteps)
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    rotary = diffusers.models.embeddings.get_2d_rotary_pos_embed_lumina(
        head_dim, 384, 384, linear_factor=factors[0], ntk_factor=factors[1]
    )
    tokens = (shape[0] // config["patch_size"]) * (shape[1] // config["patch_size"])
    base = 4096 if proportional else None
    kwargs = {"base_sequence_length": base} if base is not None else {}
    with torch.inference_mode():
        expected = reference(
            hidden, timestep, text, mask, rotary, cross_attention_kwargs=kwargs, return_dict=False
        )[0]
        actual = model(hidden, timestep, text, mask, rotary, base_sequence_length=base)
    assert tokens > 0 and math.isfinite(float(actual.sum()))
    record(name, max_abs_diff=float((actual - expected).abs().max()))
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("proportional", [True, False])
@pytest.mark.parametrize("factors", [(1.0, 1.0), (1.5, 1.0), (1.0, 1.5)])
def test_lumina_next_tiny_exact(proportional, factors, record):
    _compare(
        TINY, (12, 20), 8, [8, 5], [0.1, 0.9], proportional, factors, record, "lumina_next_tiny"
    )


def test_lumina_next_tiny_without_qk_norm_exact(record):
    _compare(
        TINY | {"qk_norm": False},
        (8, 8),
        8,
        [8, 8],
        [0.3, 0.6],
        True,
        (1.0, 1.0),
        record,
        "lumina_next_tiny_no_qk_norm",
    )


def test_lumina_next_tiny_single_batch_exact(record):
    _compare(TINY, (16, 8), 16, [11], [0.5], True, (1.0, 1.0), record, "lumina_next_tiny_single")


def test_lumina_next_published_config_exact(record):
    config = published_config() | {"num_layers": 2}
    _compare(
        config, (16, 24), 16, [16, 9], [0.2, 0.8], True, (1.3, 1.0), record, "lumina_next_full"
    )


def test_lumina_next_published_layout() -> None:
    config = published_config()
    with torch.device("meta"):
        expected = diffusers.LuminaNextDiT2DModel(**config).state_dict()
        actual = LuminaNextDiT2DModel(config).state_dict()
    assert sorted(actual) == sorted(expected)
    assert {name: tuple(value.shape) for name, value in actual.items()} == {
        name: tuple(value.shape) for name, value in expected.items()
    }
