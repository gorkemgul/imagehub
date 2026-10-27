"""Gemma (first generation) text model parity with random weights."""

import json
from typing import Any

import pytest
import torch
from fixture_files import fixture_path

from imagehub.models.gemma import GemmaModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

FIXTURES = fixture_path("lumina_next_configs")
CONFIG = FIXTURES / "text_encoder_config.json"

TINY: dict[str, Any] = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "hidden_act": "gelu",
    "hidden_activation": "gelu_pytorch_tanh",
    "pad_token_id": 0,
}
# The published attention geometry (8 heads of 256 with one key-value head), shortened.
PUBLISHED_GEOMETRY = TINY | {
    "hidden_size": 2048,
    "intermediate_size": 512,
    "num_hidden_layers": 3,
    "num_attention_heads": 8,
    "head_dim": 256,
}


def _reference(config: dict[str, Any]) -> Any:
    reference_config = transformers.GemmaConfig(**config)
    reference_config._attn_implementation = "sdpa"
    return transformers.GemmaModel(reference_config).eval()


def _randomize(model: Any, std: float) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, std)


def test_gemma_published_layout() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with torch.device("meta"):
        expected = transformers.GemmaModel(transformers.GemmaConfig(**config)).state_dict()
        actual = GemmaModel(config).state_dict()
    assert sorted(actual) == sorted(expected)
    assert {name: tuple(value.shape) for name, value in actual.items()} == {
        name: tuple(value.shape) for name, value in expected.items()
    }


@pytest.mark.parametrize("config", [TINY, PUBLISHED_GEOMETRY], ids=["tiny", "published-geometry"])
@pytest.mark.parametrize("length", [1, 3, 8, 16])
@pytest.mark.parametrize("side", ["right", "left", "none"])
def test_gemma_random_weights_exact(config: dict[str, Any], length: int, side: str, record) -> None:
    torch.manual_seed(0)
    reference = _reference(config)
    _randomize(reference, 0.5 if config is TINY else 0.05)
    model = GemmaModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    input_ids = torch.randint(1, int(config["vocab_size"]), (2, length))
    attention_mask = torch.ones(2, length, dtype=torch.long)
    if side != "none" and length > 3:
        if side == "right":
            attention_mask[0, -3:] = 0
            attention_mask[1, -1:] = 0
        else:
            attention_mask[0, :3] = 0
            attention_mask[1, :1] = 0

    with torch.inference_mode():
        expected = reference(input_ids, attention_mask=attention_mask).last_hidden_state
        actual = model(input_ids, attention_mask)
    assert torch.equal(actual, expected)
    record(f"gemma1_{side}_{length}_{config['hidden_size']}", max_abs_diff=0.0, tolerance="exact")


@pytest.mark.parametrize("config", [TINY, PUBLISHED_GEOMETRY], ids=["tiny", "published-geometry"])
@pytest.mark.parametrize("index", [-2, -1, 0, 1, 2])
def test_gemma_hidden_state_index_exact(config: dict[str, Any], index: int, record) -> None:
    torch.manual_seed(0)
    reference = _reference(config)
    _randomize(reference, 0.5 if config is TINY else 0.05)
    model = GemmaModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    input_ids = torch.randint(1, int(config["vocab_size"]), (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    attention_mask[0, -3:] = 0
    attention_mask[1, -1:] = 0
    with torch.inference_mode():
        expected = reference(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        ).hidden_states[index]
        actual = model(input_ids, attention_mask, hidden_state_index=index)
    assert torch.equal(actual, expected)
    record(
        f"gemma1_hidden_state_{index}_{config['hidden_size']}", max_abs_diff=0.0, tolerance="exact"
    )
