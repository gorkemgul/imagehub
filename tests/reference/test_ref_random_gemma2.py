"""Gemma-2 text model parity with random weights."""

import json
from typing import Any

import pytest
import torch
from fixture_files import fixture_path

from imagehub.models.gemma2 import Gemma2Model

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

FIXTURES = fixture_path("sana_configs")
CONFIG = FIXTURES / "1600m" / "text_encoder_config.json"

TINY: dict[str, Any] = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "attn_logit_softcapping": 50.0,
    "query_pre_attn_scalar": 8,
    "sliding_window": 4,
    "hidden_activation": "gelu_pytorch_tanh",
    "pad_token_id": 0,
}
MEDIUM = TINY | {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 32,
    "query_pre_attn_scalar": 32,
}


def _reference(config: dict[str, Any]) -> Any:
    reference_config = transformers.Gemma2Config(**config)
    reference_config._attn_implementation = "sdpa"
    return transformers.Gemma2Model(reference_config).eval()


def test_gemma2_published_layout() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with torch.device("meta"):
        expected = transformers.Gemma2Model(transformers.Gemma2Config(**config)).state_dict()
        actual = Gemma2Model(config).state_dict()
    assert sorted(actual) == sorted(expected)
    assert {name: tuple(value.shape) for name, value in actual.items()} == {
        name: tuple(value.shape) for name, value in expected.items()
    }


@pytest.mark.parametrize("config", [TINY, MEDIUM], ids=["tiny", "medium-width"])
@pytest.mark.parametrize("length", [3, 11, 16], ids=["shorter-than-window", "padded", "longer"])
@pytest.mark.parametrize("side", ["right", "left", "none"])
def test_gemma2_random_weights_exact(
    config: dict[str, Any], length: int, side: str, record
) -> None:
    torch.manual_seed(0)
    reference = _reference(config)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(0.0, 0.5)
    model = Gemma2Model(config).eval()
    model.load_checkpoint(reference.state_dict())
    layer_types = reference.config.layer_types
    assert layer_types.count("sliding_attention") == layer_types.count("full_attention") == 2

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
    record(f"gemma2_{side}_{length}_{config['hidden_size']}", max_abs_diff=0.0, tolerance="exact")


@pytest.mark.parametrize("config", [TINY, MEDIUM], ids=["tiny", "medium-width"])
@pytest.mark.parametrize("index", [-2, -1, 0, 1, 3])
def test_gemma2_hidden_state_index_exact(config: dict[str, Any], index: int, record) -> None:
    torch.manual_seed(0)
    reference = _reference(config)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(0.0, 0.5)
    model = Gemma2Model(config).eval()
    model.load_checkpoint(reference.state_dict())

    input_ids = torch.randint(1, int(config["vocab_size"]), (2, 11))
    attention_mask = torch.ones(2, 11, dtype=torch.long)
    attention_mask[0, -3:] = 0
    attention_mask[1, -1:] = 0
    with torch.inference_mode():
        expected = reference(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        ).hidden_states[index]
        actual = model(input_ids, attention_mask, hidden_state_index=index)
    assert torch.equal(actual, expected)
    record(
        f"gemma2_hidden_state_{index}_{config['hidden_size']}", max_abs_diff=0.0, tolerance="exact"
    )
