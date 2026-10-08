"""GLM text encoder structural parity with random weights."""

from typing import Any

import pytest
import torch

from imagehub.models.glm import GlmModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

TINY = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1.5625e-7,
    "rope_theta": 10_000.0,
    "partial_rotary_factor": 0.5,
    "attention_bias": True,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "pad_token_id": 127,
}
MEDIUM = TINY | {
    "hidden_size": 512,
    "intermediate_size": 1360,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 64,
}


def _reference_config(config: dict[str, Any]) -> Any:
    reference_config = transformers.GlmConfig(**config)
    reference_config._attn_implementation = "sdpa"
    return reference_config


def test_glm_published_text_layout() -> None:
    with torch.device("meta"):
        expected = transformers.GlmModel(_reference_config({})).state_dict()
        actual = GlmModel({}).state_dict()
    assert sorted(actual) == sorted(expected)
    assert {name: tuple(value.shape) for name, value in actual.items()} == {
        name: tuple(value.shape) for name, value in expected.items()
    }


@pytest.mark.parametrize("config", [TINY, MEDIUM], ids=["tiny", "medium-width"])
def test_glm_random_weights_exact(config: dict[str, Any]) -> None:
    torch.manual_seed(0)
    reference = transformers.GlmModel(_reference_config(config)).eval()
    model = GlmModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    input_ids = torch.randint(0, int(config["vocab_size"]), (2, 16))
    input_ids[0, :8] = int(config["pad_token_id"])
    input_ids[1, :3] = int(config["pad_token_id"])

    with torch.inference_mode():
        expected = reference(input_ids, output_hidden_states=True).hidden_states[-2]
        actual = model(input_ids)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
