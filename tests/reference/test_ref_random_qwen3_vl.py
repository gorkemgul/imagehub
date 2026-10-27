"""Qwen3-VL text encoder structural parity with random weights."""

import json
from typing import Any

import pytest
import torch
from fixture_files import fixture_path

from imagehub.models.qwen3_vl_text import Qwen3VLTextModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

FIXTURES = fixture_path()
CONFIG = FIXTURES / "qwen_image21_configs" / "text_encoder_config.json"

TINY = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5_000_000,
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
        "rope_type": "default",
    },
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
}
MEDIUM = TINY | {
    "hidden_size": 512,
    "intermediate_size": 1536,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    },
}


def _reference_config(config: dict[str, Any]) -> Any:
    reference_config = transformers.Qwen3VLTextConfig(**config)
    reference_config._attn_implementation = "sdpa"
    return reference_config


def test_qwen3_vl_published_text_layout() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))["text_config"]
    with torch.device("meta"):
        expected = transformers.Qwen3VLTextModel(_reference_config(config)).state_dict()
        actual = Qwen3VLTextModel(config).state_dict()
    assert sorted(actual) == sorted(expected)
    assert {name: tuple(value.shape) for name, value in actual.items()} == {
        name: tuple(value.shape) for name, value in expected.items()
    }


@pytest.mark.parametrize("config", [TINY, MEDIUM], ids=["tiny", "medium-width"])
def test_qwen3_vl_text_random_weights_exact(config: dict[str, Any]) -> None:
    torch.manual_seed(0)
    reference = transformers.Qwen3VLTextModel(_reference_config(config)).eval()
    model = Qwen3VLTextModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    input_ids = torch.randint(0, int(config["vocab_size"]), (2, 13))
    attention_mask = torch.ones(2, 13, dtype=torch.long)
    attention_mask[0, :4] = 0
    attention_mask[1, :1] = 0

    handle = reference.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        with torch.inference_mode():
            expected = reference(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states[-1]
            actual = model(input_ids, attention_mask)[-1]
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_qwen3_vl_explicit_three_axis_position_ids_exact() -> None:
    torch.manual_seed(1)
    reference = transformers.Qwen3VLTextModel(_reference_config(TINY)).eval()
    model = Qwen3VLTextModel(TINY).eval()
    model.load_checkpoint(reference.state_dict())
    input_ids = torch.randint(0, TINY["vocab_size"], (2, 9))
    attention_mask = torch.tensor(
        [[1, 1, 1, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool
    )
    positions = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    position_ids = positions.unsqueeze(0).expand(3, -1, -1)
    pre_norm: list[torch.Tensor] = []
    handle = reference.norm.register_forward_hook(
        lambda module, args, output: pre_norm.append(args[0])
    )
    try:
        with torch.inference_mode():
            expected = list(
                reference(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    use_cache=False,
                ).hidden_states
            )
            actual = model(
                input_ids,
                attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )
    finally:
        handle.remove()
    expected[-1] = pre_norm[0]
    assert len(actual) == len(expected)
    for actual_state, expected_state in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)
