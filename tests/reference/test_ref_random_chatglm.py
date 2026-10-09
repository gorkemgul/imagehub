"""ChatGLM3 encoder parity with random weights, as Kolors calls it."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from imagehub.models.chatglm import ChatGLMModel

pytestmark = pytest.mark.reference
pytest.importorskip("transformers")
modeling = pytest.importorskip("diffusers.pipelines.kolors.text_encoder")

FIXTURES = Path(__file__).parents[1] / "fixtures" / "kolors_configs"
TINY = {
    "num_layers": 3,
    "padded_vocab_size": 128,
    "hidden_size": 64,
    "ffn_hidden_size": 96,
    "kv_channels": 16,
    "num_attention_heads": 4,
    "seq_length": 64,
    "layernorm_epsilon": 1e-5,
    "rmsnorm": True,
    "add_bias_linear": False,
    "add_qkv_bias": True,
    "multi_query_attention": True,
    "multi_query_group_num": 2,
    "original_rope": True,
    "post_layer_norm": True,
}
VARIANTS = {
    "kolors": TINY,
    "layernorm-post-residual": TINY
    | {"rmsnorm": False, "apply_residual_connection_post_layernorm": True},
    "mha-no-final-norm": TINY | {"multi_query_attention": False, "post_layer_norm": False},
    "all-bias": TINY | {"add_bias_linear": True},
    "wide": TINY
    | {
        "hidden_size": 256,
        "ffn_hidden_size": 384,
        "kv_channels": 32,
        "num_attention_heads": 8,
        "seq_length": 256,
    },
}


def _reference(config: dict[str, Any]) -> Any:
    reference = modeling.ChatGLMModel(modeling.ChatGLMConfig(**config), empty_init=False).eval()
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)
    return reference


def test_chatglm_published_layout() -> None:
    published = json.loads((FIXTURES / "text_encoder_config.json").read_text())
    with torch.device("meta"):
        expected = modeling.ChatGLMModel(modeling.ChatGLMConfig(**published)).state_dict()
        actual = ChatGLMModel(published).state_dict()
    assert sorted(actual) == sorted(expected)
    assert {name: tuple(value.shape) for name, value in actual.items()} == {
        name: tuple(value.shape) for name, value in expected.items()
    }


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_chatglm_left_padded_batch_exact(variant: str, record) -> None:
    config = VARIANTS[variant]
    reference = _reference(config)
    model = ChatGLMModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    torch.manual_seed(0)
    length = 200 if config["seq_length"] == 256 else 24
    input_ids = torch.randint(1, int(config["padded_vocab_size"]), (2, length))
    attention_mask = torch.ones(2, length, dtype=torch.long)
    position_ids = torch.arange(length).repeat(2, 1)
    for row, padding in enumerate((9, 3)):
        attention_mask[row, :padding] = 0
        input_ids[row, :padding] = 0
        position_ids[row, :padding] = 0
        position_ids[row, padding:] = torch.arange(length - padding)

    with torch.inference_mode():
        output = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        expected_embeds = output.hidden_states[-2].permute(1, 0, 2).clone()
        expected_pooled = output.hidden_states[-1][-1, :, :].clone()
        embeds, pooled = model(input_ids, position_ids, attention_mask.bool())
        unmasked, unmasked_pooled = model(input_ids[:, 3:])
        expected_unmasked = reference(input_ids=input_ids[:, 3:], output_hidden_states=True)

    assert torch.equal(embeds, expected_embeds)
    assert torch.equal(pooled, expected_pooled)
    assert torch.equal(unmasked, expected_unmasked.hidden_states[-2].permute(1, 0, 2))
    assert torch.equal(unmasked_pooled, expected_unmasked.hidden_states[-1][-1])
    record(f"chatglm_random_{variant}", max_abs_diff=0.0, tolerance="exact")
