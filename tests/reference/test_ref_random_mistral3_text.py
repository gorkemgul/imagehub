"""Ministral 3 language model (Mistral3 text encoder) parity with random weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.mistral3_text import Mistral3TextModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

FIXTURES = Path(__file__).parents[1] / "fixtures" / "ernie_image_configs"
YARN = {
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "factor": 16.0,
    "llama_4_scaling_beta": 0.1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 16,
    "rope_theta": 1000000.0,
    "rope_type": "yarn",
    "type": "yarn",
}
TINY_TEXT = {
    "model_type": "ministral3",
    "vocab_size": 97,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "sliding_window": None,
    "pad_token_id": 11,
    "rope_parameters": YARN,
}
TINY_VISION = {
    "model_type": "pixtral",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "head_dim": 8,
    "image_size": 28,
    "patch_size": 14,
}


def _published_text_config():
    published = json.loads((FIXTURES / "text_encoder" / "config.json").read_text())
    return published["text_config"]


def _build(text_config, seed=0):
    torch.manual_seed(seed)
    config = transformers.Mistral3Config(
        text_config=text_config, vision_config=TINY_VISION, image_token_index=10
    )
    reference = transformers.Mistral3Model(config).eval()
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=gen) * 0.1)
    return reference


def _published_layout(reference):
    # The Hub stores the language model as ``language_model.model.*``.
    state = {}
    for name, value in reference.state_dict().items():
        if name.startswith("language_model."):
            name = "language_model.model." + name.removeprefix("language_model.")
        state[name] = value
    return state


@pytest.mark.parametrize("length", [1, 2, 9, 40])
@pytest.mark.parametrize("rope", ["yarn", "default"])
def test_tiny_text_model_exact(length, rope):
    text = TINY_TEXT | (
        {} if rope == "yarn" else {"rope_parameters": {**YARN, "rope_type": "default"}}
    )
    reference = _build(text)
    model = Mistral3TextModel({"text_config": text}).eval()
    model.load_checkpoint(_published_layout(reference))
    ids = torch.randint(0, 97, (1, length), generator=torch.Generator().manual_seed(length))
    with torch.inference_mode():
        expected = reference(input_ids=ids, output_hidden_states=True)
        last, hidden = model(ids, output_hidden_states=True)
    assert len(hidden) == len(expected.hidden_states)
    for ours, theirs in zip(hidden, expected.hidden_states, strict=True):
        assert torch.equal(ours, theirs)
    assert torch.equal(last, expected.last_hidden_state)
    assert torch.equal(hidden[-2], expected.hidden_states[-2])


def test_published_text_config_small_width_exact():
    text = _published_text_config() | {
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "vocab_size": 300,
        "head_dim": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
    }
    reference = _build(text)
    model = Mistral3TextModel({"text_config": text}).eval()
    model.load_checkpoint(_published_layout(reference))
    ids = torch.randint(0, 300, (1, 33), generator=torch.Generator().manual_seed(5))
    with torch.inference_mode():
        expected = reference(input_ids=ids, output_hidden_states=True)
        _, hidden = model(ids, output_hidden_states=True)
    assert all(torch.equal(a, b) for a, b in zip(hidden, expected.hidden_states, strict=True))
