import json

import pytest
import torch
from fixture_files import restricted_fixture

from imagehub.exceptions import CheckpointError
from imagehub.models.llama_text import LlamaTextModel, inverse_frequencies


def published_config():
    return json.loads(
        restricted_fixture("hidream_configs/text_encoder_4_config.json").read_text(encoding="utf-8")
    )


TINY = {
    "vocab_size": 20,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
}


def test_matches_llama_31_8b_tensor_layout():
    index = json.loads(
        restricted_fixture("hidream_configs/text_encoder_4_model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )
    published = {name for name in index["weight_map"] if name != "lm_head.weight"}
    with torch.device("meta"):
        model = LlamaTextModel(published_config())
    state = model.state_dict()
    assert {f"model.{name}" for name in state} == published
    assert list(state["embed_tokens.weight"].shape) == [128256, 4096]
    assert list(state["layers.31.self_attn.q_proj.weight"].shape) == [4096, 4096]
    assert list(state["layers.31.self_attn.k_proj.weight"].shape) == [1024, 4096]
    assert list(state["layers.31.self_attn.v_proj.weight"].shape) == [1024, 4096]
    assert list(state["layers.0.mlp.gate_proj.weight"].shape) == [14336, 4096]
    assert len(model.layers) == 32
    prepared = model.prepare_state_dict(
        {
            **{f"model.{name}": value for name, value in state.items()},
            "lm_head.weight": torch.zeros(1),
        }
    )
    assert set(prepared) == set(state)


def test_llama3_rope_scaling_of_the_published_config():
    config = published_config()
    scaled = inverse_frequencies(config)
    plain = inverse_frequencies({**config, "rope_scaling": None})
    assert scaled.shape == plain.shape == (64,)
    assert plain[0] == 1.0
    # High frequencies are untouched, the lowest are divided by the factor, the rest in between.
    ratio = scaled / plain
    assert ratio[0] == 1.0
    assert torch.allclose(ratio[-8:], torch.full((8,), 1 / 8))
    assert bool((ratio[1:] <= ratio[:-1]).all())
    assert bool(((ratio > 1 / 8) & (ratio < 1)).any())


def test_rope_parameters_accept_the_current_config_form():
    config = published_config()
    legacy = inverse_frequencies(config)
    current = inverse_frequencies(
        {
            "head_dim": 128,
            "rope_theta": None,
            "rope_scaling": None,
            "rope_parameters": {**config["rope_scaling"], "rope_theta": config["rope_theta"]},
        }
    )
    assert torch.equal(legacy, current)


def test_forward_returns_every_layer_with_the_last_normalized():
    torch.manual_seed(0)
    model = LlamaTextModel(TINY).eval()
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.1)
    ids = torch.randint(0, 20, (2, 6))
    mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    stacked = model(ids, mask)
    assert stacked.shape == (3, 2, 6, 16)
    # Padding never changes the positions in front of it.
    changed = ids.clone()
    changed[1, 4:] = 7
    assert torch.equal(model(changed, mask)[:, 1, :4], stacked[:, 1, :4])
    # Without padding the mask argument is optional.
    assert torch.equal(model(ids, torch.ones_like(mask)), model(ids))


def test_state_dict_conversion_ignores_the_language_model_head(tmp_path):
    from helpers import write_tiny_hidream

    root = write_tiny_hidream(tmp_path / "snapshot")
    model = LlamaTextModel.from_pretrained(root / "text_encoder_4")
    assert model.embed_tokens.weight.shape == (262, 32)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"hidden_act": "gelu"}, "activation"),
        ({"attention_bias": True}, "attention bias"),
        ({"mlp_bias": True}, "MLP bias"),
        ({"attention_dropout": 0.1}, "attention dropout"),
        ({"rope_scaling": {"rope_type": "yarn", "factor": 2.0}}, "rope type"),
        ({"num_key_value_heads": 3}, "head layout"),
        ({"sliding_window": 4}, "sliding_window"),
    ],
)
def test_unsupported_settings_fail_loudly(override, message):
    with pytest.raises(ValueError, match=message):
        LlamaTextModel(TINY | override)


def test_checkpoint_with_missing_weights_is_rejected():
    model = LlamaTextModel(TINY)
    with pytest.raises(CheckpointError):
        model.load_checkpoint({"embed_tokens.weight": torch.zeros(20, 16)})
