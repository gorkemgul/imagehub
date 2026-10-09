import json
from pathlib import Path

import pytest
import torch

from imagehub.models.gemma2 import Gemma2Model

FIXTURES = Path(__file__).parent / "fixtures" / "sana_configs"
CONFIG = FIXTURES / "1600m" / "text_encoder_config.json"
TENSORS = FIXTURES / "text_encoder_tensors.json"


def _published_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_matches_published_tensor_names_and_shapes():
    config = _published_config()
    with torch.device("meta"):
        model = Gemma2Model(config)
    state = model.state_dict()

    assert sorted(state) == json.loads(TENSORS.read_text(encoding="utf-8"))
    assert list(state["embed_tokens.weight"].shape) == [256000, 2304]
    assert list(state["norm.weight"].shape) == [2304]
    for layer in range(26):
        prefix = f"layers.{layer}"
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            assert list(state[f"{prefix}.{norm}.weight"].shape) == [2304]
        assert list(state[f"{prefix}.self_attn.q_proj.weight"].shape) == [2048, 2304]
        assert list(state[f"{prefix}.self_attn.k_proj.weight"].shape) == [1024, 2304]
        assert list(state[f"{prefix}.self_attn.v_proj.weight"].shape) == [1024, 2304]
        assert list(state[f"{prefix}.self_attn.o_proj.weight"].shape) == [2304, 2048]
        assert list(state[f"{prefix}.mlp.gate_proj.weight"].shape) == [9216, 2304]
        assert list(state[f"{prefix}.mlp.up_proj.weight"].shape) == [9216, 2304]
        assert list(state[f"{prefix}.mlp.down_proj.weight"].shape) == [2304, 9216]


def test_layers_alternate_sliding_and_global_attention():
    with torch.device("meta"):
        model = Gemma2Model(_published_config())
    assert model.layer_types[:4] == [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert len(model.layer_types) == 26


@pytest.mark.parametrize(
    ("option", "value"),
    [("hidden_activation", "gelu"), ("layer_types", ["full_attention"])],
)
def test_rejects_unsupported_options(option, value):
    with pytest.raises(ValueError, match="Unsupported Gemma2"):
        Gemma2Model(_published_config() | {option: value})


def test_masks_follow_the_reference_skip_rules():
    config = {
        "vocab_size": 8,
        "hidden_size": 8,
        "intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "sliding_window": 3,
    }
    model = Gemma2Model(config)
    ones = torch.ones(1, 5, dtype=torch.long)
    masks = model._masks(ones, 1, 5, torch.device("cpu"))
    assert masks["full_attention"] is None
    assert masks["sliding_attention"] is not None
    assert masks["sliding_attention"][0, 0].tolist()[4] == [False, False, True, True, True]
    short = model._masks(ones[:, :2], 1, 2, torch.device("cpu"))
    assert short == {"full_attention": None, "sliding_attention": None}


def test_forward_shape_and_finite_with_padding():
    torch.manual_seed(0)
    config = {
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "sliding_window": 3,
    }
    model = Gemma2Model(config).eval()
    ids = torch.randint(0, 16, (2, 6))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    output = model(ids, mask)
    assert output.shape == (2, 6, 8)
    assert torch.isfinite(output).all()


def test_hidden_state_index_selects_entries_of_the_hidden_states_tuple():
    torch.manual_seed(0)
    config = {
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "sliding_window": 3,
    }
    model = Gemma2Model(config).eval()
    ids = torch.randint(0, 16, (2, 6))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    with torch.inference_mode():
        final = model(ids, mask)
        assert torch.equal(model(ids, mask, hidden_state_index=-1), final)
        assert torch.equal(model(ids, mask, hidden_state_index=3), final)
        penultimate = model(ids, mask, hidden_state_index=-2)
        assert torch.equal(model(ids, mask, hidden_state_index=2), penultimate)
        assert not torch.equal(penultimate, final)
        embeddings = model(ids, mask, hidden_state_index=0)
        assert torch.equal(model(ids, mask, hidden_state_index=-4), embeddings)
