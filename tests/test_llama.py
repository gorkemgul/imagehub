import json
from pathlib import Path

import pytest
import torch

from imagehub.models.llama import LlamaCache, LlamaModel

CONFIG_PATH = Path(__file__).parent / "fixtures" / "janus_pro_1b_config.json"


def test_matches_janus_pro_tensor_layout():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = LlamaModel(config)
    state = model.state_dict()

    assert list(state["embed_tokens.weight"].shape) == [102400, 2048]
    assert list(state["norm.weight"].shape) == [2048]
    for layer in range(24):
        prefix = f"layers.{layer}"
        assert list(state[f"{prefix}.input_layernorm.weight"].shape) == [2048]
        assert list(state[f"{prefix}.post_attention_layernorm.weight"].shape) == [2048]
        assert list(state[f"{prefix}.self_attn.q_proj.weight"].shape) == [2048, 2048]
        assert list(state[f"{prefix}.self_attn.k_proj.weight"].shape) == [2048, 2048]
        assert list(state[f"{prefix}.self_attn.v_proj.weight"].shape) == [2048, 2048]
        assert list(state[f"{prefix}.self_attn.o_proj.weight"].shape) == [2048, 2048]
        assert list(state[f"{prefix}.mlp.gate_proj.weight"].shape) == [5632, 2048]
        assert list(state[f"{prefix}.mlp.up_proj.weight"].shape) == [5632, 2048]
        assert list(state[f"{prefix}.mlp.down_proj.weight"].shape) == [2048, 5632]


def test_cache_is_preallocated_and_masks_unused_slots():
    config = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
    }
    model = LlamaModel(config).eval()
    cache = LlamaCache(config, max_cache_len=25)
    inputs_embeds = torch.randn(2, 4, 16)
    attention_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])

    with torch.inference_mode():
        output = model(inputs_embeds, attention_mask, cache, torch.arange(4))

    assert output.shape == (2, 4, 16)
    assert cache.keys[0].shape == (2, 2, 25, 8)
    assert cache.values[0].shape == (2, 2, 25, 8)
    assert torch.count_nonzero(cache.keys[0][:, :, 4:]) == 0
    assert cache.sequence_length == 4


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("hidden_act", "gelu"),
        ("attention_bias", True),
        ("attention_dropout", 0.1),
        ("mlp_bias", True),
        ("rope_scaling", {"rope_type": "linear", "factor": 2.0}),
    ],
)
def test_rejects_unsupported_options(option, value):
    with pytest.raises(ValueError, match="Unsupported Llama"):
        LlamaModel({option: value})
