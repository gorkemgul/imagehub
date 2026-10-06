import json
from pathlib import Path

import pytest
import torch

from imagehub.models.qwen3 import Qwen3Model

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_PATHS = (
    FIXTURES / "z_image_turbo_configs" / "text_encoder_config.json",
    FIXTURES / "z_image_configs" / "text_encoder_config.json",
)
TENSORS_PATH = FIXTURES / "z_image_turbo_text_encoder_tensors.json"


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_matches_z_image_tensor_layout(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_names = json.loads(TENSORS_PATH.read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = Qwen3Model(config)
    state = model.state_dict()

    assert sorted(state) == expected_names
    assert list(state["model.embed_tokens.weight"].shape) == [151936, 2560]
    assert list(state["model.norm.weight"].shape) == [2560]
    for layer in range(36):
        prefix = f"model.layers.{layer}"
        assert list(state[f"{prefix}.input_layernorm.weight"].shape) == [2560]
        assert list(state[f"{prefix}.post_attention_layernorm.weight"].shape) == [2560]
        assert list(state[f"{prefix}.self_attn.q_proj.weight"].shape) == [4096, 2560]
        assert list(state[f"{prefix}.self_attn.k_proj.weight"].shape) == [1024, 2560]
        assert list(state[f"{prefix}.self_attn.v_proj.weight"].shape) == [1024, 2560]
        assert list(state[f"{prefix}.self_attn.o_proj.weight"].shape) == [2560, 4096]
        assert list(state[f"{prefix}.self_attn.q_norm.weight"].shape) == [128]
        assert list(state[f"{prefix}.self_attn.k_norm.weight"].shape) == [128]
        assert list(state[f"{prefix}.mlp.gate_proj.weight"].shape) == [9728, 2560]
        assert list(state[f"{prefix}.mlp.up_proj.weight"].shape) == [9728, 2560]
        assert list(state[f"{prefix}.mlp.down_proj.weight"].shape) == [2560, 9728]


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("hidden_act", "gelu"),
        ("attention_bias", True),
        ("rope_scaling", {"rope_type": "linear", "factor": 2.0}),
        ("use_sliding_window", True),
    ],
)
def test_rejects_unsupported_options(option, value):
    config = json.loads(CONFIG_PATHS[0].read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="Unsupported Qwen3"):
        Qwen3Model(config | {option: value})


def test_causal_lm_head_is_ignored_when_loading_text_encoder():
    config = json.loads(CONFIG_PATHS[0].read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = Qwen3Model(config)
    state = dict(model.state_dict())
    state["lm_head.weight"] = torch.empty(config["vocab_size"], config["hidden_size"])
    assert "lm_head.weight" not in model.prepare_state_dict(state)
