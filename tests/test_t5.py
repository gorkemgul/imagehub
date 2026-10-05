import json
from pathlib import Path

import pytest
import torch

from imagehub.models.t5 import T5EncoderModel

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_PATH = FIXTURES / "sd35_configs" / "medium_text_encoder_3_config.json"
TENSORS_PATH = FIXTURES / "sd35_t5_tensors.json"


def test_matches_sd35_medium_tensor_layout() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_names = json.loads(TENSORS_PATH.read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = T5EncoderModel(config)
    state = model.state_dict()

    assert sorted(state) == expected_names
    assert list(state["shared.weight"].shape) == [32128, 4096]
    assert list(state["encoder.final_layer_norm.weight"].shape) == [4096]
    for layer in range(24):
        prefix = f"encoder.block.{layer}"
        assert list(state[f"{prefix}.layer.0.SelfAttention.q.weight"].shape) == [4096, 4096]
        assert list(state[f"{prefix}.layer.0.SelfAttention.k.weight"].shape) == [4096, 4096]
        assert list(state[f"{prefix}.layer.0.SelfAttention.v.weight"].shape) == [4096, 4096]
        assert list(state[f"{prefix}.layer.0.SelfAttention.o.weight"].shape) == [4096, 4096]
        assert list(state[f"{prefix}.layer.0.layer_norm.weight"].shape) == [4096]
        assert list(state[f"{prefix}.layer.1.DenseReluDense.wi_0.weight"].shape) == [
            10240,
            4096,
        ]
        assert list(state[f"{prefix}.layer.1.DenseReluDense.wi_1.weight"].shape) == [
            10240,
            4096,
        ]
        assert list(state[f"{prefix}.layer.1.DenseReluDense.wo.weight"].shape) == [
            4096,
            10240,
        ]
        assert list(state[f"{prefix}.layer.1.layer_norm.weight"].shape) == [4096]
    assert list(
        state["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"].shape
    ) == [32, 64]


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("feed_forward_proj", "relu"),
        ("dense_act_fn", "relu"),
        ("is_gated_act", False),
    ],
)
def test_rejects_unsupported_options(option: str, value: object) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="Unsupported T5"):
        T5EncoderModel(config | {option: value})
