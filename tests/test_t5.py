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


def test_loads_a_sequence_to_sequence_checkpoint_as_an_encoder(tmp_path) -> None:
    from imagehub.checkpoints import save_safetensors

    config = {
        "vocab_size": 11,
        "d_model": 16,
        "d_kv": 4,
        "d_ff": 32,
        "num_layers": 2,
        "num_heads": 4,
        "relative_attention_num_buckets": 8,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.manual_seed(0)
    state = T5EncoderModel(config).state_dict()
    full = dict(state) | {
        "encoder.embed_tokens.weight": state["shared.weight"].clone(),
        "decoder.embed_tokens.weight": torch.zeros(11, 16),
        "decoder.block.0.layer.0.SelfAttention.q.weight": torch.zeros(16, 16),
        "lm_head.weight": torch.zeros(11, 16),
    }
    save_safetensors(full, tmp_path / "model.safetensors")

    model = T5EncoderModel.from_pretrained(tmp_path)

    assert torch.equal(model.shared.weight, state["shared.weight"])
    assert not any(name.startswith(("decoder.", "lm_head.")) for name in model.state_dict())
