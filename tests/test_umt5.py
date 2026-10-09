import json
from pathlib import Path

import pytest
import torch

from imagehub.models.umt5 import UMT5EncoderModel

FIXTURES = Path(__file__).parent / "fixtures" / "auraflow_configs"


def test_matches_published_tensor_layout() -> None:
    config = json.loads((FIXTURES / "text_encoder_config.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURES / "text_encoder_tensors.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        state = UMT5EncoderModel(config).state_dict()

    assert sorted(state) == sorted(expected)
    assert {name: list(value.shape) for name, value in state.items()} == expected
    assert "encoder.block.23.layer.0.SelfAttention.relative_attention_bias.weight" in state


@pytest.mark.parametrize(
    ("option", "value"),
    [("feed_forward_proj", "relu"), ("dense_act_fn", "relu"), ("is_gated_act", False)],
)
def test_rejects_unsupported_options(option: str, value: object) -> None:
    config = json.loads((FIXTURES / "text_encoder_config.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="Unsupported UMT5"):
        UMT5EncoderModel(config | {option: value})
