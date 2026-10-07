import json
from pathlib import Path

import torch

from imagehub.models.cogview4_transformer import CogView4Transformer2DModel

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_PATH = FIXTURES / "cogview4_configs" / "transformer_config.json"
TENSORS_PATH = FIXTURES / "cogview4_transformer_tensors.json"


def test_matches_cogview4_transformer_tensor_layout():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = json.loads(TENSORS_PATH.read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = CogView4Transformer2DModel(config)

    assert {name: list(value.shape) for name, value in model.state_dict().items()} == expected
