import json
from pathlib import Path

import torch

from imagehub.models.krea2_transformer import Krea2Transformer2DModel

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_PATH = FIXTURES / "krea2_configs" / "transformer_config.json"
TENSORS_PATH = FIXTURES / "krea2_transformer_tensors.json"
CHECKPOINT_INDEX = (
    Path(__file__).parents[1]
    / "checkpoints"
    / "krea2-turbo"
    / "transformer"
    / "diffusion_pytorch_model.safetensors.index.json"
)


def test_matches_krea2_transformer_tensor_layout():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = json.loads(TENSORS_PATH.read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = Krea2Transformer2DModel(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}

    assert actual == expected
    if CHECKPOINT_INDEX.exists():
        index = json.loads(CHECKPOINT_INDEX.read_text(encoding="utf-8"))
        assert sorted(actual) == sorted(index["weight_map"])
