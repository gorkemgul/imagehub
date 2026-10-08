import json
from pathlib import Path

import torch

from imagehub.models.longcat_image_transformer import LongCatImageTransformer2DModel

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_PATH = FIXTURES / "longcat_image_configs" / "transformer_config.json"
TENSORS_PATH = FIXTURES / "longcat_image_transformer_tensors.json"


def test_matches_longcat_image_transformer_tensor_layout():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = json.loads(TENSORS_PATH.read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = LongCatImageTransformer2DModel(config)
    state = model.state_dict()

    assert {name: list(value.shape) for name, value in state.items()} == expected
    assert list(state["time_embed.timestep_embedder.linear_1.weight"].shape) == [3072, 256]
    assert list(state["time_embed.timestep_embedder.linear_2.weight"].shape) == [3072, 3072]
    assert list(state["context_embedder.weight"].shape) == [3072, 3584]
    assert list(state["x_embedder.weight"].shape) == [3072, 64]
    assert list(state["proj_out.weight"].shape) == [64, 3072]
    for layer in range(10):
        prefix = f"transformer_blocks.{layer}"
        assert list(state[f"{prefix}.attn.to_q.weight"].shape) == [3072, 3072]
        assert list(state[f"{prefix}.attn.add_q_proj.weight"].shape) == [3072, 3072]
        assert list(state[f"{prefix}.ff.net.0.proj.weight"].shape) == [12288, 3072]
    for layer in range(20):
        prefix = f"single_transformer_blocks.{layer}"
        assert list(state[f"{prefix}.attn.to_q.weight"].shape) == [3072, 3072]
        assert list(state[f"{prefix}.proj_mlp.weight"].shape) == [12288, 3072]
        assert list(state[f"{prefix}.proj_out.weight"].shape) == [3072, 15360]
