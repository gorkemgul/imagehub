import json
from pathlib import Path

import pytest
import torch

from imagehub.models.qwen_image21_transformer import QwenImage21Transformer2DModel

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_PATH = FIXTURES / "qwen_image21_configs" / "transformer_config.json"
TENSORS_PATH = FIXTURES / "qwen_image21_transformer_tensors.json"


def test_matches_qwen_image21_transformer_tensor_layout():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_names = json.loads(TENSORS_PATH.read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = QwenImage21Transformer2DModel(config)
    state = model.state_dict()

    assert sorted(state) == expected_names
    assert list(state["time_text_embed.timestep_embedder.linear_1.weight"].shape) == [4096, 256]
    assert list(state["time_text_embed.timestep_embedder.linear_2.weight"].shape) == [4096, 4096]
    assert list(state["txt_in.text_norm.weight"].shape) == [4096]
    assert list(state["txt_in.in_layer.weight"].shape) == [4096, 4096]
    assert list(state["img_in.weight"].shape) == [4096, 64]
    assert list(state["modulation.1.weight"].shape) == [16384, 4096]
    assert list(state["norm_out.linear.weight"].shape) == [4096, 4096]
    assert list(state["proj_out.weight"].shape) == [64, 4096]
    for layer in range(32):
        prefix = f"transformer_blocks.{layer}"
        assert list(state[f"{prefix}.attn.to_q.weight"].shape) == [4096, 4096]
        assert list(state[f"{prefix}.attn.to_k.weight"].shape) == [4096, 4096]
        assert list(state[f"{prefix}.attn.to_v.weight"].shape) == [4096, 4096]
        assert list(state[f"{prefix}.attn.to_out.0.weight"].shape) == [4096, 4096]
        assert list(state[f"{prefix}.attn.norm_q.weight"].shape) == [128]
        assert list(state[f"{prefix}.attn.norm_k.weight"].shape) == [128]
        assert list(state[f"{prefix}.img_mlp.proj.weight"].shape) == [12288, 4096]
        assert list(state[f"{prefix}.img_mlp.out.weight"].shape) == [4096, 12288]
        assert list(state[f"{prefix}.img_mlp.gate_layer.weight"].shape) == [12288, 4096]


def test_rejects_condition_images():
    config = {
        "patch_size": 1,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "context_in_dim": 12,
        "mlp_ratio": 3,
        "axes_dims_rope": [2, 2, 4],
        "eps": 1e-6,
        "causal_condition": True,
    }
    model = QwenImage21Transformer2DModel(config)
    with pytest.raises(ValueError, match="condition images"):
        model(
            hidden_states=torch.randn(1, 16, 4),
            encoder_hidden_states=torch.randn(1, 4, 12),
            timestep=torch.tensor([0.5]),
            img_shapes=[[(1, 2, 2), (1, 3, 4)]],
            img_mask=torch.tensor([[False, False, False, False, False, True, True, True]]),
        )
