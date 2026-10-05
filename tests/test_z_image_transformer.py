import json
from pathlib import Path

import pytest
import torch

from imagehub.models.z_image_transformer import ZImageTransformer2DModel

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_PATH = FIXTURES / "z_image_turbo_configs" / "transformer_config.json"
TENSORS_PATH = FIXTURES / "z_image_turbo_transformer_tensors.json"


def test_matches_z_image_turbo_tensor_layout():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_names = json.loads(TENSORS_PATH.read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = ZImageTransformer2DModel(config)
    state = model.state_dict()

    assert sorted(state) == expected_names
    assert list(state["all_x_embedder.2-1.weight"].shape) == [3840, 64]
    assert list(state["all_x_embedder.2-1.bias"].shape) == [3840]
    assert list(state["all_final_layer.2-1.linear.weight"].shape) == [64, 3840]
    assert list(state["all_final_layer.2-1.linear.bias"].shape) == [64]
    assert list(state["all_final_layer.2-1.adaLN_modulation.1.weight"].shape) == [3840, 256]
    assert list(state["t_embedder.mlp.0.weight"].shape) == [1024, 256]
    assert list(state["t_embedder.mlp.2.weight"].shape) == [256, 1024]
    assert list(state["cap_embedder.0.weight"].shape) == [2560]
    assert list(state["cap_embedder.1.weight"].shape) == [3840, 2560]
    assert list(state["x_pad_token"].shape) == [1, 3840]
    assert list(state["cap_pad_token"].shape) == [1, 3840]
    for layer in range(30):
        prefix = f"layers.{layer}"
        assert list(state[f"{prefix}.attention.to_q.weight"].shape) == [3840, 3840]
        assert list(state[f"{prefix}.attention.to_k.weight"].shape) == [3840, 3840]
        assert list(state[f"{prefix}.attention.to_v.weight"].shape) == [3840, 3840]
        assert list(state[f"{prefix}.attention.to_out.0.weight"].shape) == [3840, 3840]
        assert list(state[f"{prefix}.attention.norm_q.weight"].shape) == [128]
        assert list(state[f"{prefix}.attention.norm_k.weight"].shape) == [128]
        assert list(state[f"{prefix}.attention_norm1.weight"].shape) == [3840]
        assert list(state[f"{prefix}.ffn_norm1.weight"].shape) == [3840]
        assert list(state[f"{prefix}.attention_norm2.weight"].shape) == [3840]
        assert list(state[f"{prefix}.ffn_norm2.weight"].shape) == [3840]
        assert list(state[f"{prefix}.feed_forward.w1.weight"].shape) == [10240, 3840]
        assert list(state[f"{prefix}.feed_forward.w3.weight"].shape) == [10240, 3840]
        assert list(state[f"{prefix}.feed_forward.w2.weight"].shape) == [3840, 10240]
        assert list(state[f"{prefix}.adaLN_modulation.0.weight"].shape) == [15360, 256]
    for layer in range(2):
        prefix = f"noise_refiner.{layer}"
        assert list(state[f"{prefix}.adaLN_modulation.0.bias"].shape) == [15360]
    for layer in range(2):
        prefix = f"context_refiner.{layer}"
        assert list(state[f"{prefix}.attention.to_q.weight"].shape) == [3840, 3840]
        assert list(state[f"{prefix}.attention.to_out.0.weight"].shape) == [3840, 3840]
        assert list(state[f"{prefix}.feed_forward.w1.weight"].shape) == [10240, 3840]
        assert list(state[f"{prefix}.attention_norm1.weight"].shape) == [3840]


TINY_TRANSFORMER = {
    "all_patch_size": [2],
    "all_f_patch_size": [1],
    "in_channels": 8,
    "dim": 32,
    "n_layers": 3,
    "n_refiner_layers": 2,
    "n_heads": 2,
    "n_kv_heads": 2,
    "norm_eps": 1e-5,
    "qk_norm": True,
    "cap_feat_dim": 24,
    "siglip_feat_dim": None,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
    "axes_dims": [4, 4, 8],
    "axes_lens": [64, 32, 32],
}


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("in_channels", 15),
        ("qk_norm", False),
        ("siglip_feat_dim", 8),
    ],
)
def test_rejects_unsupported_options(option, value):
    with pytest.raises(ValueError, match="Unsupported Z-Image transformer"):
        ZImageTransformer2DModel(TINY_TRANSFORMER | {option: value})


def test_forward_shapes_and_rejects_omni_options():
    model = ZImageTransformer2DModel(TINY_TRANSFORMER).eval()
    x = [torch.randn(8, 1, 16, 16), torch.randn(8, 1, 12, 8)]
    t = torch.tensor([875.0, 250.0])
    cap = [torch.randn(5, 24), torch.randn(9, 24)]
    with torch.inference_mode():
        out = model(x, t, cap)
    assert [list(item.shape) for item in out] == [[8, 1, 16, 16], [8, 1, 12, 8]]
    for options in (
        {"controlnet_block_samples": {0: torch.randn(1, 32)}},
        {"siglip_feats": [[torch.randn(4, 4, 8)]]},
        {"image_noise_mask": [[1, 1]]},
    ):
        with pytest.raises(ValueError, match="text-to-image"):
            model(x, t, cap, **options)
