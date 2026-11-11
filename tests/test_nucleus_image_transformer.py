import json
from pathlib import Path

import torch

from imagehub.checkpoints import save_safetensors
from imagehub.models.nucleus_image_transformer import NucleusImageTransformer2DModel

FIXTURES = Path(__file__).parent / "fixtures/nucleus_image_configs"


def test_matches_published_transformer_tensor_layout() -> None:
    expected = {}
    for filename in (FIXTURES / "transformer").glob("*.header.json"):
        expected.update(json.loads(filename.read_text()))
    with torch.device("meta"):
        actual = NucleusImageTransformer2DModel(
            json.loads((FIXTURES / "transformer/config.json").read_text())
        ).state_dict()
    assert set(actual) == set(expected)
    for name, metadata in expected.items():
        assert list(actual[name].shape) == metadata


def test_pretrained_rebuilds_computed_rope_buffers_inside_meta_context(tmp_path) -> None:
    config = json.loads((FIXTURES / "transformer/config.json").read_text())
    config.update(
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        joint_attention_dim=16,
        axes_dims_rope=[2, 2, 4],
        mlp_ratio=12.0,
        in_channels=4,
        out_channels=4,
        capacity_factors=[0.0],
        moe_enabled=False,
    )
    model = NucleusImageTransformer2DModel(config)
    component = tmp_path / "transformer"
    component.mkdir()
    (component / "config.json").write_text(json.dumps(config))
    save_safetensors(model.state_dict(), component / "diffusion_pytorch_model.safetensors")
    with torch.device("meta"):
        loaded = NucleusImageTransformer2DModel.from_pretrained(component)
    assert loaded.pos_embed.freqs.device.type == "cpu"
    assert loaded.pos_embed.neg_freqs.device.type == "cpu"
