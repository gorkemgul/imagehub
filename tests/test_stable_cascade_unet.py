import json
from pathlib import Path

import pytest
import torch

from imagehub.models.stable_cascade_unet import StableCascadeUNet

FIXTURES = Path(__file__).parent / "fixtures/stable_cascade_configs"
PUBLISHED = ["prior", "prior_lite", "decoder", "decoder_lite"]

PRIOR_TINY = {
    "in_channels": 4,
    "out_channels": 4,
    "conditioning_dim": 24,
    "block_out_channels": [32, 32],
    "num_attention_heads": [4, 4],
    "down_num_layers_per_block": [1, 2],
    "up_num_layers_per_block": [2, 1],
    "down_blocks_repeat_mappers": [1, 1],
    "up_blocks_repeat_mappers": [1, 1],
    "block_types_per_layer": [
        ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
        ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
    ],
    "clip_text_in_channels": 20,
    "clip_text_pooled_in_channels": 16,
    "clip_image_in_channels": 12,
    "clip_seq": 4,
    "dropout": [0.1, 0.1],
    "self_attn": True,
    "timestep_conditioning_type": ["sca", "crp"],
    "switch_level": [False],
}
DECODER_TINY = {
    "in_channels": 4,
    "out_channels": 4,
    "patch_size": 2,
    "conditioning_dim": 24,
    "block_out_channels": [8, 16, 32, 32],
    "num_attention_heads": [0, 0, 4, 4],
    "down_num_layers_per_block": [1, 1, 2, 1],
    "up_num_layers_per_block": [1, 2, 1, 1],
    "down_blocks_repeat_mappers": [1, 1, 1, 1],
    "up_blocks_repeat_mappers": [3, 3, 2, 2],
    "block_types_per_layer": [
        ["SDCascadeResBlock", "SDCascadeTimestepBlock"],
        ["SDCascadeResBlock", "SDCascadeTimestepBlock"],
        ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
        ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
    ],
    "clip_text_pooled_in_channels": 16,
    "effnet_in_channels": 6,
    "pixel_mapper_in_channels": 3,
    "dropout": [0, 0, 0.1, 0.1],
    "timestep_conditioning_type": ["sca"],
    "switch_level": None,
}


def _published(name: str):
    config = json.loads((FIXTURES / f"{name}_config.json").read_text(encoding="utf-8"))
    tensors = json.loads((FIXTURES / f"{name}_tensors.json").read_text(encoding="utf-8"))
    return config, {key: tuple(shape) for key, shape in tensors.items()}


@pytest.mark.parametrize("name", PUBLISHED)
def test_published_layout(name):
    config, expected = _published(name)
    with torch.device("meta"):
        model = StableCascadeUNet(config)
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    assert sorted(set(expected) - set(actual)) == []
    assert sorted(set(actual) - set(expected)) == []
    assert {k: (actual[k], expected[k]) for k in actual if actual[k] != expected[k]} == {}


def test_meta_construction_then_checkpoint_load_leaves_no_meta_tensors():
    source = StableCascadeUNet(PRIOR_TINY)
    with torch.device("meta"):
        model = StableCascadeUNet(PRIOR_TINY)
    model.load_checkpoint(source.state_dict())
    model.init_non_persistent_buffers()
    tensors = (*model.parameters(), *model.buffers())
    assert all(value.device.type == "cpu" for value in tensors)
    assert torch.equal(model.clip_txt_pooled_mapper.weight, source.clip_txt_pooled_mapper.weight)


def test_prior_like_forward_shape():
    model = StableCascadeUNet(PRIOR_TINY).eval()
    out = model(
        torch.randn(2, 4, 12, 13),
        torch.tensor([0.3, 0.8]),
        torch.randn(2, 1, 16),
        clip_text=torch.randn(2, 77, 20),
        clip_img=torch.randn(2, 1, 12),
    )
    assert out.shape == (2, 4, 12, 13)


def test_decoder_like_forward_shape():
    model = StableCascadeUNet(DECODER_TINY).eval()
    out = model(
        torch.randn(2, 4, 36, 38),
        torch.tensor([0.3, 0.8]),
        torch.randn(2, 1, 16),
        effnet=torch.randn(2, 6, 5, 5),
    )
    assert out.shape == (2, 4, 36, 38)


@pytest.mark.parametrize(
    "override",
    [
        {"timestep_conditioning_type": ["foo"]},
        {"block_types_per_layer": [["SDCascadeResBlock", "Other"]] * 2},
        {"down_num_layers_per_block": [1]},
        {"switch_level": [True, True]},
        {"unknown_option": 1},
        {"num_attention_heads": [0, 4]},
    ],
)
def test_unsupported_options_are_rejected(override):
    with pytest.raises(ValueError):
        StableCascadeUNet(PRIOR_TINY | override)
