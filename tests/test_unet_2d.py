import json
import math
from pathlib import Path

import pytest
import torch
from helpers import (
    SDXL_UNET,
    TINY_UNET_CONFIG,
    meta_shapes,
    published_shapes,
    sdxl_published_shapes,
    ssd1b_published_shapes,
)

from imagehub.models.layers import timestep_embedding
from imagehub.models.unet_2d import UNet2DConditionModel

FIXTURES = Path(__file__).parent / "fixtures"


def test_matches_published_tensor_layout():
    assert meta_shapes(UNet2DConditionModel, {}) == published_shapes("unet")


def test_matches_sdxl_published_tensor_layout():
    assert meta_shapes(UNet2DConditionModel, SDXL_UNET) == sdxl_published_shapes("unet")


@pytest.mark.parametrize(
    ("config_path", "family"),
    [
        ("playground_v25_configs/unet.json", "sdxl"),
        ("sdxl_configs/unet.json", "sdxl"),
        ("ssd1b_configs/unet.json", "ssd1b"),
        ("sd1_configs/v1-5/unet.json", "sd1"),
        ("sd1_configs/v1-1/unet.json", "sd1"),
    ],
)
def test_real_release_configs_match_published_tensor_layout(config_path, family):
    config = json.loads((FIXTURES / config_path).read_text(encoding="utf-8"))
    fixtures = {
        "sdxl": sdxl_published_shapes,
        "ssd1b": ssd1b_published_shapes,
        "sd1": published_shapes,
    }
    assert meta_shapes(UNet2DConditionModel, config) == fixtures[family]("unet")


def test_ssd1b_mid_block_has_one_resnet_and_no_attention():
    config = TINY_UNET_CONFIG | {"mid_block_type": "UNetMidBlock2D"}
    model = UNet2DConditionModel(config)
    assert len(model.mid_block.resnets) == 1
    assert model.mid_block.attentions is None
    sample = torch.randn(2, 4, 8, 8)
    with torch.inference_mode():
        out = model(sample, torch.tensor(981), torch.randn(2, 7, 32))
    assert out.shape == sample.shape


def test_nested_transformer_layers_apply_per_resnet():
    config = TINY_UNET_CONFIG | {
        "block_out_channels": [32, 64],
        "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D"],
        "up_block_types": ["CrossAttnUpBlock2D", "UpBlock2D"],
        "transformer_layers_per_block": [1, [1, 2]],
        "reverse_transformer_layers_per_block": [[2, 3, 4], 1],
        "mid_block_type": "UNetMidBlock2D",
    }
    model = UNet2DConditionModel(config)
    down = model.down_blocks[1].attentions
    assert [len(block.transformer_blocks) for block in down] == [1, 2]
    up = model.up_blocks[0].attentions
    assert [len(block.transformer_blocks) for block in up] == [2, 3, 4]


def test_mismatched_transformer_layer_depths_raise():
    config = TINY_UNET_CONFIG | {
        "block_out_channels": [32, 64],
        "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D"],
        "up_block_types": ["CrossAttnUpBlock2D", "UpBlock2D"],
        "transformer_layers_per_block": [1, [1, 2, 3]],
    }
    with pytest.raises(ValueError, match="transformer_layers_per_block"):
        UNet2DConditionModel(config)
    with pytest.raises(ValueError, match="reverse_transformer_layers_per_block"):
        UNet2DConditionModel(TINY_UNET_CONFIG | {"reverse_transformer_layers_per_block": [1]})


def test_timestep_embedding_closed_form():
    emb = timestep_embedding(torch.tensor([3]), 4, flip_sin_to_cos=True, shift=0)
    freqs = [1.0, math.exp(-math.log(10000) / 2)]
    expected = [math.cos(3 * f) for f in freqs] + [math.sin(3 * f) for f in freqs]
    torch.testing.assert_close(emb[0], torch.tensor(expected))


@pytest.mark.parametrize("hw", [(8, 8), (9, 7)])
def test_forward_shapes(hw):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(TINY_UNET_CONFIG).eval()
    sample = torch.randn(2, 4, *hw)
    context = torch.randn(2, 77, 32)
    with torch.inference_mode():
        out = unet(sample, torch.tensor(981), context)
        batched_t = unet(sample, torch.tensor([981, 981]), context)
    assert out.shape == sample.shape
    torch.testing.assert_close(out, batched_t, rtol=0, atol=0)


def test_text_time_conditioning_requires_both_inputs():
    config = {
        "block_out_channels": [32, 64, 64],
        "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
        "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        "transformer_layers_per_block": [1, 2, 2],
        "attention_head_dim": [2, 4, 4],
        "cross_attention_dim": 32,
        "use_linear_projection": True,
        "addition_embed_type": "text_time",
        "addition_time_embed_dim": 8,
        "projection_class_embeddings_input_dim": 80,
        "layers_per_block": 1,
        "norm_num_groups": 8,
    }
    model = UNet2DConditionModel(config)
    sample = torch.randn(2, 4, 8, 8)
    context = torch.randn(2, 7, 32)
    text_embeds = torch.randn(2, 32)
    time_ids = torch.randn(2, 6)
    with pytest.raises(ValueError, match="text_embeds"):
        model(sample, torch.tensor(1), context, time_ids=time_ids)
    with pytest.raises(ValueError, match="time_ids"):
        model(sample, torch.tensor(1), context, text_embeds=text_embeds)
    assert (
        model(sample, torch.tensor(1), context, text_embeds=text_embeds, time_ids=time_ids).shape
        == sample.shape
    )


def test_timestep_conditioning_shape_and_requirement():
    model = UNet2DConditionModel(TINY_UNET_CONFIG | {"time_cond_proj_dim": 16})
    sample = torch.randn(2, 4, 8, 8)
    context = torch.randn(2, 7, 32)
    timestep_cond = torch.randn(2, 16)
    with torch.inference_mode():
        unconditioned = model(sample, torch.tensor(999), context)
        output = model(sample, torch.tensor(999), context, timestep_cond=timestep_cond)
    assert output.shape == unconditioned.shape == sample.shape
    assert not torch.equal(output, unconditioned)


@pytest.mark.parametrize(
    "option",
    [
        {"class_embed_type": "timestep"},
        {"num_class_embeds": 10},
        {"addition_embed_type": "text"},
        {"only_cross_attention": True},
        {"upcast_attention": True},
        {"center_input_sample": True},
        {"act_fn": "gelu"},
        {"conv_in_kernel": 5},
        {"resnet_time_scale_shift": "ada_group"},
        {"attention_mask": True},
        {"unknown_option": True},
    ],
)
def test_rejects_unsupported_options(option):
    with pytest.raises(ValueError, match="Unsupported U-Net setting"):
        UNet2DConditionModel(option)


def test_explicit_num_attention_heads_override_historical_name():
    config = TINY_UNET_CONFIG | {"num_attention_heads": [2, 4, 4, 4]}
    model = UNet2DConditionModel(config)
    attention = model.down_blocks[1].attentions
    assert attention is not None
    assert attention[0].transformer_blocks[0].attn1.heads == 4
