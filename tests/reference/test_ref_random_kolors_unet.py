"""Kolors U-Net (``text_proj`` encoder projection plus SDXL text/time conditioning) vs diffusers."""

import pytest
import torch

from imagehub.models.unet_2d import UNet2DConditionModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

TINY_KOLORS_UNET = {
    "in_channels": 4,
    "out_channels": 4,
    "block_out_channels": [32, 64, 64],
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    "transformer_layers_per_block": [1, 2, 3],
    "attention_head_dim": [2, 4, 4],
    "cross_attention_dim": 32,
    "encoder_hid_dim": 48,
    "encoder_hid_dim_type": "text_proj",
    "use_linear_projection": True,
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 8,
    "projection_class_embeddings_input_dim": 48 + 6 * 8,
    "layers_per_block": 2,
    "norm_num_groups": 8,
    "sample_size": 16,
}


@pytest.mark.parametrize("hw", [(16, 16), (13, 9)])
def test_kolors_unet_random_weights(hw, record):
    torch.manual_seed(0)
    ref = diffusers.UNet2DConditionModel.from_config(TINY_KOLORS_UNET).eval()
    for parameter in ref.parameters():
        parameter.data.normal_(0, 0.02)
    ours = UNet2DConditionModel(TINY_KOLORS_UNET).eval()
    ours.load_checkpoint(ref.state_dict())
    context = torch.randn(4, 9, TINY_KOLORS_UNET["encoder_hid_dim"])
    text_embeds = torch.randn(4, TINY_KOLORS_UNET["encoder_hid_dim"])
    time_ids = torch.randn(4, 6)
    sample = torch.randn(4, 4, *hw)
    for timestep in (torch.tensor(1079.0), torch.tensor(1.0)):
        with torch.inference_mode():
            actual = ours(sample, timestep, context, text_embeds=text_embeds, time_ids=time_ids)
            expected = ref(
                sample,
                timestep,
                context,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            ).sample
        assert torch.equal(actual, expected)
    record(f"kolors_unet_random_{hw[0]}x{hw[1]}", max_abs_diff=0.0, tolerance="exact")
