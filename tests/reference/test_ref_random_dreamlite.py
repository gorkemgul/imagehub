"""DreamLite native component parity against random diffusers weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.autoencoder_tiny import AutoencoderTiny
from imagehub.models.dreamlite import DreamLiteUNetModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

TINY_UNET = {
    "sample_size": 8,
    "in_channels": 4,
    "out_channels": 4,
    "block_out_channels": [32, 64, 64],
    "layers_per_block": 1,
    "attention_head_dim": [2, 4, 4],
    "cross_attention_dim": 48,
    "transformer_layers_per_block": [1, 1, 1],
    "down_block_types": [
        "CrossAttnDownRemoveSelfAttnBlock2D",
        "CrossAttnDownRemoveSelfAttnBlock2D",
        "CrossAttnDownBlock2D",
    ],
    "up_block_types": [
        "CrossAttnUpBlock2D",
        "CrossAttnUpRemoveSelfAttnBlock2DV1",
        "UpBlock2D",
    ],
    "norm_num_groups": 8,
    "encoder_hid_dim": 32,
    "encoder_hid_dim_type": "text_proj_rms",
    "addition_embed_type": "time",
    "addition_time_embed_dim": 16,
    "projection_class_embeddings_input_dim": 32,
    "num_kv_heads": 1,
    "ff_mult": 3,
    "num_mid_layers": 1,
    "use_linear_projection": True,
    "use_sep_conv": True,
}
TINY_VAE = {
    "encoder_block_out_channels": [8, 8, 8, 8],
    "decoder_block_out_channels": [8, 8, 8, 8],
    "num_encoder_blocks": [1, 1, 1, 1],
    "num_decoder_blocks": [1, 1, 1, 1],
}
PUBLISHED_CONFIG = Path(__file__).parents[1] / "fixtures/dreamlite_configs/base-unet_config.json"


def test_dreamlite_unet_random_weights_exact():
    torch.manual_seed(0)
    reference = diffusers.DreamLiteUNetModel(**TINY_UNET).eval()
    ours = DreamLiteUNetModel(TINY_UNET).eval()
    reference.load_state_dict(ours.state_dict(), strict=True)
    sample = torch.randn(1, 4, 8, 12)
    context = torch.randn(1, 5, 32)
    time_ids = torch.tensor([[8.0, 12.0]])
    with torch.inference_mode():
        expected = reference(
            sample,
            torch.tensor(500.0),
            encoder_hidden_states=context,
            added_cond_kwargs={"time_ids": time_ids},
        ).sample
        actual = ours(sample, torch.tensor(500.0), context, time_ids=time_ids)
    assert torch.equal(actual, expected)


def test_dreamlite_published_config_random_weights_exact():
    config = json.loads(PUBLISHED_CONFIG.read_text())
    torch.manual_seed(2)
    reference = diffusers.DreamLiteUNetModel.from_config(config).eval()
    ours = DreamLiteUNetModel(config).eval()
    reference.load_state_dict(ours.state_dict(), strict=True)
    sample = torch.randn(1, 4, 8, 12)
    context = torch.randn(1, 5, 2048)
    time_ids = torch.tensor([[64.0, 96.0]])
    with torch.inference_mode():
        expected = reference(
            sample,
            torch.tensor(500.0),
            encoder_hidden_states=context,
            added_cond_kwargs={"time_ids": time_ids},
        ).sample
        actual = ours(sample, torch.tensor(500.0), context, time_ids=time_ids)
    assert torch.equal(actual, expected)


def test_dreamlite_tiny_vae_random_weights_exact():
    torch.manual_seed(1)
    reference = diffusers.AutoencoderTiny(**TINY_VAE).eval()
    ours = AutoencoderTiny(TINY_VAE).eval()
    reference.load_state_dict(ours.state_dict(), strict=True)
    image = torch.randn(1, 3, 16, 24)
    latents = torch.randn(1, 4, 2, 3)
    with torch.inference_mode():
        assert torch.equal(ours.encode(image), reference.encode(image).latents)
        assert torch.equal(ours.decode(latents), reference.decode(latents).sample)
