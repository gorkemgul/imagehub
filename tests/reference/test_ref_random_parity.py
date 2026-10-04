"""Structural parity with random weights: no checkpoint download needed."""

import pytest
import torch
from helpers import (
    SD1_UNET,
    TINY_TEXT_CONFIG,
    TINY_UNET_CONFIG,
    TINY_VAE_CONFIG,
    legacy_vae_state,
    text_state_for_ours,
)

from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIP_TEXT_DEFAULTS, CLIPTextModel
from imagehub.models.unet_2d import UNet2DConditionModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")
diffusers = pytest.importorskip("diffusers")


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("config", [TINY_TEXT_CONFIG, {}], ids=["tiny", "sd1"])
def test_clip_text_random_weights(config):
    torch.manual_seed(0)
    ref = transformers.CLIPTextModel(
        transformers.CLIPTextConfig(**(CLIP_TEXT_DEFAULTS | config))
    ).eval()
    ours = CLIPTextModel(config).eval()
    ours.load_checkpoint(text_state_for_ours(ref.state_dict()))
    ids = torch.randint(0, ours.config["vocab_size"], (3, 77))
    with torch.inference_mode():
        exact(ours(ids), ref(ids)[0])


SD1_VAE = {
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "latent_channels": 4,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "sample_size": 512,
}


@pytest.mark.parametrize(
    ("config", "latent_hw"),
    [(SD1_VAE | TINY_VAE_CONFIG, (8, 6)), (SD1_VAE, (64, 64))],
    ids=["tiny", "sd1"],
)
def test_vae_random_weights(config, latent_hw):
    torch.manual_seed(0)
    ref = diffusers.AutoencoderKL.from_config(config).eval()
    for p in ref.parameters():
        p.data.normal_(0, 0.02)
    ours = AutoencoderKL(config).eval()
    ours.load_checkpoint(legacy_vae_state(ref.state_dict()))
    z = torch.randn(1, 4, *latent_hw)
    image = torch.rand(1, 3, latent_hw[0] * 8, latent_hw[1] * 8) * 2 - 1
    with torch.inference_mode():
        exact(ours.decode(z), ref.decode(z).sample)
        exact(ours.encode(image), ref.encode(image).latent_dist.mean)


@pytest.mark.parametrize(
    ("config", "sizes"),
    [(TINY_UNET_CONFIG, [(16, 16), (13, 9)]), (SD1_UNET, [(64, 64), (65, 47)])],
    ids=["tiny", "sd1"],
)
def test_unet_random_weights(config, sizes):
    torch.manual_seed(0)
    ref = diffusers.UNet2DConditionModel.from_config(config).eval()
    for p in ref.parameters():
        p.data.normal_(0, 0.02)
    ours = UNet2DConditionModel(config).eval()
    ours.load_checkpoint(ref.state_dict())
    context = torch.randn(2, 77, config["cross_attention_dim"])
    for hw in sizes:
        sample = torch.randn(2, 4, *hw)
        for t in (torch.tensor(981), torch.tensor(1)):
            with torch.inference_mode():
                exact(ours(sample, t, context), ref(sample, t, context).sample)
