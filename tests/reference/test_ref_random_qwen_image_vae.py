"""Krea 2 Qwen-Image VAE structural parity with random weights."""

import pytest
import torch

from imagehub.models.autoencoder_kl_qwen_image import AutoencoderKLQwenImage

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

TINY = {
    "base_dim": 4,
    "z_dim": 4,
    "dim_mult": [1, 2, 2, 2],
    "num_res_blocks": 1,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "input_channels": 3,
    "latents_mean": [0.0] * 4,
    "latents_std": [1.0] * 4,
}
MEDIUM = TINY | {
    "base_dim": 8,
    "z_dim": 8,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "latents_mean": [0.0] * 8,
    "latents_std": [1.0] * 8,
}


@pytest.mark.parametrize("config", [TINY, MEDIUM], ids=["tiny", "medium"])
def test_qwen_image_vae_random_weights_exact(config):
    torch.manual_seed(0)
    reference = diffusers.AutoencoderKLQwenImage(**config).eval()
    model = AutoencoderKLQwenImage(config).eval()
    model.load_checkpoint(reference.state_dict())

    image = torch.randn(2, config["input_channels"], 1, 16, 24)
    latents = torch.randn(2, config["z_dim"], 1, 2, 3)
    with torch.inference_mode():
        expected_encoded = reference.encode(image).latent_dist.mode()
        actual_encoded = model.encode(image)
        expected_decoded = reference.decode(latents).sample
        actual_decoded = model.decode(latents)

    torch.testing.assert_close(actual_encoded, expected_encoded, atol=0, rtol=0)
    torch.testing.assert_close(actual_decoded, expected_decoded, atol=0, rtol=0)
