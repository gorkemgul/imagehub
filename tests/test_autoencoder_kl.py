import torch
from helpers import TINY_VAE_CONFIG, meta_shapes, published_shapes

from imagehub.models.autoencoder_kl import AutoencoderKL


def test_matches_published_tensor_layout():
    assert meta_shapes(AutoencoderKL, {}) == published_shapes("vae")


def test_shapes_and_scaling_factor():
    torch.manual_seed(0)
    vae = AutoencoderKL(TINY_VAE_CONFIG).eval()
    assert vae.scaling_factor == 0.18215
    assert AutoencoderKL(TINY_VAE_CONFIG | {"scaling_factor": 0.5}).scaling_factor == 0.5
    with torch.inference_mode():
        assert vae.decode(torch.randn(1, 4, 8, 6)).shape == (1, 3, 64, 48)
        assert vae.encode(torch.rand(1, 3, 64, 48) * 2 - 1).shape == (1, 4, 8, 6)
