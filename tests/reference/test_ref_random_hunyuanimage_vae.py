"""HunyuanImage 2.1 VAE structural parity with random weights."""

import json

import pytest
import torch
from fixture_files import fixture_path

from imagehub.models.autoencoder_kl_hunyuanimage import AutoencoderKLHunyuanImage

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

FIXTURES = fixture_path("hunyuanimage_configs")
TINY = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 8,
    "block_out_channels": [32, 32, 64],
    "layers_per_block": 1,
    "spatial_compression_ratio": 4,
    "sample_size": 64,
    "scaling_factor": 0.5,
}


def _reference(config):
    return diffusers.AutoencoderKLHunyuanImage(
        **{name: value for name, value in config.items() if name != "_class_name"}
    ).eval()


def _randomize(model):
    torch.manual_seed(0)
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.05)


def test_published_layout():
    config = json.loads((FIXTURES / "vae_config.json").read_text(encoding="utf-8"))
    published = json.loads((FIXTURES / "vae_tensors.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        expected = _reference(config).state_dict()
        actual = AutoencoderKLHunyuanImage(config).state_dict()
    assert {name: list(value.shape) for name, value in expected.items()} == published
    assert {name: list(value.shape) for name, value in actual.items()} == published


def _check(config, size):
    reference = _reference(config)
    _randomize(reference)
    model = AutoencoderKLHunyuanImage(config).eval()
    model.load_checkpoint(reference.state_dict())
    ratio = config["spatial_compression_ratio"]
    image = torch.randn(2, 3, size, size + ratio)
    latents = torch.randn(2, config["latent_channels"], size // ratio, size // ratio + 1)
    with torch.inference_mode():
        assert torch.equal(model.decode(latents), reference.decode(latents, return_dict=False)[0])
        expected = reference.encode(image).latent_dist.mode()
        assert torch.equal(model.encode(image), expected)


def test_tiny_exact():
    _check(TINY, 16)


def test_published_config_exact_at_small_size():
    config = json.loads((FIXTURES / "vae_config.json").read_text(encoding="utf-8"))
    _check(config, 64)
