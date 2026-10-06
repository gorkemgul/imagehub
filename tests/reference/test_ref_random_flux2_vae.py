"""FLUX.2 VAE structural parity with random weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.autoencoder_kl_flux2 import (
    AutoencoderKLFlux2,
    denormalize_latents,
    normalize_latents,
    patchify_latents,
    unpatchify_latents,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

CONFIG_PATH = Path(__file__).parents[1] / "fixtures/flux2_klein_4b_configs/vae_config.json"
PUBLISHED = {
    name: value
    for name, value in json.loads(CONFIG_PATH.read_text(encoding="utf-8")).items()
    if not name.startswith("_")
}
TINY = PUBLISHED | {
    "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
    "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
    "block_out_channels": [8, 16],
    "layers_per_block": 1,
    "latent_channels": 4,
    "norm_num_groups": 4,
    "sample_size": 64,
}


def test_flux2_vae_published_layout():
    with torch.device("meta"):
        expected = diffusers.AutoencoderKLFlux2(**PUBLISHED).state_dict()
        actual = AutoencoderKLFlux2(PUBLISHED).state_dict()
    assert sorted(actual) == sorted(expected)
    assert {name: tuple(value.shape) for name, value in actual.items()} == {
        name: tuple(value.shape) for name, value in expected.items()
    }


@pytest.mark.parametrize("config", [TINY, PUBLISHED], ids=["tiny", "published"])
def test_flux2_vae_random_weights_exact(config):
    torch.manual_seed(0)
    reference = diffusers.AutoencoderKLFlux2(**config).eval()
    with torch.no_grad():
        reference.bn.running_mean.copy_(torch.randn_like(reference.bn.running_mean))
        reference.bn.running_var.copy_(torch.rand_like(reference.bn.running_var) + 0.5)
    model = AutoencoderKLFlux2(config).eval()
    model.load_checkpoint(reference.state_dict())

    image = torch.randn(2, config["in_channels"], 64, 64)
    latent_size = 64 // (2 ** (len(config["block_out_channels"]) - 1))
    patch_height = latent_size // config["patch_size"][0]
    patch_width = latent_size // config["patch_size"][1]
    packed_channels = config["latent_channels"] * config["patch_size"][0] * config["patch_size"][1]
    packed = torch.randn(2, patch_height * patch_width, packed_channels)

    with torch.inference_mode():
        expected_latents = reference.encode(image).latent_dist.mode()
        expected_encoded = diffusers.Flux2KleinPipeline._patchify_latents(expected_latents)
        mean = reference.bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(reference.bn.running_var.view(1, -1, 1, 1) + config["batch_norm_eps"])
        expected_encoded = (expected_encoded - mean) / std

        expected_patched = packed.permute(0, 2, 1).reshape(
            2, packed_channels, patch_height, patch_width
        )
        expected_patched = expected_patched * std + mean
        expected_unpatched = diffusers.Flux2KleinPipeline._unpatchify_latents(expected_patched)
        expected_decoded = reference.decode(expected_unpatched).sample

        actual_encoded = normalize_latents(
            patchify_latents(model.encode(image)),
            model.bn.running_mean,
            model.bn.running_var,
            model.batch_norm_eps,
        )
        actual_patched = packed.permute(0, 2, 1).reshape(
            2, packed_channels, patch_height, patch_width
        )
        actual_patched = denormalize_latents(
            actual_patched,
            model.bn.running_mean,
            model.bn.running_var,
            model.batch_norm_eps,
        )
        actual_decoded = model.decode(unpatchify_latents(actual_patched))

    torch.testing.assert_close(actual_decoded, expected_decoded, atol=0, rtol=0)
    torch.testing.assert_close(actual_encoded, expected_encoded, atol=0, rtol=0)
