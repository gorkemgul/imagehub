"""Qwen-Image 2.1 VAE structural parity with random weights."""

import json

import pytest
import torch
from fixture_files import restricted_fixture

from imagehub.models.autoencoder_kl_qwen_image21 import AutoencoderKLQwenImage21

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "QwenImage21Pipeline"):
    pytest.skip(
        "needs the diffusers commit pinned for Qwen-Image 2.1 (.venv-ref-qwen-image-2.1)",
        allow_module_level=True,
    )


def published_config():
    config = json.loads(
        restricted_fixture("qwen_image21_configs/vae_config.json").read_text(encoding="utf-8")
    )
    return {name: value for name, value in config.items() if not name.startswith("_")}


TINY = {
    "attn_scales": [],
    "base_dim": 4,
    "decoder_base_dim": 4,
    "dropout": 0.0,
    "in_channels": 4,
    "is_residual": True,
    "out_channels": 4,
    "patch_size": None,
    "z_dim": 4,
    "dim_mult": [1, 2, 2],
    "num_res_blocks": 1,
    "temperal_downsample": [False, False],
    "latents_mean": [0.0] * 4,
    "latents_std": [1.0] * 4,
    "scale_factor_spatial": 4,
    "scale_factor_temporal": 8,
}


def test_qwen_vae21_published_layout():
    config = published_config()
    with torch.device("meta"):
        expected = diffusers.AutoencoderKLQwenImage21(**config).state_dict()
        actual = AutoencoderKLQwenImage21(config).state_dict()
    assert sorted(actual) == sorted(expected)
    assert {name: tuple(value.shape) for name, value in actual.items()} == {
        name: tuple(value.shape) for name, value in expected.items()
    }


@pytest.mark.parametrize(
    ("config", "sample_size"),
    [(TINY, 8), (None, 64)],
    ids=["tiny", "published"],
)
def test_qwen_vae21_random_weights_exact(config, sample_size):
    config = published_config() if config is None else config
    torch.manual_seed(0)
    reference = diffusers.AutoencoderKLQwenImage21(**config).eval()
    model = AutoencoderKLQwenImage21(config).eval()
    model.load_checkpoint(reference.state_dict())

    image = torch.randn(2, config["in_channels"], sample_size, sample_size)
    latent_size = sample_size // config["scale_factor_spatial"]
    latents = torch.randn(2, config["z_dim"], latent_size, latent_size)
    with torch.inference_mode():
        expected_decoded = reference.decode(latents.unsqueeze(2)).sample[:, :, 0]
        actual_decoded = model.decode(latents)
        expected_encoded = reference.encode(image.unsqueeze(2)).latent_dist.mode()[:, :, 0]
        actual_encoded = model.encode(image)

    torch.testing.assert_close(actual_decoded, expected_decoded, atol=0, rtol=0)
    torch.testing.assert_close(actual_encoded, expected_encoded, atol=0, rtol=0)
