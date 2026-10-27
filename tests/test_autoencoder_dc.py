import json

import pytest
import torch
from fixture_files import fixture_path

from imagehub.models.autoencoder_dc import AutoencoderDC

FIXTURES = fixture_path("sana_configs")
TINY = {
    "latent_channels": 8,
    "attention_head_dim": 8,
    "encoder_block_types": ["ResBlock", "EfficientViTBlock", "EfficientViTBlock"],
    "decoder_block_types": ["ResBlock", "EfficientViTBlock", "EfficientViTBlock"],
    "encoder_block_out_channels": [8, 16, 16],
    "decoder_block_out_channels": [8, 16, 16],
    "encoder_layers_per_block": [1, 1, 1],
    "decoder_layers_per_block": [1, 1, 1],
    "encoder_qkv_multiscales": [[], [5], [5]],
    "decoder_qkv_multiscales": [[], [5], [5]],
    "scaling_factor": 0.5,
}


def test_matches_published_tensor_layout():
    config = json.loads((FIXTURES / "1600m" / "vae_config.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURES / "vae_tensors.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = AutoencoderDC(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == expected
    assert model.scaling_factor == 0.41407


def test_tiny_shapes():
    torch.manual_seed(0)
    model = AutoencoderDC(TINY).eval()
    with torch.inference_mode():
        latents = model.encode(torch.randn(1, 3, 32, 48))
        image = model.decode(latents)
    assert latents.shape == (1, 8, 8, 12)
    assert image.shape == (1, 3, 32, 48)
    assert model.scaling_factor == 0.5


@pytest.mark.parametrize(
    "override",
    [
        {"upsample_block_type": "pixel_shuffle"},
        {"downsample_block_type": "pixel_unshuffle"},
        {"decoder_norm_types": "batch_norm"},
        {"decoder_block_types": "Bogus"},
        {"decoder_layers_per_block": [0, 1, 1]},
    ],
)
def test_unsupported_option_rejected(override):
    with pytest.raises(ValueError, match="Unsupported DC-AE"):
        AutoencoderDC(TINY | override)
