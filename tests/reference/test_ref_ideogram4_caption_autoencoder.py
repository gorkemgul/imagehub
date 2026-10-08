"""Ideogram 4 caption verifier and autoencoder parity."""

import json

import pytest
import torch

from imagehub.caption_verifier import CaptionVerifier
from imagehub.models.autoencoder_ideogram4 import AutoencoderIdeogram4, get_latent_norm
from imagehub.models.autoencoder_kl_flux2 import AutoencoderKLFlux2

pytestmark = pytest.mark.reference
reference_autoencoder = pytest.importorskip("ideogram4.autoencoder")
reference_caption = pytest.importorskip("ideogram4.caption_verifier")
reference_latent_norm = pytest.importorskip("ideogram4.latent_norm")

CONFIG = {
    "sample_size": 16,
    "in_channels": 3,
    "out_channels": 3,
    "block_out_channels": [32, 32, 32, 32],
    "layers_per_block": 1,
    "latent_channels": 4,
    "norm_num_groups": 32,
    "batch_norm_eps": 1e-4,
    "batch_norm_momentum": 0.1,
    "patch_size": [2, 2],
}
CAPTIONS = (
    '{"high_level_description":"A red fox",'
    '"style_description":{"aesthetics":"natural","lighting":"soft",'
    '"photo":"wildlife","medium":"digital","color_palette":["#AABBCC"]},'
    '"compositional_deconstruction":{"background":"snow","elements":[]}}',
    "not json",
    json.dumps(
        {
            "style_description": {"photo": "x", "art_style": "y"},
            "compositional_deconstruction": {
                "background": "x",
                "elements": [
                    {
                        "type": "obj",
                        "bbox": [1001, 20, 4, 3],
                        "desc": "x",
                        "color_palette": ["red"],
                    }
                ],
            },
        }
    ),
    '{"compositional_deconstruction":{"background":"café","elements":[]},'
    '"high_level_description":"caf\\u00e9"}',
)


def test_ideogram4_caption_verifier_exact():
    actual = CaptionVerifier()
    expected = reference_caption.CaptionVerifier()
    for caption in CAPTIONS:
        assert actual.verify_raw(caption) == expected.verify_raw(caption)


def test_ideogram4_autoencoder_and_latent_norm_exact():
    params = reference_autoencoder.AutoEncoderParams(
        resolution=CONFIG["sample_size"],
        in_channels=CONFIG["in_channels"],
        ch=CONFIG["block_out_channels"][0],
        out_ch=CONFIG["out_channels"],
        ch_mult=[1, 1, 1, 1],
        num_res_blocks=CONFIG["layers_per_block"],
        z_channels=CONFIG["latent_channels"],
    )
    torch.manual_seed(0)
    source = AutoencoderKLFlux2(CONFIG).eval()
    model = AutoencoderIdeogram4(CONFIG).eval()
    reference = reference_autoencoder.AutoEncoder(params).eval()
    model.load_checkpoint(source.state_dict())
    converted = reference_autoencoder.convert_diffusers_state_dict(source.state_dict())
    reference.load_state_dict(converted)

    image = torch.randn(2, 3, 16, 16)
    latents = torch.randn(2, 4, 2, 2)
    with torch.inference_mode():
        expected_encoded = reference.encoder(image).chunk(2, dim=1)[0]
        expected_decoded = reference.decoder(latents)
        actual_encoded = model.encode(image)
        actual_decoded = model.decode(latents)

    torch.testing.assert_close(actual_encoded, expected_encoded, atol=0, rtol=0)
    torch.testing.assert_close(actual_decoded, expected_decoded, atol=0, rtol=0)
    actual_shift, actual_scale = get_latent_norm()
    expected_shift, expected_scale = reference_latent_norm.get_latent_norm()
    torch.testing.assert_close(actual_shift, expected_shift, atol=0, rtol=0)
    torch.testing.assert_close(actual_scale, expected_scale, atol=0, rtol=0)
