"""Sana transformer and DC-AE random-weight parity against diffusers."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.autoencoder_dc import AutoencoderDC
from imagehub.models.sana_transformer import SanaTransformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

CONFIGS = Path(__file__).parents[1] / "fixtures" / "sana_configs"
TINY_TRANSFORMER = {
    "num_attention_heads": 2,
    "attention_head_dim": 8,
    "num_layers": 2,
    "num_cross_attention_heads": 2,
    "cross_attention_head_dim": 8,
    "cross_attention_dim": 16,
    "caption_channels": 12,
    "in_channels": 4,
    "out_channels": 4,
    "sample_size": 8,
}
TINY_VAE = {
    "latent_channels": 8,
    "attention_head_dim": 8,
    "encoder_block_types": ["ResBlock", "EfficientViTBlock", "EfficientViTBlock"],
    "decoder_block_types": ["ResBlock", "EfficientViTBlock", "EfficientViTBlock"],
    "encoder_block_out_channels": [8, 16, 16],
    "decoder_block_out_channels": [8, 16, 16],
    "encoder_layers_per_block": [1, 2, 1],
    "decoder_layers_per_block": [2, 1, 1],
    "encoder_qkv_multiscales": [[], [5], [5]],
    "decoder_qkv_multiscales": [[], [5], [5]],
    "upsample_block_type": "interpolate",
    "downsample_block_type": "Conv",
    "scaling_factor": 0.5,
}


def _config(path: str) -> dict:
    data = json.loads((CONFIGS / path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _randomize(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)


def _transformer_case(config: dict, hidden, text, timestep, mask, record, name):
    torch.manual_seed(0)
    reference = diffusers.SanaTransformer2DModel(**config).eval()
    _randomize(reference, 1)
    model = SanaTransformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())
    with torch.inference_mode():
        expected = reference(
            hidden, encoder_hidden_states=text, timestep=timestep, encoder_attention_mask=mask
        ).sample
        actual = model(hidden, text, timestep, mask)
    record(name, max_abs=float((actual - expected).abs().max()), shape=list(actual.shape))
    assert torch.equal(actual, expected)


def test_tiny_transformer_masked_non_square(record):
    generator = torch.Generator().manual_seed(2)
    hidden = torch.randn(2, 4, 6, 10, generator=generator)
    text = torch.randn(2, 7, 12, generator=generator)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0]])
    timestep = torch.tensor([800.0, 123.5])
    _transformer_case(
        TINY_TRANSFORMER, hidden, text, timestep, mask, record, "sana_tiny_transformer"
    )


def test_published_600m_transformer(record):
    config = _config("600m/transformer_config.json")
    generator = torch.Generator().manual_seed(3)
    hidden = torch.randn(2, 32, 8, 12, generator=generator)
    text = torch.randn(2, 300, config["caption_channels"], generator=generator)
    mask = torch.zeros(2, 300, dtype=torch.long)
    mask[0, :300] = 1
    mask[1, :123] = 1
    timestep = torch.tensor([999.0, 400.0])
    _transformer_case(config, hidden, text, timestep, mask, record, "sana_600m_transformer")


def _vae_case(config: dict, images, latents, record, name):
    torch.manual_seed(0)
    reference = diffusers.AutoencoderDC(**config).eval()
    _randomize(reference, 4)
    model = AutoencoderDC(config).eval()
    model.load_checkpoint(reference.state_dict())
    result = {}
    with torch.inference_mode():
        if images is not None:
            expected = reference.encode(images).latent
            actual = model.encode(images)
            result["encode_max_abs"] = float((actual - expected).abs().max())
            assert torch.equal(actual, expected)
        expected = reference.decode(latents).sample
        actual = model.decode(latents)
        result["decode_max_abs"] = float((actual - expected).abs().max())
        assert torch.equal(actual, expected)
    record(name, **result)


def test_tiny_autoencoder(record):
    generator = torch.Generator().manual_seed(5)
    images = torch.randn(2, 3, 32, 48, generator=generator)
    latents = torch.randn(2, 8, 8, 12, generator=generator)
    _vae_case(TINY_VAE, images, latents, record, "sana_tiny_autoencoder")


def test_tiny_autoencoder_small_spatial_quadratic_path(record):
    generator = torch.Generator().manual_seed(6)
    images = torch.randn(1, 3, 16, 16, generator=generator)
    latents = torch.randn(1, 8, 2, 2, generator=generator)
    _vae_case(TINY_VAE, images, latents, record, "sana_tiny_autoencoder_quadratic")


def test_published_autoencoder_decode(record):
    config = _config("1600m/vae_config.json")
    latents = torch.randn(1, 32, 8, 8, generator=torch.Generator().manual_seed(7))
    _vae_case(config, None, latents, record, "sana_published_autoencoder_decode")
