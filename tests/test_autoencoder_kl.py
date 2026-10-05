import json
from pathlib import Path

import pytest
import torch
from helpers import TINY_VAE_CONFIG, meta_shapes, published_shapes, sdxl_published_shapes

from imagehub.models.autoencoder_kl import AutoencoderKL

FIXTURES = Path(__file__).parent / "fixtures"


def test_matches_published_tensor_layout():
    assert meta_shapes(AutoencoderKL, {}) == published_shapes("vae")


def test_matches_sdxl_published_tensor_layout_after_name_conversion():
    expected = {}
    for name, shape in sdxl_published_shapes("vae").items():
        for new, legacy in (
            ("to_q", "query"),
            ("to_k", "key"),
            ("to_v", "value"),
            ("to_out.0", "proj_attn"),
        ):
            name = name.replace(f".{new}.", f".{legacy}.")
        expected[name] = shape
    assert meta_shapes(AutoencoderKL, {}) == expected


@pytest.mark.parametrize(
    ("config_path", "family"),
    [
        ("playground_v25_configs/vae.json", "sdxl"),
        ("sdxl_configs/vae.json", "sdxl"),
        ("sd1_configs/v1-5/vae.json", "sd1"),
        ("sd1_configs/v1-1/vae.json", "sd1"),
    ],
)
def test_real_release_configs_match_published_tensor_layout(config_path, family):
    config = json.loads((FIXTURES / config_path).read_text(encoding="utf-8"))
    if family == "sdxl":
        expected = {}
        for name, shape in sdxl_published_shapes("vae").items():
            for new, legacy in (
                ("to_q", "query"),
                ("to_k", "key"),
                ("to_v", "value"),
                ("to_out.0", "proj_attn"),
            ):
                name = name.replace(f".{new}.", f".{legacy}.")
            expected[name] = shape
    else:
        expected = published_shapes("vae")
    assert meta_shapes(AutoencoderKL, config) == expected


def test_loads_legacy_and_new_attention_names_strictly():
    legacy = AutoencoderKL(TINY_VAE_CONFIG).state_dict()
    AutoencoderKL(TINY_VAE_CONFIG).load_checkpoint(legacy)
    current = {}
    for name, value in legacy.items():
        if ".attentions." in name:
            for old, new in (
                ("query", "to_q"),
                ("key", "to_k"),
                ("value", "to_v"),
                ("proj_attn", "to_out.0"),
            ):
                name = name.replace(f".{old}.", f".{new}.")
        current[name] = value
    AutoencoderKL(TINY_VAE_CONFIG).load_checkpoint(current)


def test_shapes_and_scaling_factor():
    torch.manual_seed(0)
    vae = AutoencoderKL(TINY_VAE_CONFIG).eval()
    assert vae.scaling_factor == 0.18215
    assert AutoencoderKL(TINY_VAE_CONFIG | {"scaling_factor": 0.5}).scaling_factor == 0.5
    with torch.inference_mode():
        assert vae.decode(torch.randn(1, 4, 8, 6)).shape == (1, 3, 64, 48)
        assert vae.encode(torch.rand(1, 3, 64, 48) * 2 - 1).shape == (1, 4, 8, 6)


def test_accepts_latent_statistics_config():
    config = TINY_VAE_CONFIG | {
        "latents_mean": [-1.0, 1.0, -2.0, 2.0],
        "latents_std": [2.0, 3.0, 4.0, 5.0],
        "scaling_factor": 0.5,
    }
    vae = AutoencoderKL(config)
    assert vae.latents_mean == config["latents_mean"]
    assert vae.latents_std == config["latents_std"]


def test_flux_style_options():
    config = TINY_VAE_CONFIG | {
        "latent_channels": 16,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "shift_factor": 0.1159,
        "mid_block_add_attention": True,
    }
    vae = AutoencoderKL(config).eval()
    assert vae.shift_factor == 0.1159
    assert not hasattr(vae, "quant_conv")
    assert not hasattr(vae, "post_quant_conv")
    with torch.inference_mode():
        assert vae.decode(torch.randn(1, 16, 8, 6)).shape == (1, 3, 64, 48)
        assert vae.encode(torch.rand(1, 3, 64, 48) * 2 - 1).shape == (1, 16, 8, 6)
