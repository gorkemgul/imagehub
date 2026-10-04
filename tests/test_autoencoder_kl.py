import torch
from helpers import TINY_VAE_CONFIG, meta_shapes, published_shapes, sdxl_published_shapes

from imagehub.models.autoencoder_kl import AutoencoderKL


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
