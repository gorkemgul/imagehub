import pytest
import torch

from imagehub.backends.flux_kontext import (
    PREFERRED_KONTEXT_RESOLUTIONS,
    encode_kontext_image,
    kontext_input_size,
    preprocess_kontext_image,
)
from imagehub.image import Image, as_pil
from imagehub.models.autoencoder_kl import AutoencoderKL

PILImage = pytest.importorskip("PIL.Image")


def make(width, height, mode="RGB"):
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, 256, (height * width * channels,), generator=g, dtype=torch.uint8)
    return PILImage.frombytes(mode, (width, height), bytes(data.tolist()))


def test_as_pil_modes_and_types():
    assert as_pil(make(5, 4)).mode == "RGB"
    assert as_pil(make(5, 4, "RGBA")).mode == "RGB"
    assert as_pil(make(5, 4, "L")).mode == "RGB"
    tensor = torch.randint(0, 256, (4, 5, 4), dtype=torch.uint8)
    converted = as_pil(Image(tensor))
    assert converted.mode == "RGB" and converted.size == (5, 4)
    for bad in (tensor, "x.png", None):
        with pytest.raises(TypeError):
            as_pil(bad)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((1024, 1024), (1024, 1024)),
        ((512, 768), (1248, 832)),
        ((1365, 768), (752, 1392)),
        ((300, 200), (832, 1248)),
        ((2000, 1100), (752, 1392)),
    ],
)
def test_size_selects_preferred_resolution(size, expected):
    assert kontext_input_size(make(*size)) == expected
    height, width = expected
    assert (width, height) in PREFERRED_KONTEXT_RESOLUTIONS


def test_size_without_auto_resize_rounds_down():
    assert kontext_input_size(make(300, 203), auto_resize=False) == (192, 288)
    assert kontext_input_size(make(1365, 768), auto_resize=False) == (768, 1360)


def test_preprocess_shape_and_range():
    pixels = preprocess_kontext_image(make(40, 30, "RGBA"), 32, 48)
    assert pixels.shape == (1, 3, 32, 48) and pixels.dtype == torch.float32
    assert pixels.min() >= -1 and pixels.max() <= 1


def test_encode_applies_shift_and_scale():
    config = {
        "block_out_channels": [8],
        "layers_per_block": 1,
        "latent_channels": 4,
        "norm_num_groups": 4,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "shift_factor": 0.25,
        "scaling_factor": 0.5,
    }
    vae = AutoencoderKL(config).eval()
    pixels = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        expected = (vae.encode(pixels) - 0.25) * 0.5
        assert torch.equal(encode_kontext_image(vae, pixels), expected)
