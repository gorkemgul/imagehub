"""FLUX.1 Kontext input preprocessing and VAE encoding parity on CPU float32."""

import pytest
import torch

from imagehub.backends.flux_kontext import (
    encode_kontext_image,
    kontext_input_size,
    preprocess_kontext_image,
)
from imagehub.models.autoencoder_kl import AutoencoderKL

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
PILImage = pytest.importorskip("PIL.Image")
from diffusers import FluxKontextPipeline  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402

PREFERRED = diffusers.pipelines.flux.pipeline_flux_kontext.PREFERRED_KONTEXT_RESOLUTIONS
SIZES = [(1024, 1024), (512, 768), (1365, 768), (300, 200), (2000, 1100)]
CASES = [(size, "RGB") for size in SIZES] + [((640, 480), "RGBA"), ((480, 640), "L")]
CHANNELS = {"RGB": 3, "RGBA": 4, "L": 1}


def make(size, mode, seed=0):
    width, height = size
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, 256, (height * width * CHANNELS[mode],), generator=g, dtype=torch.uint8)
    return PILImage.frombytes(mode, size, bytes(data.tolist()))


def reference_pipeline(vae):
    # The methods under test only need the VAE and the image processor.
    pipe = object.__new__(FluxKontextPipeline)
    pipe.vae_scale_factor = 8
    pipe.image_processor = VaeImageProcessor(vae_scale_factor=16)
    pipe.vae = vae
    return pipe


def reference_size(pipe, image, auto_resize):
    # Copy of the arithmetic in FluxKontextPipeline.__call__ (step 3).
    multiple_of = pipe.vae_scale_factor * 2
    image_height, image_width = pipe.image_processor.get_default_height_width(image)
    aspect_ratio = image_width / image_height
    if auto_resize:
        _, image_width, image_height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED)
    image_width = image_width // multiple_of * multiple_of
    image_height = image_height // multiple_of * multiple_of
    return image_height, image_width


def tiny_vae():
    return diffusers.AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(16, 32),
        layers_per_block=1,
        latent_channels=16,
        norm_num_groups=8,
        use_quant_conv=False,
        use_post_quant_conv=False,
        shift_factor=0.1159,
        scaling_factor=0.3611,
    ).eval()


@pytest.mark.parametrize("auto_resize", [True, False])
@pytest.mark.parametrize(("size", "mode"), CASES)
def test_size_and_preprocess_match_pipeline(size, mode, auto_resize, record):
    pipe = reference_pipeline(None)
    original = make(size, mode)
    # The pipeline's processor does not convert modes (do_convert_rgb=False), so the reference
    # receives the RGB conversion that imagehub applies up front.
    rgb = pipe.image_processor.convert_to_rgb(original)
    height, width = reference_size(pipe, rgb, auto_resize)
    assert kontext_input_size(original, auto_resize=auto_resize) == (height, width)

    resized = pipe.image_processor.resize(rgb, height, width)
    expected = pipe.image_processor.preprocess(resized, height, width)
    actual = preprocess_kontext_image(original, height, width)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    record(
        f"flux_kontext_preprocess_{size[0]}x{size[1]}_{mode}_{'auto' if auto_resize else 'raw'}",
        max_abs_diff=float((actual - expected).abs().max()),
        tolerance=0.0,
    )


def test_resized_bytes_match(record):
    pipe = reference_pipeline(None)
    for size, mode in CASES:
        rgb = pipe.image_processor.convert_to_rgb(make(size, mode))
        height, width = kontext_input_size(rgb)
        expected = pipe.image_processor.resize(rgb, height, width)
        actual = rgb.resize((width, height), resample=PILImage.Resampling.LANCZOS)
        assert actual.tobytes() == expected.tobytes()


def test_encode_matches_pipeline(record):
    torch.manual_seed(0)
    reference_vae = tiny_vae()
    pipe = reference_pipeline(reference_vae)
    ours = AutoencoderKL(dict(reference_vae.config)).eval()
    ours.load_state_dict(ours.prepare_state_dict(reference_vae.state_dict()), strict=True)
    for size, mode in CASES[:3] + CASES[5:]:
        original = make(size, mode)
        height, width = kontext_input_size(original)
        pixels = preprocess_kontext_image(original, height // 4, width // 4)
        with torch.no_grad():
            expected = pipe._encode_vae_image(pixels, generator=None)
            actual = encode_kontext_image(ours, pixels)
        assert torch.equal(actual, expected)
        record(
            f"flux_kontext_encode_{size[0]}x{size[1]}_{mode}",
            max_abs_diff=float((actual - expected).abs().max()),
            tolerance=0.0,
        )
