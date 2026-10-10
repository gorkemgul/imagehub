"""A tiny random-weight SDXL refiner against diffusers' image-to-image pipeline (CPU fp32)."""

import pytest
import torch
from helpers import write_tiny_sdxl

from imagehub import pipeline
from imagehub.backends.sdxl_refiner import preprocess_refiner_image
from imagehub.models.autoencoder_kl import AutoencoderKL

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")
PILImage = pytest.importorskip("PIL.Image")
from diffusers.image_processor import VaeImageProcessor  # noqa: E402

CHANNELS = {"RGB": 3, "RGBA": 4, "L": 1}


def make(size, mode="RGB", seed=0):
    width, height = size
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, 256, (height * width * CHANNELS[mode],), generator=g, dtype=torch.uint8)
    return PILImage.frombytes(mode, size, bytes(data.tolist()))


def reference(root):
    ref = diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained(
        root, local_files_only=True, add_watermarker=False
    )
    ref.set_progress_bar_config(disable=True)
    return ref


@pytest.fixture(scope="module")
def tiny_refiner(tmp_path_factory):
    return write_tiny_sdxl(tmp_path_factory.mktemp("tiny-refiner"), refiner=True)


@pytest.mark.parametrize(
    "case",
    [
        dict(size=(32, 32), steps=10, strength=0.3, guidance=5.0, num=1),
        dict(size=(61, 45), steps=8, strength=0.5, guidance=5.0, num=1, negative="blurry"),
        dict(size=(48, 40), steps=6, strength=1.0, guidance=1.0, num=1),
        dict(size=(40, 32), steps=7, strength=0.7, guidance=4.0, num=2, mode="RGBA"),
        dict(
            size=(32, 40),
            steps=10,
            strength=0.4,
            guidance=5.0,
            num=1,
            original_size=(64, 80),
            crops_coords_top_left=(3, 4),
            target_size=(48, 56),
            aesthetic_score=7.5,
            negative_aesthetic_score=1.0,
        ),
    ],
    ids=[
        "defaults",
        "odd-size-negative",
        "full-strength-no-cfg",
        "batch-rgba",
        "micro-conditioning",
    ],
)
def test_tiny_refiner_matches_diffusers(tiny_refiner, case):
    image = make(case["size"], case.get("mode", "RGB"))
    extra = {
        key: case[key]
        for key in (
            "original_size",
            "crops_coords_top_left",
            "target_size",
            "aesthetic_score",
            "negative_aesthetic_score",
        )
        if key in case
    }
    expected = reference(tiny_refiner)(
        "a cat",
        image=image.convert("RGB"),
        strength=case["strength"],
        negative_prompt=case.get("negative"),
        num_inference_steps=case["steps"],
        guidance_scale=case["guidance"],
        num_images_per_prompt=case["num"],
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
        **extra,
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    result = pipeline("t2i", backend="sdxl-refiner-1.0", model=tiny_refiner)(
        "a cat",
        image=image,
        strength=case["strength"],
        negative_prompt=case.get("negative"),
        steps=case["steps"],
        guidance=case["guidance"],
        num_images=case["num"],
        seed=42,
        **extra,
    )
    actual = torch.stack([item.tensor for item in result.images])
    assert torch.equal(actual, expected)


def test_tiny_refiner_defaults_match_the_pipeline_signature(tiny_refiner):
    image = make((32, 32))
    expected = reference(tiny_refiner)(
        "a cat",
        image=image,
        generator=torch.Generator("cpu").manual_seed(0),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    result = pipeline("t2i", backend="sdxl-refiner-1.0", model=tiny_refiner)(
        "a cat", image=image, seed=0
    )
    assert torch.equal(torch.stack([item.tensor for item in result.images]), expected)
    assert result.metadata["steps"] == 50 and result.metadata["strength"] == 0.3
    assert result.metadata["guidance"] == 5.0


@pytest.mark.parametrize(
    ("size", "mode"),
    [
        ((64, 64), "RGB"),
        ((61, 45), "RGB"),
        ((1000, 333), "RGB"),
        ((80, 50), "RGBA"),
        ((50, 80), "L"),
    ],
)
def test_preprocessing_matches_vae_image_processor(size, mode):
    image = make(size, mode)
    expected = VaeImageProcessor(vae_scale_factor=8).preprocess(image.convert("RGB"))
    actual = preprocess_refiner_image(image)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert actual.stride() == expected.stride()


def test_vae_sample_matches_diffusers_and_consumes_the_generator_alike(tiny_refiner):
    ours = AutoencoderKL.from_pretrained(
        tiny_refiner / "vae", device=torch.device("cpu"), dtype=torch.float32
    )
    ref = reference(tiny_refiner).vae
    pixels = preprocess_refiner_image(make((40, 32)))
    ref_generator = torch.Generator("cpu").manual_seed(5)
    our_generator = torch.Generator("cpu").manual_seed(5)
    with torch.no_grad():
        distribution = ref.encode(pixels).latent_dist
        expected = distribution.sample(ref_generator)
        mean, logvar = ours.encode_moments(pixels)
        actual = ours.encode_sample(pixels, our_generator)
    assert torch.equal(mean, distribution.mean) and torch.equal(logvar, distribution.logvar)
    assert torch.equal(actual, expected)
    assert torch.equal(ref_generator.get_state(), our_generator.get_state())
    assert torch.equal(ours.encode(pixels), distribution.mode())
