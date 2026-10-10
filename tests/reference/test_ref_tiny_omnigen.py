"""Tiny-snapshot OmniGen end-to-end parity against diffusers' OmniGenPipeline."""

import pytest
import torch
from helpers import write_tiny_omnigen

from imagehub import pipeline
from imagehub.backends.omnigen import OmniGenComponents, decode_latents, denoise
from imagehub.backends.omnigen_inputs import (
    build_inputs,
    encode_input_images,
    preprocess_input_image,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
np = pytest.importorskip("numpy")
pytest.importorskip("torchvision")
PIL = pytest.importorskip("PIL.Image")

PROMPT = "A photo of a cat, sitting on the mat in the sun."
EDIT_PROMPT = "Turn <img><|image_1|></img> into a snowy scene with a cat."


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = write_tiny_omnigen(tmp_path_factory.mktemp("omnigen") / "tiny")
    reference = diffusers.OmniGenPipeline.from_pretrained(
        root, torch_dtype=torch.float32, local_files_only=True
    )
    reference.set_progress_bar_config(disable=True)
    components = OmniGenComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    return root, reference, components


def _picture(size, mode="RGB", seed=5):
    generator = torch.Generator().manual_seed(seed)
    data = torch.randint(0, 256, (size[1], size[0], 3), generator=generator, dtype=torch.uint8)
    return PIL.fromarray(data.numpy()).convert(mode)


def _reference_call(reference, seed, output_type, prompt, image, kwargs):
    torch.manual_seed(seed)
    return reference(
        prompt,
        input_images=None if image is None else [image],
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type=output_type,
        **kwargs,
    ).images


def _check(tiny, record, name, *, image=None, seed=7, steps=3, **options):
    root, reference, components = tiny
    prompt = EDIT_PROMPT if image is not None else PROMPT
    height = options.get("height", 96)
    width = options.get("width", 128)
    kwargs = {
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": options.get("guidance", 2.5),
        "img_guidance_scale": options.get("image_guidance", 1.6),
        "max_input_image_size": options.get("max_input_image_size", 1024),
        "use_input_image_size_as_output": options.get("use_input_image_size_as_output", False),
    }
    # The reference fails on non-RGB images; parity is defined on the RGB conversion.
    rgb = None if image is None else image.convert("RGB")
    expected_latents = _reference_call(reference, seed, "latent", prompt, rgb, kwargs)
    expected_images = _reference_call(reference, seed, "pil", prompt, rgb, kwargs)

    pixels = None
    if image is not None:
        pixels = preprocess_input_image(image, kwargs["max_input_image_size"])
        assert torch.equal(pixels, preprocess_input_image(rgb, kwargs["max_input_image_size"]))
    processed = build_inputs(
        components.tokenizer,
        prompt,
        None if pixels is None else [pixels],
        height=height,
        width=width,
        use_img_cfg=image is not None,
        use_input_image_size_as_output=kwargs["use_input_image_size_as_output"],
    )
    image_latents = encode_input_images(
        components.vae,
        processed.input_pixel_values,
        torch.Generator("cpu").manual_seed(seed),
    )
    noise = torch.randn(
        1,
        4,
        processed.height // 8,
        processed.width // 8,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    actual_latents = denoise(
        components,
        processed,
        image_latents,
        steps=steps,
        guidance=kwargs["guidance_scale"],
        image_guidance=kwargs["img_guidance_scale"],
        latents=noise,
    )
    assert torch.equal(actual_latents, expected_latents)
    expected_pixels = torch.from_numpy(np.stack([np.array(i) for i in expected_images]))
    assert torch.equal(decode_latents(components.vae, actual_latents), expected_pixels)
    assert expected_pixels.float().std() > 1.0

    call = {"steps": steps, "height": height, "width": width} | options
    if image is not None:
        call["image"] = image
    result = pipeline("t2i", backend="omnigen", model=root)(prompt, seed=seed, **call)
    assert torch.equal(torch.stack([i.tensor for i in result.images]), expected_pixels)
    record(
        name,
        latent_max_abs=float((actual_latents - expected_latents).abs().max()),
        pixels_equal=True,
        shape=list(expected_pixels.shape),
    )


def test_defaults(tiny, record):
    _check(tiny, record, "omnigen_tiny_default")


def test_guidance_one(tiny, record):
    _check(tiny, record, "omnigen_tiny_guidance_1", guidance=1.0)


def test_guidance_high(tiny, record):
    _check(tiny, record, "omnigen_tiny_guidance_6", guidance=6.0, steps=4)


def test_non_square(tiny, record):
    _check(tiny, record, "omnigen_tiny_non_square", width=192, height=96)
    _check(tiny, record, "omnigen_tiny_portrait", width=80, height=160, seed=3)


@pytest.mark.parametrize("seed", [0, 123456789])
def test_seeds(tiny, record, seed):
    _check(tiny, record, f"omnigen_tiny_seed_{seed}", seed=seed, steps=2)


def test_one_step(tiny, record):
    _check(tiny, record, "omnigen_tiny_one_step", steps=1)


def test_image_input_defaults(tiny, record):
    _check(tiny, record, "omnigen_tiny_image", image=_picture((70, 50)))


def test_image_input_guidance_scales(tiny, record):
    image = _picture((64, 64))
    _check(tiny, record, "omnigen_tiny_image_scales", image=image, guidance=3.0, image_guidance=1.0)
    _check(tiny, record, "omnigen_tiny_image_scales_one", image=image, guidance=1.0, image_guidance=1.0)


def test_image_input_size_as_output(tiny, record):
    _check(
        tiny,
        record,
        "omnigen_tiny_image_size_as_output",
        image=_picture((90, 60)),
        use_input_image_size_as_output=True,
    )


def test_image_input_limit_and_modes(tiny, record):
    _check(
        tiny, record, "omnigen_tiny_image_limit", image=_picture((300, 200)), max_input_image_size=64
    )
    _check(tiny, record, "omnigen_tiny_image_rgba", image=_picture((50, 40), "RGBA"), seed=2)
    _check(tiny, record, "omnigen_tiny_image_gray", image=_picture((50, 80), "L"), seed=4)
