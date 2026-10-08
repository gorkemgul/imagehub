import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_flux_kontext import make_image, write_tiny_flux_snapshot  # noqa: E402

from imagehub import pipeline  # noqa: E402
from imagehub.backends.flux import FluxComponents, denoise, kontext_output_size  # noqa: E402
from imagehub.backends.flux_kontext import (  # noqa: E402
    encode_kontext_image,
    kontext_input_size,
    preprocess_kontext_image,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")


def _expected(root, *, seed, image, negative_prompt, true_cfg_scale, width, height, max_area):
    reference = diffusers.FluxKontextPipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    common = {
        "image": image,
        "prompt": "a cat",
        "negative_prompt": negative_prompt,
        "true_cfg_scale": true_cfg_scale,
        "num_inference_steps": 2,
        "guidance_scale": 3.5,
        "height": height,
        "width": width,
        "max_area": max_area,
        "max_sequence_length": 8,
    }

    def generator():
        return torch.Generator("cpu").manual_seed(seed)

    latents = reference(**common, generator=generator(), output_type="latent").images
    pixels = reference(**common, generator=generator(), output_type="np").images
    return latents, torch.from_numpy((pixels * 255).round().astype("uint8"))


CASES = [
    # (id, input image size, width, height, max_area, negative prompt, true cfg, seed)
    ("no-image", None, 32, 32, 32 * 32, None, 1.0, 0),
    ("square-image", (64, 64), 32, 32, 32 * 32, None, 1.0, 0),
    ("non-square-image", (96, 64), 48, 32, 48 * 32, None, 1.0, 3),
    ("true-cfg", (64, 64), 32, 32, 32 * 32, "blurry", 2.0, 0),
    ("other-seed", (96, 64), 48, 32, 48 * 32, None, 1.0, 11),
    ("rescaled-request", (64, 96), 64, 64, 32 * 48, None, 1.0, 5),
]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return write_tiny_flux_snapshot(tmp_path_factory.mktemp("tiny_kontext") / "snapshot")


@pytest.mark.parametrize(
    ("size", "width", "height", "max_area", "negative_prompt", "true_cfg_scale", "seed"),
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_tiny_flux_kontext_matches_diffusers_exactly(
    root, size, width, height, max_area, negative_prompt, true_cfg_scale, seed
):
    image = make_image(*size, seed=seed) if size else None
    expected_latents, expected_pixels = _expected(
        root,
        seed=seed,
        image=image,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        width=width,
        height=height,
        max_area=max_area,
    )
    result = pipeline("t2i", backend="flux1-kontext-dev", model=root)(
        "a cat",
        image=image,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        steps=2,
        width=width,
        height=height,
        max_area=max_area,
        max_sequence_length=8,
        seed=seed,
    )
    assert torch.equal(torch.stack([i.tensor for i in result.images]), expected_pixels)

    components = FluxComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    out_height, out_width = kontext_output_size(height, width, max_area)
    noise = torch.randn(
        1,
        4,
        out_height // 8,
        out_width // 8,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    image_latents = None
    if image is not None:
        input_size = kontext_input_size(image)
        pixels = preprocess_kontext_image(image, *input_size)
        reference = diffusers.FluxKontextPipeline.from_pretrained(root, local_files_only=True)
        processor = reference.image_processor
        expected_pixels_in = processor.preprocess(processor.resize(image, *input_size), *input_size)
        assert torch.equal(pixels, expected_pixels_in)
        assert pixels.stride() == expected_pixels_in.stride()
        image_latents = encode_kontext_image(components.vae, pixels)
    actual_latents = denoise(
        components,
        prompt="a cat",
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        steps=2,
        guidance=3.5,
        width=out_width,
        height=out_height,
        latents=noise,
        max_sequence_length=8,
        image_latents=image_latents,
    )
    assert torch.equal(actual_latents, expected_latents)
