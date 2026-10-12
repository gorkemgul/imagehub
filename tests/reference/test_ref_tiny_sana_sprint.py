"""Tiny-snapshot Sana-Sprint end-to-end parity against diffusers' SanaSprintPipeline."""

import pytest
import torch
from helpers import write_tiny_sana_sprint

from imagehub import pipeline
from imagehub.backends.sana import (
    COMPLEX_HUMAN_INSTRUCTION,
    SanaComponents,
    denoise,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
np = pytest.importorskip("numpy")

VAE_SCALE = 32
PROMPT = "A photo of a cat, sitting on the mat in the sun."


@pytest.fixture(scope="module")
def tiny_sprint(tmp_path_factory):
    root = write_tiny_sana_sprint(tmp_path_factory.mktemp("sana-sprint") / "tiny")
    reference = diffusers.SanaSprintPipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    components = SanaComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    return root, reference, components


def _reference_call(reference, seed, output_type, **kwargs):
    return reference(
        PROMPT,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type=output_type,
        clean_caption=False,
        **kwargs,
    ).images


def test_sprint_denoise_latents_match_reference(tiny_sprint):
    root, reference, components = tiny_sprint
    seed = 7
    kwargs = {
        "num_inference_steps": 2,
        "guidance_scale": 4.5,
        "height": 512,
        "width": 512,
        "use_resolution_binning": False,
        "max_sequence_length": 32,
    }
    expected_latents = _reference_call(reference, seed, "latent", **kwargs)

    generator = torch.Generator("cpu").manual_seed(seed)
    noise = torch.randn(1, 8, 512 // VAE_SCALE, 512 // VAE_SCALE, generator=generator)
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt="",
        steps=2,
        guidance=4.5,
        max_sequence_length=32,
        instruction=COMPLEX_HUMAN_INSTRUCTION,
        latents=noise,
        generator=generator,
    )
    assert torch.equal(actual_latents, expected_latents)


def test_sprint_pipeline_outputs_match_reference(tiny_sprint, record):
    root, reference, _ = tiny_sprint
    seed = 7
    kwargs = {
        "num_inference_steps": 2,
        "guidance_scale": 4.5,
        "height": 512,
        "width": 512,
        "use_resolution_binning": False,
        "max_sequence_length": 32,
    }
    expected_images = _reference_call(reference, seed, "pil", **kwargs)
    expected_pixels = torch.from_numpy(np.stack([np.array(image) for image in expected_images]))

    result = pipeline("t2i", backend="sana", model=root)(
        PROMPT,
        seed=seed,
        steps=2,
        guidance=4.5,
        width=512,
        height=512,
        use_resolution_binning=False,
        max_sequence_length=32,
    )
    pixels = torch.stack([image.tensor for image in result.images])
    assert torch.equal(pixels, expected_pixels)
    record(
        "sana_sprint_tiny_default",
        pixels_equal=True,
        shape=list(pixels.shape),
    )
