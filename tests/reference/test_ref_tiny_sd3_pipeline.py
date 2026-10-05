import pytest
import torch
from helpers import write_tiny_sd3

from imagehub import pipeline
from imagehub.backends.stable_diffusion_3 import StableDiffusion3Components, denoise

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")


@pytest.mark.parametrize(
    "guidance,negative_prompt",
    [(0.0, None), (1.0, None), (4.0, "blurry")],
    ids=["guidance-0", "guidance-1", "guided"],
)
def test_tiny_sd3_generation_matches_diffusers_exactly(tmp_path, guidance, negative_prompt):
    root = write_tiny_sd3(tmp_path / "tiny")
    reference = diffusers.StableDiffusion3Pipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    noise = torch.randn(1, 4, 4, 4, generator=torch.Generator("cpu").manual_seed(42))
    expected_latents = reference(
        "a cat",
        negative_prompt=negative_prompt,
        num_inference_steps=2,
        guidance_scale=guidance,
        height=32,
        width=32,
        max_sequence_length=8,
        latents=noise.clone(),
        output_type="latent",
    ).images
    expected_pixels = reference(
        "a cat",
        negative_prompt=negative_prompt,
        num_inference_steps=2,
        guidance_scale=guidance,
        height=32,
        width=32,
        max_sequence_length=8,
        latents=noise.clone(),
        output_type="np",
    ).images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))
    components = StableDiffusion3Components.load(
        root, device=torch.device("cpu"), dtype=torch.float32
    )
    actual_latents = denoise(
        components,
        prompt="a cat",
        negative_prompt=negative_prompt,
        steps=2,
        guidance=guidance,
        latents=noise.clone(),
        max_sequence_length=8,
    )
    torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)
    result = pipeline("t2i", backend="stable-diffusion-3", model=root)(
        "a cat",
        negative_prompt=negative_prompt,
        steps=2,
        guidance=guidance,
        width=32,
        height=32,
        max_sequence_length=8,
        latents=noise,
    )
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected_pixels)
