import pytest
import torch
from helpers import write_tiny_lcm

from imagehub import pipeline
from imagehub.backends.stable_diffusion import StableDiffusionComponents, denoise, initial_latents

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")


def test_tiny_lcm_snapshot_matches_diffusers(tmp_path):
    root = write_tiny_lcm(tmp_path / "tiny-lcm")
    ref = diffusers.LatentConsistencyModelPipeline.from_pretrained(
        root,
        safety_checker=None,
        local_files_only=True,
    )
    ref.set_progress_bar_config(disable=True)
    seed = 42
    generator = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn((1, 4, 4, 4), generator=generator)
    expected_latents = ref(
        "a cat",
        num_inference_steps=4,
        guidance_scale=8.0,
        latents=latents.clone(),
        generator=generator,
        output_type="latent",
    ).images

    components = StableDiffusionComponents.load(
        root, device=torch.device("cpu"), dtype=torch.float32
    )
    actual_generator = torch.Generator("cpu").manual_seed(seed)
    actual_initial = torch.randn((1, 4, 4, 4), generator=actual_generator)
    actual_latents = denoise(
        components,
        prompt="a cat",
        negative_prompt=None,
        steps=4,
        guidance=8.0,
        latents=actual_initial,
        generator=actual_generator,
    )
    torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)

    expected_pixels = ref(
        "a cat",
        num_inference_steps=4,
        guidance_scale=8.0,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="np",
    ).images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))
    result = pipeline("t2i", backend="stable-diffusion", model=root)(
        "a cat", seed=seed, steps=4, guidance=8.0
    )
    actual_pixels = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual_pixels, expected_pixels)
    assert result.metadata["scheduler"] == "lcm"
    assert result.metadata["width"] == result.metadata["height"] == 32
    assert torch.equal(initial_latents(seed, (1, 4, 4, 4)), latents)
