import pytest
import torch
from helpers import write_tiny_z_image

from imagehub import pipeline
from imagehub.backends.z_image import ZImageComponents, denoise

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")


@pytest.mark.parametrize(
    "guidance,negative_prompt", [(0.0, None), (4.0, "blurry")], ids=["unguided", "guided"]
)
def test_tiny_z_image_generation_matches_diffusers_exactly(tmp_path, guidance, negative_prompt):
    root = write_tiny_z_image(tmp_path / "tiny")
    reference = diffusers.ZImagePipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    noise = torch.randn(1, 16, 4, 4, generator=torch.Generator("cpu").manual_seed(42))
    expected = reference(
        "a cat",
        negative_prompt=negative_prompt,
        num_inference_steps=2,
        guidance_scale=guidance,
        height=32,
        width=32,
        latents=noise.clone(),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    components = ZImageComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt="a cat",
        negative_prompt=negative_prompt,
        steps=2,
        guidance=guidance,
        cfg_normalization=False,
        cfg_truncation=1.0,
        latents=noise.clone(),
        max_sequence_length=512,
    )
    expected_latents = reference(
        "a cat",
        negative_prompt=negative_prompt,
        num_inference_steps=2,
        guidance_scale=guidance,
        height=32,
        width=32,
        latents=noise.clone(),
        output_type="latent",
    ).images
    torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)
    result = pipeline("t2i", backend="z-image", model=root)(
        "a cat",
        negative_prompt=negative_prompt,
        steps=2,
        guidance=guidance,
        width=32,
        height=32,
        latents=noise,
    )
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected)
