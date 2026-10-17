import pytest
import torch
from helpers import write_tiny_qwen_image

from imagehub import pipeline
from imagehub.backends.qwen_image import (
    OriginalQwenImageComponents,
    decode_original_latents,
    denoise_original,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")


@pytest.mark.parametrize(
    "seed,true_cfg,negative_prompt,height,width",
    [(0, 1.0, None, 32, 32), (42, 4.0, "blurry", 32, 48)],
    ids=["unguided-square", "guided-non-square"],
)
def test_tiny_qwen_image_generation_matches_diffusers_exactly(
    tmp_path, seed, true_cfg, negative_prompt, height, width
):
    root = write_tiny_qwen_image(tmp_path / "tiny", seed=7)
    reference = diffusers.QwenImagePipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    generator = torch.Generator("cpu").manual_seed(seed)
    raw = torch.randn(1, 1, 16, height // 8, width // 8, generator=generator)
    noise = diffusers.QwenImagePipeline._pack_latents(raw.clone(), 1, 16, height // 8, width // 8)
    expected_latents = reference(
        "a cat",
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg,
        num_inference_steps=2,
        height=height,
        width=width,
        latents=noise.clone(),
        output_type="latent",
    ).images
    expected_pixels = reference(
        "a cat",
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg,
        num_inference_steps=2,
        height=height,
        width=width,
        latents=noise.clone(),
        output_type="np",
    ).images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))

    components = OriginalQwenImageComponents.load(
        root, device=torch.device("cpu"), dtype=torch.float32
    )
    actual_latents = denoise_original(
        components,
        prompt="a cat",
        negative_prompt=negative_prompt,
        steps=2,
        true_cfg=true_cfg,
        width=width,
        height=height,
        latents=noise.clone(),
    )
    torch.testing.assert_close(actual_latents, expected_latents, atol=0, rtol=0)
    actual_pixels = decode_original_latents(components.vae, actual_latents, height, width)
    assert torch.equal(actual_pixels, expected_pixels)
    result = pipeline("t2i", backend="qwen-image", model=root)(
        "a cat",
        negative_prompt=negative_prompt,
        true_cfg=true_cfg,
        steps=2,
        width=width,
        height=height,
        latents=noise,
    )
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected_pixels)
