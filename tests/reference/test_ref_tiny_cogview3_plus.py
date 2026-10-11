"""Tiny-snapshot CogView3-Plus end-to-end parity against the diffusers pipeline."""

import pytest
import torch
from helpers import write_tiny_cogview3_plus

from imagehub import pipeline
from imagehub.backends.cogview3_plus import CogView3PlusComponents, decode_latents, denoise

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")


def _reference(root):
    reference = diffusers.CogView3PlusPipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    return reference


@pytest.mark.parametrize("use_seed", [False, True], ids=["latents", "seed"])
@pytest.mark.parametrize("negative_prompt", [None, "a hi"], ids=["zero-negative", "negative"])
@pytest.mark.parametrize(
    ("guidance", "height", "width"),
    [(5.0, 64, 64), (1.0, 64, 64), (4.0, 48, 80)],
    ids=["square", "no-guidance", "non-square"],
)
def test_tiny_cogview3_plus_matches_diffusers_exactly(
    tmp_path, use_seed, negative_prompt, guidance, height, width
):
    root = write_tiny_cogview3_plus(tmp_path / "tiny")
    reference = _reference(root)
    shape = (1, 4, height // 8, width // 8)
    noise = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(42))
    common = {
        "prompt": "a cat",
        "negative_prompt": negative_prompt,
        "num_inference_steps": 3,
        "guidance_scale": guidance,
        "height": height,
        "width": width,
        "max_sequence_length": 16,
    }

    def source():
        if use_seed:
            return {"generator": torch.Generator("cpu").manual_seed(42)}
        return {"latents": noise.clone()}

    expected_latents = reference(**common, **source(), output_type="latent").images
    expected_pixels = reference(**common, **source(), output_type="np").images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))

    components = CogView3PlusComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt="a cat",
        negative_prompt=negative_prompt,
        steps=3,
        guidance=guidance,
        width=width,
        height=height,
        latents=noise.clone(),
        max_sequence_length=16,
    )
    assert torch.equal(actual_latents, expected_latents)
    assert torch.equal(decode_latents(components.vae, actual_latents), expected_pixels)

    call = {
        "negative_prompt": negative_prompt,
        "steps": 3,
        "guidance": guidance,
        "width": width,
        "height": height,
        "max_sequence_length": 16,
    }
    call.update({"seed": 42} if use_seed else {"latents": noise})
    result = pipeline("t2i", backend="cogview3-plus-3b", model=root)("a cat", **call)
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected_pixels)


def test_tiny_cogview3_plus_size_conditions_match_diffusers(tmp_path):
    root = write_tiny_cogview3_plus(tmp_path / "tiny")
    reference = _reference(root)
    noise = torch.randn(1, 4, 8, 8, generator=torch.Generator("cpu").manual_seed(5))
    expected = reference(
        prompt="a cat",
        num_inference_steps=2,
        height=64,
        width=64,
        original_size=(128, 96),
        crops_coords_top_left=(16, 8),
        max_sequence_length=16,
        latents=noise.clone(),
        output_type="latent",
    ).images
    components = CogView3PlusComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    actual = denoise(
        components,
        prompt="a cat",
        negative_prompt=None,
        steps=2,
        guidance=5.0,
        width=64,
        height=64,
        latents=noise.clone(),
        max_sequence_length=16,
        original_size=(128, 96),
        crops_coords_top_left=(16, 8),
    )
    assert torch.equal(actual, expected)
