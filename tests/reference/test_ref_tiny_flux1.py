import pytest
import torch
from helpers import write_tiny_flux1

from imagehub import pipeline
from imagehub.backends.flux import FluxComponents, _pack_latents, denoise

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")


# The dev-style snapshot (guidance embedding) uses dynamic shifting, the schnell-style one not.
@pytest.mark.parametrize(
    ("guidance_embeds", "use_seed", "negative_prompt", "true_cfg_scale"),
    [
        (False, False, None, 1.0),
        (False, True, None, 1.0),
        (True, False, None, 1.0),
        (True, True, None, 1.0),
        (True, False, "blurry", 2.0),
    ],
    ids=["schnell-latents", "schnell-seed", "dev-latents", "dev-seed", "dev-true-cfg"],
)
def test_tiny_flux1_matches_diffusers_exactly(
    tmp_path, guidance_embeds, use_seed, negative_prompt, true_cfg_scale
):
    root = write_tiny_flux1(tmp_path / "tiny", guidance_embeds=guidance_embeds)
    reference = diffusers.FluxPipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    noise = torch.randn(1, 4, 4, 4, generator=torch.Generator("cpu").manual_seed(42))
    common = {
        "prompt": "a cat",
        "negative_prompt": negative_prompt,
        "true_cfg_scale": true_cfg_scale,
        "num_inference_steps": 2,
        "guidance_scale": 3.5 if guidance_embeds else 0.0,
        "height": 32,
        "width": 32,
        "max_sequence_length": 8,
    }
    source = (
        {"generator": torch.Generator("cpu").manual_seed(42)}
        if use_seed
        else {"latents": _pack_latents(noise.clone(), 1, 4, 4, 4)}
    )
    expected_latents = reference(**common, **source, output_type="latent").images
    if use_seed:
        source = {"generator": torch.Generator("cpu").manual_seed(42)}
    expected_pixels = reference(**common, **source, output_type="np").images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))

    components = FluxComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    actual_noise = (
        torch.randn(1, 4, 4, 4, generator=torch.Generator("cpu").manual_seed(42))
        if use_seed
        else noise.clone()
    )
    actual_latents = denoise(
        components,
        prompt="a cat",
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        steps=2,
        guidance=3.5 if guidance_embeds else 0.0,
        width=32,
        height=32,
        latents=actual_noise,
        max_sequence_length=8,
    )
    torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)
    call = {
        "negative_prompt": negative_prompt,
        "true_cfg_scale": true_cfg_scale,
        "steps": 2,
        "guidance": 3.5 if guidance_embeds else 0.0,
        "width": 32,
        "height": 32,
        "max_sequence_length": 8,
    }
    if use_seed:
        call["seed"] = 42
    else:
        call["latents"] = noise
    result = pipeline("t2i", backend="flux", model=root)("a cat", **call)
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected_pixels)
