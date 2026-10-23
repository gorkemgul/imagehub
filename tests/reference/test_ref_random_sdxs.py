"""Exact SDXS component and tiny-pipeline parity against diffusers."""

import json
from pathlib import Path

import pytest
import torch
from helpers import write_tiny_sd

from imagehub.backends.stable_diffusion import (
    StableDiffusionComponents,
    decode_latents,
    denoise,
    initial_latents,
)
from imagehub.models.unet_2d import UNet2DConditionModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

PUBLISHED_UNET = Path(__file__).parents[1] / "fixtures/sdxs_configs/unet_config.json"
PROMPT = "a photograph of an astronaut riding a horse"


def _tiny_unet_config() -> dict:
    config = json.loads(PUBLISHED_UNET.read_text())
    config |= {
        "block_out_channels": [32, 64, 64],
        "attention_head_dim": [2, 4, 4],
        "cross_attention_dim": 32,
        "norm_num_groups": 8,
        "sample_size": 4,
    }
    return config


@pytest.fixture(scope="module")
def tiny_sdxs(tmp_path_factory):
    return write_tiny_sd(
        tmp_path_factory.mktemp("tiny-ref-sdxs"),
        tiny_vae=True,
        unet_config=_tiny_unet_config(),
        scheduler_config={"_class_name": "EulerDiscreteScheduler", "timestep_spacing": "trailing"},
    )


def test_sdxs_unet_random_weights_exact():
    config = _tiny_unet_config()
    torch.manual_seed(0)
    reference = diffusers.UNet2DConditionModel.from_config(config).eval()
    ours = UNet2DConditionModel(config).eval()
    reference.load_state_dict(ours.state_dict(), strict=True)
    sample = torch.randn(1, 4, 8, 12)
    context = torch.randn(1, 7, 32)
    with torch.inference_mode():
        expected = reference(sample, torch.tensor(999.0), context).sample
        actual = ours(sample, torch.tensor(999.0), context)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("seed", "steps", "guidance", "size"),
    [(0, 1, 0.0, (32, 32)), (42, 2, 2.0, (48, 32))],
)
def test_sdxs_tiny_pipeline_exact(tiny_sdxs, seed, steps, guidance, size):
    ours = StableDiffusionComponents.load(
        tiny_sdxs, device=torch.device("cpu"), dtype=torch.float32
    )
    reference = diffusers.StableDiffusionPipeline.from_pretrained(
        tiny_sdxs,
        local_files_only=True,
        torch_dtype=torch.float32,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    reference.set_progress_bar_config(disable=True)
    width, height = size
    latents = initial_latents(seed, (1, 4, height // 8, width // 8))
    with torch.inference_mode():
        expected_latents = reference(
            PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            latents=latents.clone(),
            output_type="latent",
        ).images
        expected_image = reference.vae.decode(
            expected_latents / reference.vae.config.scaling_factor
        ).sample
    expected_pixels = (
        ((expected_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1) * 255).round().to(torch.uint8)
    )
    actual_latents = denoise(
        ours,
        prompt=PROMPT,
        negative_prompt=None,
        steps=steps,
        guidance=guidance,
        latents=latents.clone(),
    )
    assert torch.equal(actual_latents, expected_latents)
    assert torch.equal(decode_latents(ours, actual_latents), expected_pixels)
