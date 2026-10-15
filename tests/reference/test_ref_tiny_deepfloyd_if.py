"""Tiny DeepFloyd IF pipeline parity; caption cleaning is deliberately not applied."""

import pytest
import torch
from test_ref_random_deepfloyd_if import _random_pair, _tiny

from imagehub.backends.deepfloyd_if import _denoise
from imagehub.schedulers.ddpm import DDPMScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def tiny():
    stage_i_config = _tiny("stage_i") | {"sample_size": 8}
    stage_ii_config = _tiny("stage_ii") | {"sample_size": 32}
    stage_i, native_i = _random_pair(stage_i_config)
    stage_ii, native_ii = _random_pair(stage_ii_config)
    scheduler_config = {
        "num_train_timesteps": 10,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "variance_type": "learned_range",
        "thresholding": True,
    }
    noising_config = scheduler_config | {"variance_type": "fixed_small", "thresholding": False}
    first = diffusers.IFPipeline(
        None, None, stage_i, diffusers.DDPMScheduler(**scheduler_config), None, None, None, False
    )
    second = diffusers.IFSuperResolutionPipeline(
        None,
        None,
        stage_ii,
        diffusers.DDPMScheduler(**scheduler_config),
        diffusers.DDPMScheduler(**noising_config),
        None,
        None,
        None,
        False,
    )
    first.set_progress_bar_config(disable=True)
    second.set_progress_bar_config(disable=True)
    return first, second, native_i, native_ii, scheduler_config, noising_config


@pytest.mark.parametrize("guidance", (1.0, 3.0))
@pytest.mark.parametrize("seed", (7, 42))
def test_tiny_if_and_superresolution_match_reference(tiny, guidance, seed, record):
    first, second, native_i, native_ii, scheduler_config, noising_config = tiny
    generator = torch.Generator().manual_seed(seed)
    context = torch.randn(1, 7, 48, generator=generator)
    negative = torch.randn(1, 7, 48, generator=generator)
    expected_i = first(
        prompt_embeds=context,
        negative_prompt_embeds=negative,
        num_inference_steps=2,
        guidance_scale=guidance,
        output_type="pt",
        generator=torch.Generator().manual_seed(seed),
    ).images
    actual_generator = torch.Generator().manual_seed(seed)
    actual_i = _denoise(
        native_i,
        scheduler_config,
        torch.cat([negative, context]) if guidance > 1 else context,
        torch.randn(1, 3, 8, 8, generator=actual_generator),
        steps=2,
        guidance=guidance,
        generator=actual_generator,
    )
    assert torch.equal(actual_i, expected_i)
    expected_ii = second(
        image=expected_i,
        prompt_embeds=context,
        negative_prompt_embeds=negative,
        num_inference_steps=2,
        guidance_scale=guidance,
        noise_level=5,
        output_type="pt",
        generator=torch.Generator().manual_seed(seed + 1),
    ).images
    actual_generator = torch.Generator().manual_seed(seed + 1)
    noiser = DDPMScheduler(noising_config)
    initial_ii = torch.randn(1, 3, 32, 32, generator=actual_generator)
    upscaled = torch.nn.functional.interpolate(
        actual_i, size=(32, 32), mode="bilinear", align_corners=True
    )
    upscaled = noiser.add_noise(
        upscaled, torch.randn(upscaled.shape, generator=actual_generator), torch.tensor([5])
    )
    actual_ii = _denoise(
        native_ii,
        scheduler_config,
        torch.cat([negative, context]) if guidance > 1 else context,
        initial_ii,
        steps=2,
        guidance=guidance,
        generator=actual_generator,
        upscaled=upscaled,
        noise_level=5,
    )
    assert torch.equal(actual_ii, expected_ii)
    pixels = ((actual_ii / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
    expected_pixels = ((expected_ii / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
    assert torch.equal(pixels, expected_pixels)
    record(f"deepfloyd_if_tiny_g{guidance}_seed{seed}", pixels_equal=True)
