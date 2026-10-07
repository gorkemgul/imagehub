"""DDPM scheduler parity for the HunyuanDiT release configuration."""

import json

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub.schedulers.ddpm import DDPMScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")


@pytest.mark.parametrize("steps", [25, 50])
def test_ddpm_timesteps_match_reference(steps, record):
    root = snapshot_dir("hunyuan-dit-v1.1-distilled")
    config = json.loads((root / "scheduler/scheduler_config.json").read_text(encoding="utf-8"))
    ours = DDPMScheduler(config)
    reference = diffusers.DDPMScheduler.from_config(config)
    ours.set_timesteps(steps)
    reference.set_timesteps(steps)
    exact = torch.equal(ours.timesteps, reference.timesteps)
    record(f"hunyuan_dit_ddpm_timesteps_{steps}", exact=exact, tolerance="exact")
    assert exact


@pytest.mark.parametrize("prediction_type", ["v_prediction", "epsilon"])
def test_ddpm_trajectory_matches_reference(prediction_type, record):
    root = snapshot_dir("hunyuan-dit-v1.1-distilled")
    config = json.loads((root / "scheduler/scheduler_config.json").read_text(encoding="utf-8"))
    config["prediction_type"] = prediction_type
    ours = DDPMScheduler(config)
    reference = diffusers.DDPMScheduler.from_config(config)
    ours.set_timesteps(25)
    reference.set_timesteps(25)

    output_generator = torch.Generator().manual_seed(1)
    sample = torch.randn(2, 4, 8, 8, generator=output_generator)
    expected = sample.clone()
    outputs = [torch.randn(sample.shape, generator=output_generator) for _ in ours.timesteps]
    actual_generator = torch.Generator().manual_seed(42)
    expected_generator = torch.Generator().manual_seed(42)
    worst = 0.0
    for timestep, model_output in zip(ours.timesteps, outputs, strict=True):
        expected = reference.step(
            model_output,
            timestep,
            expected,
            generator=expected_generator,
        ).prev_sample
        sample = ours.step(model_output, timestep, sample, generator=actual_generator)
        worst = max(worst, max_diff(sample, expected)["max_abs"])
        torch.testing.assert_close(sample, expected, atol=0, rtol=0)
    record(
        f"hunyuan_dit_ddpm_{prediction_type}",
        max_abs=worst,
        steps=25,
        tolerance=0.0,
    )
