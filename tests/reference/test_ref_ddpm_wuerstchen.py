"""Wuerstchen DDPM scheduler parity against diffusers on the Stable Cascade config."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.schedulers.ddpm_wuerstchen import DDPMWuerstchenScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

CONFIG = json.loads(
    (
        Path(__file__).parents[1] / "fixtures/stable_cascade_configs/scheduler_config.json"
    ).read_text()
)


@pytest.mark.parametrize("steps", [12, 20, 60])
def test_timesteps_match_reference(steps, record):
    ours = DDPMWuerstchenScheduler(CONFIG)
    reference = diffusers.DDPMWuerstchenScheduler.from_config(CONFIG)
    ours.set_timesteps(steps)
    reference.set_timesteps(steps)
    exact = torch.equal(ours.timesteps, reference.timesteps)
    record(f"ddpm_wuerstchen_timesteps_{steps}", exact=exact, tolerance="exact")
    assert exact


@pytest.mark.parametrize("scaler", [1.0, 2.0, 0.5])
def test_trajectory_matches_reference(scaler, record):
    config = CONFIG | {"scaler": scaler}
    ours = DDPMWuerstchenScheduler(config)
    reference = diffusers.DDPMWuerstchenScheduler.from_config(config)
    ours.set_timesteps(12)
    reference.set_timesteps(12)
    data = torch.Generator().manual_seed(1)
    sample = torch.randn(2, 16, 6, 6, generator=data)
    expected = sample.clone()
    steps = reference.timesteps[:-1]
    outputs = [torch.randn(sample.shape, generator=data) for _ in steps]
    actual_generator = torch.Generator().manual_seed(42)
    expected_generator = torch.Generator().manual_seed(42)
    for timestep, model_output in zip(steps, outputs, strict=True):
        ratio = timestep.expand(sample.size(0))
        expected = reference.step(
            model_output, ratio, expected, generator=expected_generator
        ).prev_sample
        sample = ours.step(model_output, ratio, sample, generator=actual_generator)
        assert torch.equal(sample, expected)
    record(f"ddpm_wuerstchen_trajectory_scaler_{scaler}", steps=12, tolerance="exact")
