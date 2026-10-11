import json
from pathlib import Path

import pytest
import torch

from imagehub.schedulers.euler_ancestral import EulerAncestralDiscreteScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
CONFIG = json.loads(
    (Path(__file__).parents[1] / "fixtures/sdxl_turbo_configs/scheduler_config.json").read_text()
)


@pytest.mark.parametrize("steps", [1, 2, 4, 50])
@pytest.mark.parametrize("spacing", ["leading", "trailing"])
def test_ancestral_timesteps_and_sigmas_match_diffusers_exactly(steps, spacing):
    config = CONFIG | {"timestep_spacing": spacing}
    actual = EulerAncestralDiscreteScheduler(config)
    expected = diffusers.EulerAncestralDiscreteScheduler.from_config(config)
    actual.set_timesteps(steps)
    expected.set_timesteps(steps)
    assert torch.equal(actual.timesteps, expected.timesteps)
    assert torch.equal(actual.sigmas, expected.sigmas)


def test_ancestral_seeded_trajectory_matches_diffusers_exactly():
    actual = EulerAncestralDiscreteScheduler(CONFIG)
    expected = diffusers.EulerAncestralDiscreteScheduler.from_config(CONFIG)
    actual.set_timesteps(4)
    expected.set_timesteps(4)
    sample = torch.randn((2, 4, 3, 5), generator=torch.Generator().manual_seed(2))
    actual_sample = sample.clone()
    expected_sample = sample.clone()
    actual_generator = torch.Generator("cpu").manual_seed(3)
    expected_generator = torch.Generator("cpu").manual_seed(3)
    for index, timestep in enumerate(actual.timesteps):
        model_output = torch.randn(
            sample.shape, generator=torch.Generator().manual_seed(10 + index)
        )
        actual.scale_model_input(actual_sample, timestep)
        expected.scale_model_input(expected_sample, expected.timesteps[index])
        actual_sample = actual.step(
            model_output, timestep, actual_sample, generator=actual_generator
        )
        expected_sample = expected.step(
            model_output,
            expected.timesteps[index],
            expected_sample,
            generator=expected_generator,
        ).prev_sample
        assert torch.equal(actual_sample, expected_sample)


@pytest.mark.parametrize("steps", [2, 30, 50])
def test_playground_v2_schedule_matches_diffusers_exactly(steps):
    config = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/playground_v2_configs/scheduler_config.json"
        ).read_text()
    )
    actual = EulerAncestralDiscreteScheduler(config)
    expected = diffusers.EulerAncestralDiscreteScheduler.from_config(config)
    actual.set_timesteps(steps)
    expected.set_timesteps(steps)
    assert torch.equal(actual.timesteps, expected.timesteps)
    assert torch.equal(actual.sigmas, expected.sigmas)
    assert torch.equal(actual.init_noise_sigma, expected.init_noise_sigma)
