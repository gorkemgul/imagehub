import json
from pathlib import Path

import pytest
import torch

from imagehub.schedulers.ddim import DDIMScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
CONFIG = json.loads(
    (
        Path(__file__).parents[1] / "fixtures/sd21_configs/scheduler_scheduler_config.json"
    ).read_text()
)


@pytest.mark.parametrize("steps", [1, 2, 25, 50, 999, 1000])
@pytest.mark.parametrize("spacing", ["leading", "linspace", "trailing"])
def test_timesteps_match_diffusers_exactly(steps, spacing):
    config = CONFIG | {"timestep_spacing": spacing}
    actual = DDIMScheduler(config)
    expected = diffusers.DDIMScheduler.from_config(config)
    actual.set_timesteps(steps)
    expected.set_timesteps(steps)
    assert torch.equal(actual.timesteps, expected.timesteps)
    assert torch.equal(actual.alphas_cumprod, expected.alphas_cumprod)
    assert torch.equal(actual.final_alpha_cumprod, expected.final_alpha_cumprod)


@pytest.mark.parametrize("steps", [25, 50])
@pytest.mark.parametrize("prediction_type", ["v_prediction", "epsilon", "sample"])
@pytest.mark.parametrize("clip_sample", [False, True])
def test_trajectory_matches_diffusers_exactly(steps, prediction_type, clip_sample):
    config = CONFIG | {"prediction_type": prediction_type, "clip_sample": clip_sample}
    actual = DDIMScheduler(config)
    expected = diffusers.DDIMScheduler.from_config(config)
    actual.set_timesteps(steps)
    expected.set_timesteps(steps)
    generator = torch.Generator().manual_seed(2)
    actual_sample = torch.randn((2, 4, 3, 5), generator=generator)
    expected_sample = actual_sample.clone()
    for index, timestep in enumerate(actual.timesteps):
        model_output = torch.randn(
            actual_sample.shape, generator=torch.Generator().manual_seed(10 + index)
        )
        actual_sample = actual.step(model_output, timestep, actual_sample)
        expected_sample = expected.step(
            model_output, expected.timesteps[index], expected_sample
        ).prev_sample
        assert torch.equal(actual_sample, expected_sample)
