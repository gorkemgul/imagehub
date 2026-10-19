"""Exact parity for the DEIS configuration published with Segmind Small-SD."""

import pytest
import torch

from imagehub.schedulers.deis_multistep import DEISMultistepScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

CONFIG = {
    "algorithm_type": "deis",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "solver_order": 2,
    "solver_type": "logrho",
    "steps_offset": 1,
    "thresholding": False,
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_karras_sigmas": False,
}


@pytest.mark.parametrize("steps", [4, 14, 20, 50])
def test_deis_trajectory_matches_reference(steps):
    expected_scheduler = diffusers.DEISMultistepScheduler.from_config(CONFIG)
    actual_scheduler = DEISMultistepScheduler(CONFIG)
    expected_scheduler.set_timesteps(steps)
    actual_scheduler.set_timesteps(steps)
    assert torch.equal(actual_scheduler.timesteps, expected_scheduler.timesteps)
    assert torch.equal(actual_scheduler.sigmas, expected_scheduler.sigmas)
    generator = torch.Generator("cpu").manual_seed(0)
    expected = torch.randn(2, 4, 6, 10, generator=generator)
    actual = expected.clone()
    for timestep in expected_scheduler.timesteps:
        model_output = torch.randn(expected.shape, generator=generator)
        expected = expected_scheduler.step(model_output, timestep, expected).prev_sample
        actual = actual_scheduler.step(model_output, timestep, actual)
        assert torch.equal(actual, expected)
