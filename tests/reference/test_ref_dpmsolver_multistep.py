"""DPM-Solver++ multistep scheduler parity."""

import pytest
import torch

from imagehub.schedulers.dpmsolver_multistep import DPMSolverMultistepScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

CONFIG = {
    "algorithm_type": "dpmsolver++",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "dynamic_thresholding_ratio": 0.995,
    "euler_at_final": False,
    "lambda_min_clipped": float("-inf"),
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_type": "midpoint",
    "steps_offset": 0,
    "thresholding": False,
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_karras_sigmas": False,
    "use_lu_lambdas": False,
    "variance_type": None,
}


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("steps", [20, 25])
def test_timesteps_match_reference(steps):
    expected = diffusers.DPMSolverMultistepScheduler.from_config(CONFIG)
    actual = DPMSolverMultistepScheduler(CONFIG)
    expected.set_timesteps(steps)
    actual.set_timesteps(steps)
    exact(actual.timesteps, expected.timesteps)
    exact(actual.sigmas, expected.sigmas)


def test_twenty_step_trajectory_matches_reference():
    expected_scheduler = diffusers.DPMSolverMultistepScheduler.from_config(CONFIG)
    actual_scheduler = DPMSolverMultistepScheduler(CONFIG)
    expected_scheduler.set_timesteps(20)
    actual_scheduler.set_timesteps(20)
    generator = torch.Generator("cpu").manual_seed(0)
    expected = torch.randn(2, 4, 6, 10, generator=generator)
    actual = expected.clone()
    for timestep in expected_scheduler.timesteps:
        model_output = torch.randn(expected.shape, generator=generator)
        expected = expected_scheduler.step(model_output, timestep, expected).prev_sample
        actual = actual_scheduler.step(model_output, timestep, actual)
        exact(actual, expected)
