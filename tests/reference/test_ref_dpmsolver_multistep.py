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


def test_scaled_linear_published_sd_trajectory_matches_reference():
    config = CONFIG | {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "steps_offset": 1,
    }
    expected_scheduler = diffusers.DPMSolverMultistepScheduler.from_config(config)
    actual_scheduler = DPMSolverMultistepScheduler(config)
    expected_scheduler.set_timesteps(20)
    actual_scheduler.set_timesteps(20)
    exact(actual_scheduler.timesteps, expected_scheduler.timesteps)
    exact(actual_scheduler.sigmas, expected_scheduler.sigmas)
    generator = torch.Generator("cpu").manual_seed(0)
    expected = torch.randn(2, 4, 6, 10, generator=generator)
    actual = expected.clone()
    for timestep in expected_scheduler.timesteps:
        model_output = torch.randn(expected.shape, generator=generator)
        expected = expected_scheduler.step(model_output, timestep, expected).prev_sample
        actual = actual_scheduler.step(model_output, timestep, actual)
        exact(actual, expected)


FLOW_CONFIG = CONFIG | {
    "prediction_type": "flow_prediction",
    "use_flow_sigmas": True,
    "flow_shift": 3.0,
    "final_sigmas_type": "zero",
}


def _sana_config():
    import json
    from pathlib import Path

    path = Path(__file__).parents[1] / "fixtures/sana_configs/1600m/scheduler_scheduler_config.json"
    return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}


@pytest.mark.parametrize("steps", [1, 2, 5, 14, 20, 25, 50])
@pytest.mark.parametrize("shift", [1.0, 3.0, 7.5])
def test_flow_timesteps_match_reference(steps, shift):
    config = FLOW_CONFIG | {"flow_shift": shift}
    expected = diffusers.DPMSolverMultistepScheduler.from_config(config)
    actual = DPMSolverMultistepScheduler(config)
    expected.set_timesteps(steps)
    actual.set_timesteps(steps)
    exact(actual.timesteps, expected.timesteps)
    exact(actual.sigmas, expected.sigmas)


def test_published_sana_config_is_accepted_and_matches_reference():
    config = _sana_config()
    expected = diffusers.DPMSolverMultistepScheduler.from_config(config)
    actual = DPMSolverMultistepScheduler(config)
    expected.set_timesteps(20)
    actual.set_timesteps(20)
    exact(actual.timesteps, expected.timesteps)
    exact(actual.sigmas, expected.sigmas)


@pytest.mark.parametrize("steps", [3, 14, 20])
def test_flow_trajectory_matches_reference(steps):
    expected_scheduler = diffusers.DPMSolverMultistepScheduler.from_config(_sana_config())
    actual_scheduler = DPMSolverMultistepScheduler(_sana_config())
    expected_scheduler.set_timesteps(steps)
    actual_scheduler.set_timesteps(steps)
    generator = torch.Generator("cpu").manual_seed(0)
    expected = torch.randn(2, 32, 6, 10, generator=generator)
    actual = expected.clone()
    for timestep in expected_scheduler.timesteps:
        model_output = torch.randn(expected.shape, generator=generator)
        expected = expected_scheduler.step(model_output, timestep, expected).prev_sample
        actual = actual_scheduler.step(model_output, timestep, actual)
        exact(actual, expected)
