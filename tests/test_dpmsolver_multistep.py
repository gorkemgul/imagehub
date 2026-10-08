import pytest
import torch

from imagehub.schedulers.dpmsolver_multistep import DPMSolverMultistepScheduler

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


def test_release_config_and_state_reset():
    scheduler = DPMSolverMultistepScheduler(CONFIG)
    scheduler.set_timesteps(2)
    assert scheduler.timesteps.dtype == torch.int64
    assert scheduler.sigmas.dtype == torch.float32
    assert scheduler.init_noise_sigma == 1.0
    sample = torch.randn(1, 4, 3, 3)
    assert scheduler.scale_model_input(sample, scheduler.timesteps[0]) is sample
    scheduler.step(torch.randn_like(sample), scheduler.timesteps[0], sample)
    scheduler.set_timesteps(1)
    assert scheduler.step_index is None


def test_requires_set_timesteps_before_sampling():
    scheduler = DPMSolverMultistepScheduler(CONFIG)
    sample = torch.randn(1, 4, 3, 3)
    with pytest.raises(RuntimeError, match="set_timesteps"):
        scheduler.step(sample, torch.tensor(1), sample)


@pytest.mark.parametrize(
    "option",
    [
        {"algorithm_type": "sde-dpmsolver++"},
        {"beta_schedule": "scaled_linear"},
        {"euler_at_final": True},
        {"final_sigmas_type": "sigma_min"},
        {"lower_order_final": False},
        {"prediction_type": "v_prediction"},
        {"solver_order": 3},
        {"solver_type": "heun"},
        {"thresholding": True},
        {"timestep_spacing": "leading"},
        {"trained_betas": [0.1, 0.2]},
        {"use_beta_sigmas": True},
        {"use_exponential_sigmas": True},
        {"use_flow_sigmas": True},
        {"flow_shift": 3.0},
        {"prediction_type": "flow_prediction"},
        {"use_karras_sigmas": True},
        {"use_lu_lambdas": True},
        {"unknown_option": True},
    ],
)
def test_rejects_unsupported_options(option):
    with pytest.raises(ValueError, match="Unsupported DPM-Solver setting"):
        DPMSolverMultistepScheduler(CONFIG | option)


SANA_CONFIG = CONFIG | {
    "prediction_type": "flow_prediction",
    "use_flow_sigmas": True,
    "flow_shift": 3.0,
}


def test_flow_schedule_shape_and_ends():
    scheduler = DPMSolverMultistepScheduler(SANA_CONFIG)
    scheduler.set_timesteps(20)
    assert scheduler.timesteps.dtype == torch.int64
    assert scheduler.sigmas.dtype == torch.float32
    assert len(scheduler.timesteps) == 20 and len(scheduler.sigmas) == 21
    assert 0.999 < scheduler.sigmas[0] < 1.0 and scheduler.sigmas[-1] == 0.0
    assert torch.all(scheduler.sigmas[:-1] > scheduler.sigmas[1:])
    assert scheduler.timesteps[0] == 999


@pytest.mark.parametrize("flow_shift", [0.0, -1.0])
def test_rejects_non_positive_flow_shift(flow_shift):
    with pytest.raises(ValueError, match="flow_shift"):
        DPMSolverMultistepScheduler(SANA_CONFIG | {"flow_shift": flow_shift})
