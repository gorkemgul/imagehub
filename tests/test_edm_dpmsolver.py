import json
from pathlib import Path

import pytest
import torch

from imagehub.schedulers.edm_dpmsolver import EDMDPMSolverMultistepScheduler

CONFIG = json.loads(
    (Path(__file__).parent / "fixtures/playground_v25_configs/scheduler_config.json").read_text(
        encoding="utf-8"
    )
)


def test_release_config_and_state_reset():
    scheduler = EDMDPMSolverMultistepScheduler(CONFIG)
    scheduler.set_timesteps(2)
    assert scheduler.timesteps.dtype == torch.float32
    assert scheduler.sigmas.dtype == torch.float32
    assert scheduler.init_noise_sigma == pytest.approx((80.0**2 + 1) ** 0.5)
    sample = torch.randn(1, 4, 3, 3)
    scheduler.scale_model_input(sample, scheduler.timesteps[0])
    scheduler.step(torch.randn_like(sample), scheduler.timesteps[0], sample)
    scheduler.set_timesteps(1)
    assert scheduler.step_index is None


def test_requires_set_timesteps_before_sampling():
    scheduler = EDMDPMSolverMultistepScheduler(CONFIG)
    sample = torch.randn(1, 4, 3, 3)
    with pytest.raises(RuntimeError, match="set_timesteps"):
        scheduler.scale_model_input(sample, torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="set_timesteps"):
        scheduler.step(sample, torch.tensor(1.0), sample)


@pytest.mark.parametrize(
    "option",
    [
        {"algorithm_type": "sde-dpmsolver++"},
        {"euler_at_final": True},
        {"final_sigmas_type": "sigma_min"},
        {"lower_order_final": False},
        {"prediction_type": "v_prediction"},
        {"rho": 3.0},
        {"sigma_data": 1.0},
        {"sigma_max": 10.0},
        {"sigma_min": 0.01},
        {"solver_order": 3},
        {"solver_type": "heun"},
        {"thresholding": True},
        {"unknown_option": True},
    ],
)
def test_rejects_unsupported_options(option):
    with pytest.raises(ValueError, match="Unsupported EDM DPM-Solver setting"):
        EDMDPMSolverMultistepScheduler(CONFIG | option)
