import pytest
import torch
from helpers import LCM_SCHEDULER

from imagehub.schedulers.lcm import LCMScheduler


def test_lcm_timesteps():
    scheduler = LCMScheduler(LCM_SCHEDULER)
    scheduler.set_timesteps(4)
    assert torch.equal(scheduler.timesteps, torch.tensor([999, 759, 499, 259]))
    assert scheduler.init_noise_sigma == 1.0


def test_lcm_step_requires_timesteps():
    scheduler = LCMScheduler(LCM_SCHEDULER)
    with pytest.raises(ValueError, match="set_timesteps"):
        scheduler.step(torch.zeros(1), 999, torch.zeros(1))


@pytest.mark.parametrize(
    "option",
    [
        {"beta_schedule": "linear"},
        {"clip_sample": True},
        {"original_inference_steps": 40},
        {"prediction_type": "sample"},
        {"set_alpha_to_one": False},
        {"thresholding": True},
        {"timestep_spacing": "trailing"},
        {"unknown": True},
    ],
)
def test_lcm_rejects_unsupported_options(option):
    with pytest.raises(ValueError, match="Unsupported LCM setting"):
        LCMScheduler(LCM_SCHEDULER | option)
