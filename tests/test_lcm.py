import pytest
import torch
from helpers import LCM_SCHEDULER

from imagehub.schedulers.lcm import LCMScheduler


def test_lcm_timesteps():
    scheduler = LCMScheduler(LCM_SCHEDULER)
    scheduler.set_timesteps(4)
    assert torch.equal(scheduler.timesteps, torch.tensor([999, 759, 499, 259]))
    assert scheduler.init_noise_sigma == 1.0


def test_lcm_explicit_timesteps():
    scheduler = LCMScheduler(LCM_SCHEDULER)
    scheduler.set_timesteps(timesteps=[800])
    assert torch.equal(scheduler.timesteps, torch.tensor([800]))
    assert scheduler.num_inference_steps == 1
    sample = torch.randn(1, 4, 8, 8)
    assert scheduler.scale_model_input(sample, 800) is sample


@pytest.mark.parametrize(
    ("num_inference_steps", "timesteps", "match"),
    [
        (None, None, "exactly one"),
        (1, [800], "only pass one"),
        (None, [800, 800], "descending"),
        (None, [1000], "must start before"),
    ],
)
def test_lcm_rejects_invalid_explicit_timesteps(num_inference_steps, timesteps, match):
    scheduler = LCMScheduler(LCM_SCHEDULER)
    with pytest.raises(ValueError, match=match):
        scheduler.set_timesteps(num_inference_steps, timesteps=timesteps)


def test_lcm_step_requires_timesteps():
    scheduler = LCMScheduler(LCM_SCHEDULER)
    with pytest.raises(ValueError, match="set_timesteps"):
        scheduler.step(torch.zeros(1), 999, torch.zeros(1))


@pytest.mark.parametrize(
    "option",
    [
        {"unknown": True},
    ],
)
def test_lcm_rejects_unsupported_options(option):
    with pytest.raises(ValueError, match="Unsupported LCM setting"):
        LCMScheduler(LCM_SCHEDULER | option)


def test_published_pixart_lcm_schedule_is_supported():
    scheduler = LCMScheduler(
        {
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "original_inference_steps": 50,
        }
    )
    scheduler.set_timesteps(4)
    assert scheduler.timesteps.tolist() == [999, 759, 499, 259]
