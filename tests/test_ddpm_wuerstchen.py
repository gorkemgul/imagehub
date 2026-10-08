import pytest
import torch

from imagehub.schedulers.ddpm_wuerstchen import DDPMWuerstchenScheduler

CONFIG = {
    "_class_name": "DDPMWuerstchenScheduler",
    "_diffusers_version": "0.27.0.dev0",
    "s": 0.008,
    "scaler": 1.0,
}


def test_rejects_unknown_option():
    with pytest.raises(ValueError, match="Unsupported"):
        DDPMWuerstchenScheduler(CONFIG | {"num_train_timesteps": 1000})


def test_rejects_non_positive_scaler():
    with pytest.raises(ValueError, match="scaler"):
        DDPMWuerstchenScheduler(CONFIG | {"scaler": 0.0})


def test_timesteps_run_from_one_to_zero():
    scheduler = DDPMWuerstchenScheduler(CONFIG)
    scheduler.set_timesteps(4)
    assert scheduler.timesteps.tolist() == [1.0, 0.75, 0.5, 0.25, 0.0]


def test_last_step_still_consumes_generator_noise():
    scheduler = DDPMWuerstchenScheduler(CONFIG)
    scheduler.set_timesteps(2)
    sample = torch.zeros(1, 2, 3, 3)
    ratio = scheduler.timesteps[1:2]
    generator = torch.Generator().manual_seed(0)
    scheduler.step(torch.zeros_like(sample), ratio, sample, generator=generator)
    untouched = torch.Generator().manual_seed(0)
    assert not torch.equal(torch.randn(4, generator=generator), torch.randn(4, generator=untouched))
