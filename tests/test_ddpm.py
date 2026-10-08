import pytest
import torch

from imagehub.schedulers.ddpm import DDPMScheduler

DDPM_CONFIG = {
    "_class_name": "DDPMScheduler",
    "_diffusers_version": "0.29.0",
    "beta_end": 0.03,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "clip_sample_range": 1.0,
    "dynamic_thresholding_ratio": 0.995,
    "num_train_timesteps": 1000,
    "prediction_type": "v_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "thresholding": False,
    "timestep_spacing": "leading",
    "trained_betas": None,
    "variance_type": "fixed_small",
}


@pytest.mark.parametrize("steps", [25, 50])
def test_ddpm_timesteps(steps):
    scheduler = DDPMScheduler(DDPM_CONFIG)
    scheduler.set_timesteps(steps)
    ratio = 1000 // steps
    expected = torch.arange(steps - 1, -1, -1, dtype=torch.int64) * ratio + 1
    assert torch.equal(scheduler.timesteps, expected)
    assert scheduler.init_noise_sigma == 1.0


def test_ddpm_scale_model_input_is_identity():
    scheduler = DDPMScheduler(DDPM_CONFIG)
    sample = torch.randn(2, 4, 8, 8)
    assert scheduler.scale_model_input(sample, 1) is sample


def test_ddpm_step_requires_timesteps():
    scheduler = DDPMScheduler(DDPM_CONFIG)
    with pytest.raises(ValueError, match="set_timesteps"):
        scheduler.step(torch.zeros(1), 1, torch.zeros(1))


@pytest.mark.parametrize(
    "option",
    [
        {"beta_schedule": "squaredcos_cap_v2"},
        {"variance_type": "fixed_large"},
        {"prediction_type": "sample"},
        {"rescale_betas_zero_snr": True},
        {"thresholding": True},
        {"timestep_spacing": "trailing"},
        {"trained_betas": [0.1]},
        {"variance_type": "learned"},
        {"unknown": True},
    ],
)
def test_ddpm_rejects_unsupported_options(option):
    with pytest.raises(ValueError, match="Unsupported DDPM setting"):
        DDPMScheduler(DDPM_CONFIG | option)
