import json
from pathlib import Path

import pytest
import torch
from helpers import SDXL_SCHEDULER

from imagehub.schedulers.euler import EulerDiscreteScheduler
from imagehub.schedulers.pndm import PNDMScheduler

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("scheduler_cls", "config_path", "training_tensor"),
    [
        (EulerDiscreteScheduler, "sdxl_configs/scheduler.json", "_training_sigmas"),
        (PNDMScheduler, "sd1_configs/v1-5/scheduler.json", "alphas_cumprod"),
        (PNDMScheduler, "sd1_configs/v1-1/scheduler.json", "alphas_cumprod"),
    ],
)
def test_real_release_configs_construct_matching_scheduler(
    scheduler_cls, config_path, training_tensor
):
    config = json.loads((FIXTURES / config_path).read_text(encoding="utf-8"))
    scheduler = scheduler_cls(config)
    assert getattr(scheduler, training_tensor).shape == (config["num_train_timesteps"],)


@pytest.mark.parametrize(
    "option",
    [
        {"clip_sample": True},
        {"sample_max_value": -1.0},
        {"set_alpha_to_one": True},
        {"skip_prk_steps": False},
    ],
)
def test_ignores_legacy_diffusers_options(option):
    scheduler = EulerDiscreteScheduler(SDXL_SCHEDULER | option)
    expected = EulerDiscreteScheduler(SDXL_SCHEDULER)
    torch.testing.assert_close(scheduler._training_sigmas, expected._training_sigmas)


def test_requires_supported_release_settings():
    for option in (
        {"prediction_type": "v_prediction"},
        {"interpolation_type": "log_linear"},
        {"use_karras_sigmas": True},
        {"timestep_spacing": "trailing"},
        {"beta_schedule": "linear"},
    ):
        with pytest.raises(ValueError, match="Unsupported Euler setting"):
            EulerDiscreteScheduler(SDXL_SCHEDULER | option)


def test_set_timesteps_resets_state_and_outputs_float32():
    scheduler = EulerDiscreteScheduler(SDXL_SCHEDULER)
    scheduler.set_timesteps(7)
    assert scheduler.timesteps.dtype == torch.float32
    assert scheduler.sigmas.dtype == torch.float32
    sample = torch.randn(1, 4, 3, 3)
    scaled = scheduler.scale_model_input(sample, scheduler.timesteps[0])
    out = scheduler.step(torch.randn_like(sample), scheduler.timesteps[0], sample)
    assert scaled.dtype == torch.float32
    assert out.dtype == torch.float32
    scheduler.set_timesteps(5)
    assert scheduler.step_index is None


def test_all_training_steps_clamp_the_offset_endpoint():
    scheduler = EulerDiscreteScheduler(SDXL_SCHEDULER)
    scheduler.set_timesteps(1000)
    assert scheduler.timesteps[0] == 1000
    assert scheduler.sigmas.shape == (1001,)


def test_requires_set_timesteps_before_sampling():
    scheduler = EulerDiscreteScheduler(SDXL_SCHEDULER)
    sample = torch.randn(1, 4, 3, 3)
    with pytest.raises(RuntimeError, match="set_timesteps"):
        scheduler.scale_model_input(sample, torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="set_timesteps"):
        scheduler.step(sample, torch.tensor(1.0), sample)


@pytest.mark.parametrize(
    "option",
    [
        {"use_exponential_sigmas": True},
        {"use_beta_sigmas": True},
        {"rescale_betas_zero_snr": True},
        {"final_sigmas_type": "sigma_min"},
        {"unknown_option": True},
    ],
)
def test_rejects_unsupported_options(option):
    with pytest.raises(ValueError, match="Unsupported Euler setting"):
        EulerDiscreteScheduler(SDXL_SCHEDULER | option)
