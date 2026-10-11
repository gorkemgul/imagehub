import json
from pathlib import Path

import pytest
import torch

from imagehub.schedulers.euler_ancestral import EulerAncestralDiscreteScheduler

CONFIG = json.loads(
    (Path(__file__).parent / "fixtures/sdxl_turbo_configs/scheduler_config.json").read_text()
)


def test_accepts_inert_published_settings():
    config = json.loads(
        (Path(__file__).parent / "fixtures/playground_v2_configs/scheduler_config.json").read_text()
    )
    scheduler = EulerAncestralDiscreteScheduler(config)
    scheduler.set_timesteps(4)
    assert scheduler.sigmas.shape == (5,)


@pytest.mark.parametrize("option", [{"use_karras_sigmas": True}, {"sigma_max": 80.0}])
def test_rejects_active_inert_settings(option):
    with pytest.raises(ValueError, match="Unsupported Euler ancestral setting"):
        EulerAncestralDiscreteScheduler(CONFIG | option)


@pytest.mark.parametrize("spacing", ["leading", "trailing"])
def test_constructs_supported_release_schedules(spacing):
    scheduler = EulerAncestralDiscreteScheduler(CONFIG | {"timestep_spacing": spacing})
    scheduler.set_timesteps(4)
    assert scheduler.timesteps.dtype == torch.float32
    assert scheduler.sigmas.dtype == torch.float32
    assert scheduler.sigmas.shape == (5,)


@pytest.mark.parametrize(
    "option",
    [
        {"prediction_type": "v_prediction"},
        {"beta_schedule": "linear"},
        {"timestep_spacing": "linspace"},
        {"trained_betas": [0.1]},
        {"clip_sample": True},
        {"unknown_option": True},
    ],
)
def test_rejects_unsupported_options(option):
    with pytest.raises(ValueError, match="Unsupported Euler ancestral setting"):
        EulerAncestralDiscreteScheduler(CONFIG | option)


def test_requires_set_timesteps_before_sampling():
    scheduler = EulerAncestralDiscreteScheduler(CONFIG)
    sample = torch.randn(1, 4, 3, 3)
    with pytest.raises(RuntimeError, match="set_timesteps"):
        scheduler.scale_model_input(sample, torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="set_timesteps"):
        scheduler.step(sample, torch.tensor(1.0), sample)


def test_set_timesteps_resets_state():
    scheduler = EulerAncestralDiscreteScheduler(CONFIG)
    scheduler.set_timesteps(2)
    sample = torch.randn(1, 4, 2, 2)
    timestep = scheduler.timesteps[0]
    scheduler.scale_model_input(sample, timestep)
    scheduler.step(sample, timestep, sample, generator=torch.Generator().manual_seed(1))
    scheduler.set_timesteps(1)
    assert scheduler.step_index is None
