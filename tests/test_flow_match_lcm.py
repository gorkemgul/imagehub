import json
from pathlib import Path

import pytest
import torch

from imagehub.schedulers.flow_match_lcm import FlowMatchLCMScheduler


def published_config():
    path = Path(__file__).parent / "fixtures/hidream_configs/scheduler_scheduler_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_published_config_schedule():
    scheduler = FlowMatchLCMScheduler(published_config())
    assert len(scheduler) == 1000
    scheduler.set_timesteps(4, mu=0.8)
    assert scheduler.timesteps.dtype == torch.float32
    assert scheduler.sigmas.shape == (5,)
    assert scheduler.timesteps[0].item() == 1000.0
    assert scheduler.sigmas[-1].item() == 0.0
    assert torch.equal(scheduler.timesteps, scheduler.sigmas[:-1] * 1000)


def test_step_renoises_from_the_generator():
    scheduler = FlowMatchLCMScheduler(published_config())
    scheduler.set_timesteps(2)
    sample = torch.randn(2, 3, 4, 4)
    model_output = torch.randn(2, 3, 4, 4)
    first = scheduler.step(
        model_output, scheduler.timesteps[0], sample, generator=torch.Generator().manual_seed(1)
    )
    noise = torch.randn(sample.shape, generator=torch.Generator().manual_seed(1))
    sigma, sigma_next = scheduler.sigmas[0], scheduler.sigmas[1]
    expected = (1 - sigma_next) * (sample - sigma * model_output) + sigma_next * noise
    assert torch.equal(first, expected)
    assert scheduler.step_index == 1


def test_last_step_is_the_prediction():
    scheduler = FlowMatchLCMScheduler(published_config())
    scheduler.set_timesteps(1)
    sample, model_output = torch.randn(1, 2, 2, 2), torch.randn(1, 2, 2, 2)
    result = scheduler.step(model_output, scheduler.timesteps[0], sample)
    assert torch.equal(result, sample - scheduler.sigmas[0] * model_output)


def test_generator_list_draws_per_sample_noise():
    scheduler = FlowMatchLCMScheduler(published_config())
    scheduler.set_timesteps(2)
    zeros = torch.zeros(2, 1, 2, 2)
    generators = [torch.Generator().manual_seed(seed) for seed in (3, 4)]
    result = scheduler.step(zeros, scheduler.timesteps[0], zeros, generator=generators)
    for row, seed in enumerate((3, 4)):
        noise = torch.randn((1, 1, 2, 2), generator=torch.Generator().manual_seed(seed))
        assert torch.equal(result[row : row + 1], scheduler.sigmas[1] * noise)


@pytest.mark.parametrize(
    "change",
    [
        {"use_dynamic_shifting": True},
        {"shift_terminal": 0.1},
        {"use_karras_sigmas": True},
        {"scale_factors": [0.5]},
        {"invert_sigmas": True},
        {"upscale_mode": "nearest"},
        {"time_shift_type": "cubic"},
        {"stochastic_sampling": True},
    ],
)
def test_rejects_unsupported_options(change):
    with pytest.raises(ValueError, match="Unsupported FlowMatch LCM"):
        FlowMatchLCMScheduler(published_config() | change)


def test_step_validation():
    scheduler = FlowMatchLCMScheduler(published_config())
    sample = torch.zeros(1, 1, 2, 2)
    with pytest.raises(RuntimeError):
        scheduler.step(sample, torch.tensor(1000.0), sample)
    scheduler.set_timesteps(2)
    with pytest.raises(ValueError, match="floating-point"):
        scheduler.step(sample, 3, sample)
    with pytest.raises(ValueError, match="not in the current schedule"):
        scheduler.step(sample, torch.tensor(1.5), sample)
    with pytest.raises(ValueError):
        scheduler.set_timesteps(0)
    with pytest.raises(ValueError):
        scheduler.set_timesteps(2, sigmas=[1.0])
