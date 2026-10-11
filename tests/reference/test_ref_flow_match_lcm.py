"""FlowMatchLCMScheduler parity: schedules and seeded stochastic trajectories (exact)."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.schedulers.flow_match_lcm import FlowMatchLCMScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("numpy")

CONFIG = json.loads(
    (
        Path(__file__).parents[1] / "fixtures/hidream_configs/scheduler_scheduler_config.json"
    ).read_text(encoding="utf-8")
)
# The pipeline passes a resolution-dependent mu; the static shift ignores it.
MU = 1.0986122886681098


def _pair():
    config = {k: v for k, v in CONFIG.items() if not k.startswith("_")}
    return diffusers.FlowMatchLCMScheduler.from_config(config), FlowMatchLCMScheduler(CONFIG)


def test_initial_schedule_matches():
    reference, scheduler = _pair()
    assert torch.equal(scheduler.timesteps, reference.timesteps)
    assert torch.equal(scheduler.sigmas, reference.sigmas)
    assert scheduler.sigma_min == reference.sigma_min
    assert scheduler.sigma_max == reference.sigma_max


@pytest.mark.parametrize("steps", [1, 2, 3, 4, 7, 8, 16, 28, 50, 100])
def test_set_timesteps_matches(steps):
    reference, scheduler = _pair()
    reference.set_timesteps(steps, device="cpu", mu=MU)
    scheduler.set_timesteps(steps, mu=MU)
    assert torch.equal(scheduler.timesteps, reference.timesteps)
    assert torch.equal(scheduler.sigmas, reference.sigmas)
    assert scheduler.timesteps.dtype == reference.timesteps.dtype


def test_custom_sigmas_match():
    reference, scheduler = _pair()
    sigmas = [1.0, 0.8, 0.55, 0.3, 0.1]
    reference.set_timesteps(sigmas=sigmas, device="cpu", mu=MU)
    scheduler.set_timesteps(len(sigmas), sigmas=sigmas, mu=MU)
    assert torch.equal(scheduler.timesteps, reference.timesteps)
    assert torch.equal(scheduler.sigmas, reference.sigmas)


def _trajectory(reference, scheduler, steps, make_generators, shape=(2, 16, 12, 10)):
    reference.set_timesteps(steps, device="cpu", mu=MU)
    scheduler.set_timesteps(steps, mu=MU)
    source = torch.Generator().manual_seed(11)
    expected = actual = torch.randn(shape, generator=source)
    generators = make_generators()
    for index in range(steps):
        output = torch.randn(shape, generator=source)
        step_generators = generators()
        expected = reference.step(
            output,
            reference.timesteps[index],
            expected,
            generator=step_generators[0],
            return_dict=False,
        )[0]
        actual = scheduler.step(output, scheduler.timesteps[index], actual, step_generators[1])
        assert torch.equal(actual, expected), f"step {index}"
    assert scheduler.step_index == reference.step_index == steps


@pytest.mark.parametrize("steps", [1, 2, 4, 8, 16])
def test_trajectory_with_shared_seed_generators(steps):
    reference, scheduler = _pair()
    state = (torch.Generator().manual_seed(5), torch.Generator().manual_seed(5))
    _trajectory(reference, scheduler, steps, lambda: lambda: state)


def test_trajectory_with_global_generator():
    reference, scheduler = _pair()
    reference.set_timesteps(4, device="cpu", mu=MU)
    scheduler.set_timesteps(4, mu=MU)
    source = torch.Generator().manual_seed(2)
    expected = actual = torch.randn(1, 16, 8, 8, generator=source)
    for index in range(4):
        output = torch.randn(1, 16, 8, 8, generator=source)
        torch.manual_seed(100 + index)
        expected = reference.step(output, reference.timesteps[index], expected, return_dict=False)[
            0
        ]
        torch.manual_seed(100 + index)
        actual = scheduler.step(output, scheduler.timesteps[index], actual)
        assert torch.equal(actual, expected)


def test_trajectory_with_generator_list():
    reference, scheduler = _pair()
    reference.set_timesteps(4, device="cpu", mu=MU)
    scheduler.set_timesteps(4, mu=MU)
    source = torch.Generator().manual_seed(3)
    expected = actual = torch.randn(2, 16, 8, 8, generator=source)
    left = [torch.Generator().manual_seed(seed) for seed in (7, 8)]
    right = [torch.Generator().manual_seed(seed) for seed in (7, 8)]
    for index in range(4):
        output = torch.randn(2, 16, 8, 8, generator=source)
        expected = reference.step(
            output, reference.timesteps[index], expected, generator=left, return_dict=False
        )[0]
        actual = scheduler.step(output, scheduler.timesteps[index], actual, right)
        assert torch.equal(actual, expected)


def test_half_precision_output_dtype_matches():
    reference, scheduler = _pair()
    reference.set_timesteps(2, device="cpu", mu=MU)
    scheduler.set_timesteps(2, mu=MU)
    sample = torch.randn(1, 4, 4, 4)
    output = torch.randn(1, 4, 4, 4).bfloat16()
    expected = reference.step(
        output,
        reference.timesteps[0],
        sample,
        generator=torch.Generator().manual_seed(1),
        return_dict=False,
    )[0]
    actual = scheduler.step(
        output, scheduler.timesteps[0], sample, torch.Generator().manual_seed(1)
    )
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual, expected)
