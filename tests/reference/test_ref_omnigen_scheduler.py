"""Inverted flow-match Euler schedule and steps against diffusers (OmniGen's scheduler config)."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.backends.omnigen import omnigen_sigmas
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
np = pytest.importorskip("numpy")

CONFIG = json.loads(
    (
        Path(__file__).parents[1] / "fixtures/omnigen_configs/scheduler_scheduler_config.json"
    ).read_text()
)


@pytest.mark.parametrize("steps", [1, 2, 3, 5, 25, 50, 100])
def test_schedule_and_trajectory_match(record, steps):
    reference = diffusers.FlowMatchEulerDiscreteScheduler.from_config(CONFIG)
    reference.set_timesteps(sigmas=np.linspace(1, 0, steps + 1)[:steps])
    ours = FlowMatchEulerDiscreteScheduler(CONFIG, allow_inverted_sigmas=True)
    ours.set_timesteps(steps, sigmas=omnigen_sigmas(steps))
    assert torch.equal(ours.timesteps, reference.timesteps)
    assert torch.equal(ours.sigmas, reference.sigmas)
    generator = torch.Generator().manual_seed(steps)
    a = b = torch.randn(1, 4, 8, 8, generator=generator)
    for t in reference.timesteps:
        output = torch.randn(1, 4, 8, 8, generator=generator)
        a = reference.step(output, t, a, return_dict=False)[0]
        b = ours.step(output, t, b)
    assert torch.equal(a, b)
    record(f"omnigen_scheduler_{steps}", max_abs_diff=0.0)
