"""Exact small-tensor parity for HiDream Full's published UniPC scheduler."""

import json

import pytest
import torch
from fixture_files import restricted_fixture

from imagehub.schedulers.unipc import UniPCMultistepScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")


def published_config():
    return json.loads(
        restricted_fixture("hidream_configs/full/scheduler_scheduler_config.json").read_text()
    )


@pytest.mark.parametrize("steps", [1, 2, 4, 7, 16, 50])
def test_unipc_flow_schedule_and_trajectory_match(steps):
    config = published_config()
    reference = diffusers.UniPCMultistepScheduler.from_config(config)
    native = UniPCMultistepScheduler(config)
    reference.set_timesteps(steps)
    native.set_timesteps(steps)
    assert torch.equal(native.timesteps, reference.timesteps)
    assert torch.equal(native.sigmas, reference.sigmas)
    generator = torch.Generator().manual_seed(42)
    expected = actual = torch.randn(2, 4, 8, 8, generator=generator)
    for index in range(steps):
        output = torch.randn(actual.shape, generator=generator)
        expected = reference.step(output, reference.timesteps[index], expected, return_dict=False)[
            0
        ]
        actual = native.step(output, native.timesteps[index], actual)
        assert torch.equal(actual, expected)
