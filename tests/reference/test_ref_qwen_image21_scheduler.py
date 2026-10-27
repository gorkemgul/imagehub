"""Qwen-Image 2.1 flow-match scheduler parity."""

import json

import pytest
import torch
from fixture_files import restricted_fixture

from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler, qwen_image_mu

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
np = pytest.importorskip("numpy")
if not hasattr(diffusers, "QwenImage21Pipeline"):
    pytest.skip(
        "needs the diffusers commit pinned for Qwen-Image 2.1 (.venv-ref-qwen-image-2.1)",
        allow_module_level=True,
    )


def published_config():
    return json.loads(
        restricted_fixture("qwen_image21_configs/scheduler_scheduler_config.json").read_text(
            encoding="utf-8"
        )
    )


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("steps", [1, 4, 20, 40])
@pytest.mark.parametrize("height,width", [(1024, 1024), (2048, 2048), (1024, 2048)])
def test_qwen_image21_schedule_matches_pipeline(steps, height, width):
    config = published_config()
    image_seq_len = (height // 16) * (width // 16)
    expected_mu = diffusers.pipelines.qwenimage21.pipeline_qwenimage21.calculate_shift(
        image_seq_len,
        config["base_image_seq_len"],
        config["max_image_seq_len"],
        config["base_shift"],
        config["max_shift"],
    )
    assert qwen_image_mu(height, width, config) == expected_mu
    sigmas = np.linspace(1.0, 1 / steps, steps)
    expected = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    actual = FlowMatchEulerDiscreteScheduler(config)
    expected.set_timesteps(steps, sigmas=sigmas, mu=expected_mu)
    actual.set_timesteps(steps, sigmas=sigmas, mu=expected_mu)
    exact(actual.timesteps, expected.timesteps)
    exact(actual.sigmas, expected.sigmas)


def test_qwen_image21_multistep_trajectory_matches_reference():
    config = published_config()
    steps = 20
    mu = qwen_image_mu(1024, 2048, config)
    sigmas = np.linspace(1.0, 1 / steps, steps)
    expected_scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    actual_scheduler = FlowMatchEulerDiscreteScheduler(config)
    expected_scheduler.set_timesteps(steps, sigmas=sigmas, mu=mu)
    actual_scheduler.set_timesteps(steps, sigmas=sigmas, mu=mu)
    expected_scheduler.set_begin_index(0)
    actual_scheduler.set_begin_index(0)
    generator = torch.Generator("cpu").manual_seed(0)
    expected = torch.randn(2, 64, 8, generator=generator)
    actual = expected.clone()
    for timestep in expected_scheduler.timesteps:
        model_output = torch.randn(expected.shape, generator=generator)
        expected = expected_scheduler.step(model_output, timestep, expected).prev_sample
        actual = actual_scheduler.step(model_output, timestep, actual)
        exact(actual, expected)
