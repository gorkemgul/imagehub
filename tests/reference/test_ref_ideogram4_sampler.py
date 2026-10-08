"""Ideogram 4 logit-normal sampler parity."""

import pytest
import torch

from imagehub.schedulers.ideogram4 import (
    PRESETS,
    get_schedule_for_resolution,
    ideogram4_step,
    make_step_intervals,
)

pytestmark = pytest.mark.reference
ideogram4 = pytest.importorskip("ideogram4")
reference_scheduler = pytest.importorskip("ideogram4.scheduler")
REFERENCE_PRESETS = ideogram4.PRESETS
reference_schedule = reference_scheduler.get_schedule_for_resolution
reference_intervals = reference_scheduler.make_step_intervals


@pytest.mark.parametrize("name", sorted(REFERENCE_PRESETS))
def test_ideogram4_intervals_and_schedule_exact(name):
    expected_preset = REFERENCE_PRESETS[name]
    preset = PRESETS[name]
    assert preset.num_steps == expected_preset.num_steps
    assert preset.guidance_schedule == expected_preset.guidance_schedule
    assert preset.mu == expected_preset.mu
    assert preset.std == expected_preset.std
    torch.testing.assert_close(
        make_step_intervals(preset.num_steps),
        reference_intervals(expected_preset.num_steps),
        atol=0,
        rtol=0,
    )
    points = make_step_intervals(preset.num_steps)
    expected = reference_schedule((768, 1024), known_mean=preset.mu, std=preset.std)(points)
    actual = get_schedule_for_resolution((768, 1024), known_mean=preset.mu, std=preset.std)(points)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("name", sorted(REFERENCE_PRESETS))
def test_ideogram4_full_trajectory_exact(name):
    preset = PRESETS[name]
    expected_schedule = reference_schedule((768, 1024), known_mean=preset.mu, std=preset.std)
    actual_schedule = get_schedule_for_resolution((768, 1024), known_mean=preset.mu, std=preset.std)
    intervals = make_step_intervals(preset.num_steps)
    generator = torch.Generator("cpu").manual_seed(0)
    expected = torch.randn(2, 6, 8, generator=generator)
    actual = expected.clone()

    for index in range(preset.num_steps - 1, -1, -1):
        positive = torch.randn(expected.shape, generator=generator)
        negative = torch.randn(expected.shape, generator=generator)
        expected_t = float(expected_schedule(intervals[index + 1].unsqueeze(0)).item())
        expected_s = float(expected_schedule(intervals[index].unsqueeze(0)).item())
        actual_t = float(actual_schedule(intervals[index + 1].unsqueeze(0)).item())
        actual_s = float(actual_schedule(intervals[index].unsqueeze(0)).item())
        guidance = preset.guidance_schedule[index]
        velocity = guidance * positive + (1.0 - guidance) * negative
        expected = expected + velocity * (expected_s - expected_t)
        actual = ideogram4_step(actual, positive, negative, guidance, actual_t, actual_s)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
