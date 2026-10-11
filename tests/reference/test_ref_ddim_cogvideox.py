import json
from pathlib import Path

import pytest
import torch

from imagehub.schedulers.ddim_cogvideox import CogVideoXDDIMScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
CONFIG = json.loads(
    (
        Path(__file__).parents[1] / "fixtures/cogview3_plus_configs/scheduler_scheduler_config.json"
    ).read_text()
)
VARIANTS = [
    {},
    {"timestep_spacing": "leading"},
    {"timestep_spacing": "linspace"},
    {"rescale_betas_zero_snr": False, "snr_shift_scale": 3.0},
    {"beta_schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02},
    {"beta_schedule": "squaredcos_cap_v2"},
    {"set_alpha_to_one": False},
]


@pytest.mark.parametrize("steps", [1, 2, 25, 50, 999, 1000])
@pytest.mark.parametrize("variant", range(len(VARIANTS)))
def test_timesteps_match_diffusers_exactly(steps, variant):
    config = CONFIG | VARIANTS[variant]
    actual = CogVideoXDDIMScheduler(config)
    expected = diffusers.CogVideoXDDIMScheduler.from_config(config)
    actual.set_timesteps(steps)
    expected.set_timesteps(steps)
    assert torch.equal(actual.timesteps, expected.timesteps)
    assert torch.equal(actual.alphas_cumprod, expected.alphas_cumprod)
    assert torch.equal(actual.final_alpha_cumprod, expected.final_alpha_cumprod)


@pytest.mark.parametrize("steps", [25, 50])
@pytest.mark.parametrize("prediction_type", ["v_prediction", "epsilon", "sample"])
@pytest.mark.parametrize("variant", range(len(VARIANTS)))
def test_trajectory_matches_diffusers_exactly(steps, prediction_type, variant):
    config = CONFIG | VARIANTS[variant] | {"prediction_type": prediction_type}
    if prediction_type == "epsilon":
        # Zero terminal SNR makes alpha_prod_t exactly 0 at the first step, which only yields NaN.
        config["rescale_betas_zero_snr"] = False
    actual = CogVideoXDDIMScheduler(config)
    expected = diffusers.CogVideoXDDIMScheduler.from_config(config)
    actual.set_timesteps(steps)
    expected.set_timesteps(steps)
    generator = torch.Generator().manual_seed(2)
    actual_sample = torch.randn((2, 4, 3, 5), generator=generator)
    expected_sample = actual_sample.clone()
    for index, timestep in enumerate(actual.timesteps):
        model_output = torch.randn(
            actual_sample.shape, generator=torch.Generator().manual_seed(10 + index)
        )
        actual_sample = actual.step(model_output, timestep, actual_sample)
        expected_sample = expected.step(
            model_output, expected.timesteps[index], expected_sample, return_dict=False
        )[0]
        assert torch.equal(actual_sample, expected_sample)
