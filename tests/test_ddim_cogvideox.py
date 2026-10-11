import json
from pathlib import Path

import pytest
import torch

from imagehub.schedulers.ddim_cogvideox import CogVideoXDDIMScheduler

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = json.loads(
    (FIXTURES / "cogview3_plus_configs/scheduler_scheduler_config.json").read_text()
)


def test_release_config_tables():
    s = CogVideoXDDIMScheduler(CONFIG)
    assert s.alphas_cumprod.shape == (1000,) and s.alphas_cumprod.dtype == torch.float64
    assert s.alphas_cumprod[-1] == 0
    s.set_timesteps(50)
    t = s.timesteps.tolist()
    assert t[:3] == [999, 979, 959] and t[-2:] == [39, 19] and len(t) == 50


@pytest.mark.parametrize(
    ("spacing", "first", "last"),
    [("leading", 900, 0), ("linspace", 999, 0), ("trailing", 999, 99)],
)
def test_timestep_spacings(spacing, first, last):
    s = CogVideoXDDIMScheduler(CONFIG | {"timestep_spacing": spacing})
    s.set_timesteps(10)
    assert (s.timesteps[0].item(), s.timesteps[-1].item()) == (first, last)


def test_v_prediction_step_closed_form():
    s = CogVideoXDDIMScheduler(CONFIG)
    s.set_timesteps(50)
    generator = torch.Generator().manual_seed(0)
    sample, v = (
        torch.randn(1, 4, 2, 2, generator=generator),
        torch.randn(1, 4, 2, 2, generator=generator),
    )
    a_t, a_prev = s.alphas_cumprod[999], s.alphas_cumprod[979]
    x0 = a_t**0.5 * sample - (1 - a_t) ** 0.5 * v
    a = ((1 - a_prev) / (1 - a_t)) ** 0.5
    expected = a * sample + (a_prev**0.5 - a_t**0.5 * a) * x0
    assert torch.equal(s.step(v, s.timesteps[0], sample), expected)


def test_final_step_uses_final_alpha():
    s = CogVideoXDDIMScheduler(CONFIG)
    s.set_timesteps(50)
    assert s.final_alpha_cumprod == 1.0
    out = s.step(torch.zeros(1, 4, 2, 2), s.timesteps[-1], torch.ones(1, 4, 2, 2))
    assert out.shape == (1, 4, 2, 2) and torch.isfinite(out).all()


@pytest.mark.parametrize(
    "override",
    [
        {"trained_betas": [0.1, 0.2]},
        {"beta_schedule": "cosine"},
        {"prediction_type": "flow"},
        {"timestep_spacing": "other"},
        {"unknown_option": 1},
    ],
)
def test_unsupported_settings_fail(override):
    with pytest.raises(ValueError, match="Unsupported"):
        CogVideoXDDIMScheduler(CONFIG | override)


def test_set_timesteps_validation_and_step_order():
    s = CogVideoXDDIMScheduler(CONFIG)
    for bad in (0, 1001, 2.5, True):
        with pytest.raises(ValueError, match="num_inference_steps"):
            s.set_timesteps(bad)
    with pytest.raises(RuntimeError, match="set_timesteps"):
        s.step(torch.zeros(1), torch.tensor(5), torch.zeros(1))
