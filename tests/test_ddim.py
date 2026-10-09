import json
from pathlib import Path

import pytest
import torch

from imagehub.schedulers.ddim import DDIMScheduler

FIXTURES = Path(__file__).parent / "fixtures"
SD21 = json.loads((FIXTURES / "sd21_configs/scheduler_scheduler_config.json").read_text())


def test_release_config_defaults_to_leading_spacing_with_offset():
    s = DDIMScheduler(SD21)
    assert s.config["timestep_spacing"] == "leading"
    assert s.alphas_cumprod.shape == (1000,)
    assert s.final_alpha_cumprod == s.alphas_cumprod[0]
    s.set_timesteps(50)
    t = s.timesteps.tolist()
    assert t[:3] == [981, 961, 941] and t[-2:] == [21, 1] and len(t) == 50
    s.set_timesteps(25)
    assert s.timesteps.tolist()[:2] == [961, 921] and s.timesteps.tolist()[-1] == 1


@pytest.mark.parametrize(
    ("spacing", "first", "last"),
    [("leading", 901, 1), ("linspace", 999, 0), ("trailing", 999, 99)],
)
def test_timestep_spacings(spacing, first, last):
    s = DDIMScheduler(SD21 | {"timestep_spacing": spacing})
    s.set_timesteps(10)
    assert (s.timesteps[0].item(), s.timesteps[-1].item()) == (first, last)


def test_v_prediction_step_closed_form():
    s = DDIMScheduler(SD21)
    s.set_timesteps(50)
    generator = torch.Generator().manual_seed(0)
    sample, v = (
        torch.randn(1, 4, 2, 2, generator=generator),
        torch.randn(1, 4, 2, 2, generator=generator),
    )
    a_t, a_prev = s.alphas_cumprod[981], s.alphas_cumprod[961]
    x0 = a_t**0.5 * sample - (1 - a_t) ** 0.5 * v
    eps = a_t**0.5 * v + (1 - a_t) ** 0.5 * sample
    expected = a_prev**0.5 * x0 + (1 - a_prev) ** 0.5 * eps
    assert torch.equal(s.step(v, s.timesteps[0], sample), expected)


def test_final_step_uses_final_alpha():
    s = DDIMScheduler(SD21 | {"prediction_type": "epsilon"})
    s.set_timesteps(50)
    sample = torch.ones(1, 4, 2, 2)
    out = s.step(torch.zeros_like(sample), s.timesteps[-1], sample)
    a_t = s.alphas_cumprod[1]
    expected = s.final_alpha_cumprod**0.5 * (sample / a_t**0.5) + (
        1 - s.final_alpha_cumprod
    ) ** 0.5 * torch.zeros_like(sample)
    assert torch.equal(out, expected)


def test_clip_sample_clamps_prediction():
    s = DDIMScheduler(SD21 | {"prediction_type": "epsilon", "clip_sample": True})
    s.set_timesteps(50)
    out = s.step(torch.zeros(1, 4, 2, 2), s.timesteps[0], torch.full((1, 4, 2, 2), 100.0))
    assert out.abs().max() < 2


def test_legacy_keys_are_ignored():
    assert "skip_prk_steps" in SD21
    DDIMScheduler(SD21)


@pytest.mark.parametrize(
    "override",
    [
        {"thresholding": True},
        {"rescale_betas_zero_snr": True},
        {"trained_betas": [0.1, 0.2]},
        {"beta_schedule": "squaredcos_cap_v2"},
        {"prediction_type": "flow"},
        {"timestep_spacing": "geometric"},
        {"eta": 1.0},
    ],
)
def test_unsupported_settings(override):
    with pytest.raises(ValueError):
        DDIMScheduler(SD21 | override)


def test_step_requires_timesteps():
    with pytest.raises(RuntimeError, match="set_timesteps"):
        DDIMScheduler(SD21).step(torch.zeros(1), 981, torch.zeros(1))
    for bad in (0, 1001, 2.5):
        with pytest.raises(ValueError):
            DDIMScheduler(SD21).set_timesteps(bad)
