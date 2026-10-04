import pytest
import torch
from helpers import SD1_SCHEDULER

from imagehub.schedulers.pndm import PNDMScheduler


def test_plms_timesteps_repeat_second_to_last():
    s = PNDMScheduler(SD1_SCHEDULER)
    s.set_timesteps(50)
    t = s.timesteps.tolist()
    assert len(t) == 51
    assert t[:4] == [981, 961, 961, 941] and t[-2:] == [21, 1]
    s.set_timesteps(1)
    assert s.timesteps.tolist() == [1]


def test_first_step_closed_form():
    s = PNDMScheduler(SD1_SCHEDULER)
    s.set_timesteps(50)
    torch.manual_seed(0)
    sample, eps = torch.randn(1, 4, 2, 2), torch.randn(1, 4, 2, 2)
    a_t, a_prev = s.alphas_cumprod[981], s.alphas_cumprod[961]
    expected = (a_prev / a_t) ** 0.5 * sample - (a_prev - a_t) * eps / (
        a_t * (1 - a_prev) ** 0.5 + (a_t * (1 - a_t) * a_prev) ** 0.5
    )
    torch.testing.assert_close(s.step(eps, s.timesteps[0], sample), expected)


@pytest.mark.parametrize(
    "override",
    [
        {"skip_prk_steps": False},
        {"prediction_type": "v_prediction"},
        {"timestep_spacing": "trailing"},
        {"beta_schedule": "squaredcos_cap_v2"},
    ],
)
def test_unsupported_settings(override):
    with pytest.raises(ValueError):
        PNDMScheduler(SD1_SCHEDULER | override)


def test_step_requires_timesteps():
    with pytest.raises(RuntimeError, match="set_timesteps"):
        PNDMScheduler(SD1_SCHEDULER).step(torch.zeros(1), 981, torch.zeros(1))
    with pytest.raises(ValueError):
        PNDMScheduler(SD1_SCHEDULER).set_timesteps(0)
