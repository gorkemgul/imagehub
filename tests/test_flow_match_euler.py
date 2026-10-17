import json
from pathlib import Path

import pytest
import torch

from imagehub.schedulers.flow_match_euler import (
    FlowMatchEulerDiscreteScheduler,
    _beta_ppf,
    qwen_image_mu,
    z_image_sigmas,
)

CONFIG = {"num_train_timesteps": 1000, "shift": 3.0, "use_dynamic_shifting": False}


def test_z_image_sigmas_use_float32_linspace():
    assert z_image_sigmas(1) == [1.0]
    assert z_image_sigmas(8) == torch.linspace(1.0, 1 / 8, 8).tolist()


def test_scheduler_shapes_and_euler_step():
    scheduler = FlowMatchEulerDiscreteScheduler(CONFIG)
    scheduler.set_timesteps(2, sigmas=z_image_sigmas(2), mu=0.9)
    assert scheduler.timesteps.dtype == torch.float32
    assert scheduler.sigmas.dtype == torch.float32
    assert scheduler.sigmas.shape == (3,)
    assert scheduler.sigmas[-1].item() == 0.0
    scheduler.set_begin_index(0)
    sample = torch.ones(1, dtype=torch.float16)
    model_output = torch.full((1,), 2.0, dtype=torch.float16)
    expected = (sample.float() + (scheduler.sigmas[1] - scheduler.sigmas[0]) * 2).half()
    assert torch.equal(scheduler.step(model_output, scheduler.timesteps[0], sample), expected)


def test_qwen_image_mu_uses_pipeline_latent_sequence_length():
    fixture = (
        Path(__file__).parent / "fixtures/qwen_image21_configs/scheduler_scheduler_config.json"
    )
    config = json.loads(fixture.read_text(encoding="utf-8"))
    assert qwen_image_mu(1024, 1024, config) == pytest.approx(0.6935483870967742)
    assert qwen_image_mu(1024, 2048, config) == 0.9


def test_dynamic_shifting_requires_mu_and_stretches_to_terminal():
    config = {
        "num_train_timesteps": 1000,
        "use_dynamic_shifting": True,
        "shift_terminal": 0.02,
    }
    scheduler = FlowMatchEulerDiscreteScheduler(config)
    with pytest.raises(ValueError, match="mu"):
        scheduler.set_timesteps(4, sigmas=[1.0, 0.75, 0.5, 0.25])
    scheduler.set_timesteps(4, sigmas=[1.0, 0.75, 0.5, 0.25], mu=0.7)
    assert scheduler.sigmas[-2].item() == pytest.approx(0.02, abs=1e-6)


def test_beta_ppf_matches_scipy_float64():
    np = pytest.importorskip("numpy")
    scipy = pytest.importorskip("scipy.stats")
    probabilities = 1 - np.linspace(0, 1, 40)
    expected = scipy.beta.ppf(probabilities, 0.6, 0.6).tolist()
    assert [_beta_ppf(float(value)) for value in probabilities] == expected


def test_beta_sigma_schedule():
    scheduler = FlowMatchEulerDiscreteScheduler(
        {"num_train_timesteps": 1000, "use_beta_sigmas": True}
    )
    scheduler.set_timesteps(4)
    assert scheduler.sigmas.tolist() == pytest.approx(
        [1.0, 0.7248620986938477, 0.2761378884315491, 0.0010000000474974513, 0.0],
        abs=0,
        rel=0,
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("use_karras_sigmas", True),
        ("use_exponential_sigmas", True),
        ("stochastic_sampling", True),
        ("time_shift_type", "cosine"),
        ("invert_sigmas", True),
    ],
)
def test_rejects_unsupported_options(key, value):
    with pytest.raises(ValueError, match=key):
        FlowMatchEulerDiscreteScheduler(CONFIG | {key: value})


def test_inverted_sigmas_need_an_explicit_opt_in():
    config = CONFIG | {"invert_sigmas": True, "num_train_timesteps": 1, "shift": 1.0}
    scheduler = FlowMatchEulerDiscreteScheduler(config, allow_inverted_sigmas=True)
    scheduler.set_timesteps(4, sigmas=[1.0, 0.75, 0.5, 0.25])
    assert scheduler.timesteps.tolist() == [0.0, 0.25, 0.5, 0.75]
    assert scheduler.sigmas.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    sample = torch.zeros(2)
    out = scheduler.step(torch.ones(2), scheduler.timesteps[0], sample)
    assert torch.equal(out, torch.full((2,), 0.25))
