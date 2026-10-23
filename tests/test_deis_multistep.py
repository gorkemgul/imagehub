import pytest
import torch

from imagehub.schedulers.deis_multistep import DEISMultistepScheduler

CONFIG = {
    "algorithm_type": "deis",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "solver_order": 2,
    "solver_type": "logrho",
    "steps_offset": 1,
    "thresholding": False,
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_karras_sigmas": False,
}


def test_schedule_and_trajectory_are_repeatable():
    schedulers = [DEISMultistepScheduler(CONFIG), DEISMultistepScheduler(CONFIG)]
    for scheduler in schedulers:
        scheduler.set_timesteps(4)
    assert torch.equal(schedulers[0].timesteps, torch.tensor([999, 749, 500, 250]))
    generator = torch.Generator().manual_seed(0)
    outputs = [torch.randn(1, 4, 3, 5, generator=generator) for _ in range(4)]
    samples = [torch.randn(1, 4, 3, 5, generator=torch.Generator().manual_seed(1))] * 2
    for timestep, model_output in zip(schedulers[0].timesteps, outputs, strict=True):
        samples = [
            scheduler.step(model_output, timestep, sample)
            for scheduler, sample in zip(schedulers, samples, strict=True)
        ]
    assert torch.equal(samples[0], samples[1])


def test_leading_schedule_uses_step_offset():
    scheduler = DEISMultistepScheduler(CONFIG | {"timestep_spacing": "leading"})
    scheduler.set_timesteps(4)
    assert torch.equal(scheduler.timesteps, torch.tensor([801, 601, 401, 201]))


def test_requires_set_timesteps_before_sampling():
    scheduler = DEISMultistepScheduler(CONFIG)
    sample = torch.randn(1, 4, 3, 3)
    with pytest.raises(RuntimeError, match="set_timesteps"):
        scheduler.step(sample, torch.tensor(1), sample)


@pytest.mark.parametrize(
    "option",
    [
        {"solver_order": 3},
        {"prediction_type": "v_prediction"},
        {"solver_type": "midpoint"},
        {"thresholding": True},
        {"timestep_spacing": "trailing"},
        {"use_karras_sigmas": True},
        {"unknown_option": True},
    ],
)
def test_rejects_unsupported_options(option):
    with pytest.raises(ValueError, match="Unsupported DEIS setting"):
        DEISMultistepScheduler(CONFIG | option)
