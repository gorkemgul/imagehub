import pytest
import torch

from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler, z_image_sigmas

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


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("use_dynamic_shifting", True),
        ("use_karras_sigmas", True),
        ("use_exponential_sigmas", True),
        ("use_beta_sigmas", True),
        ("stochastic_sampling", True),
        ("time_shift_type", "linear"),
        ("invert_sigmas", True),
        ("shift_terminal", 0.1),
    ],
)
def test_rejects_unsupported_options(key, value):
    with pytest.raises(ValueError, match=key):
        FlowMatchEulerDiscreteScheduler(CONFIG | {key: value})
