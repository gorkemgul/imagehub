import pytest
import torch

from imagehub.schedulers.scm import SCMScheduler

CONFIG = {
    "_class_name": "SCMScheduler",
    "num_train_timesteps": 1000,
    "prediction_type": "trigflow",
    "sigma_data": 0.5,
}


def test_release_config_and_state_reset():
    scheduler = SCMScheduler(CONFIG)
    scheduler.set_timesteps(2)
    assert scheduler.timesteps.dtype == torch.float32
    assert len(scheduler.timesteps) == 3
    sample = torch.randn(1, 4, 3, 3)
    assert scheduler.scale_model_input(sample, scheduler.timesteps[0]) is sample
    scheduler.step(torch.randn_like(sample), scheduler.timesteps[0], sample)
    assert scheduler.step_index == 1
    scheduler.set_timesteps(2)
    assert scheduler.step_index is None


def test_requires_set_timesteps_before_sampling():
    scheduler = SCMScheduler(CONFIG)
    sample = torch.randn(1, 4, 3, 3)
    with pytest.raises(RuntimeError, match="set_timesteps"):
        scheduler.step(sample, torch.tensor(1.0), sample)


@pytest.mark.parametrize(
    "option",
    [
        {"prediction_type": "epsilon"},
        {"sigma_data": 0.0},
        {"sigma_data": -0.5},
        {"unknown_option": True},
    ],
)
def test_rejects_unsupported_options(option):
    with pytest.raises(ValueError, match="Unsupported SCM"):
        SCMScheduler(CONFIG | option)


def test_intermediate_timesteps_only_for_two_steps():
    scheduler = SCMScheduler(CONFIG)
    scheduler.set_timesteps(2, intermediate_timesteps=1.3)
    assert torch.allclose(scheduler.timesteps, torch.tensor([1.57080, 1.30000, 0.00000]), atol=1e-5)
    scheduler = SCMScheduler(CONFIG)
    scheduler.set_timesteps(4)
    assert len(scheduler.timesteps) == 5
    with pytest.raises(ValueError, match="Intermediate timesteps"):
        scheduler.set_timesteps(4, intermediate_timesteps=1.3)


def test_trigflow_step_matches_reference():
    diffusers = pytest.importorskip("diffusers")
    ref = diffusers.SCMScheduler.from_config(
        {"num_train_timesteps": 1000, "prediction_type": "trigflow", "sigma_data": 0.5}
    )
    ref.set_timesteps(2, max_timesteps=1.57080, intermediate_timesteps=1.3, device="cpu")
    ours = SCMScheduler(CONFIG)
    ours.set_timesteps(2, max_timesteps=1.57080, intermediate_timesteps=1.3)

    sample = torch.randn(2, 4, 8, 8)
    model_output = torch.randn(2, 4, 8, 8)

    ref_prev, ref_pred = ref.step(
        model_output,
        ref.timesteps[0],
        sample,
        generator=torch.Generator("cpu").manual_seed(0),
        return_dict=False,
    )
    our_prev, our_pred = ours.step(
        model_output,
        ours.timesteps[0],
        sample,
        generator=torch.Generator("cpu").manual_seed(0),
    )
    assert torch.equal(ref_prev, our_prev)
    assert torch.equal(ref_pred, our_pred)
