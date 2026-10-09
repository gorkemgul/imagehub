import pytest
import torch

from imagehub.schedulers.amused import AmusedScheduler, mask_by_random_topk


def test_timesteps_and_temperatures():
    scheduler = AmusedScheduler({"mask_token_id": 9})
    scheduler.set_timesteps(4)
    assert scheduler.timesteps.tolist() == [3, 2, 1, 0]
    expected = torch.tensor([2.0, 4 / 3, 2 / 3, 0.0])
    assert torch.allclose(scheduler.temperatures, expected)
    scheduler.set_timesteps(3, 5)
    assert scheduler.temperatures[0] == 5 and abs(float(scheduler.temperatures[-1]) - 0.01) < 1e-7


def test_denoising_unmasks_every_token_and_keeps_known_tokens():
    scheduler = AmusedScheduler({"mask_token_id": 9})
    scheduler.set_timesteps(5)
    generator = torch.Generator().manual_seed(0)
    sample = torch.full((2, 3, 3), 9)
    sample[0, 0, 0] = 4
    for timestep in scheduler.timesteps:
        logits = torch.randn(2, 9, 3, 3, generator=generator)
        out = scheduler.step(logits, timestep, sample, generator=generator)
        assert out.prev_sample[0, 0, 0] == 4
        sample = out.prev_sample
    assert not (sample == 9).any()


def test_mask_count_follows_the_cosine_schedule():
    scheduler = AmusedScheduler({"mask_token_id": 9})
    scheduler.set_timesteps(4)
    generator = torch.Generator().manual_seed(1)
    sample = torch.full((1, 4, 4), 9)
    remaining = []
    for timestep in scheduler.timesteps:
        out = scheduler.step(
            torch.randn(1, 9, 4, 4, generator=generator), timestep, sample, generator=generator
        )
        sample = out.prev_sample
        remaining.append(int((sample == 9).sum()))
    assert remaining == [14, 11, 6, 0]


def test_mask_by_random_topk_masks_the_lowest_confidence():
    probs = torch.tensor([[0.9, 0.1, 0.5, 0.2]])
    masking = mask_by_random_topk(torch.tensor([[2]]), probs, 0.0)
    assert masking.tolist() == [[False, True, False, True]]


def test_rejects_bad_config():
    with pytest.raises(ValueError, match="mask_token_id"):
        AmusedScheduler({})
    with pytest.raises(ValueError, match="masking_schedule"):
        AmusedScheduler({"mask_token_id": 1, "masking_schedule": "square"})
    with pytest.raises(ValueError, match="Unsupported aMUSEd scheduler setting"):
        AmusedScheduler({"mask_token_id": 1, "other": 1})
