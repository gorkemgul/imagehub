"""DeepFloyd IF U-Net and DDPM random-weight parity against diffusers."""

import json

import pytest
import torch
from fixture_files import fixture_path

from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.ddpm import DDPMScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

FIXTURES = fixture_path("deepfloyd_if_configs")


def _published_small(stage: str) -> dict:
    config = json.loads((FIXTURES / stage / "unet_config.json").read_text())
    config = {key: value for key, value in config.items() if not key.startswith("_")}
    if stage == "stage_i":
        return config | {
            "sample_size": 16,
            "block_out_channels": [32, 64, 96, 128],
            "attention_head_dim": 8,
            "cross_attention_dim": 128,
            "encoder_hid_dim": 64,
            "norm_num_groups": 32,
            "addition_embed_type_num_heads": 4,
        }
    return config | {
        "sample_size": 16,
        "block_out_channels": [32, 32, 64, 96, 128],
        "attention_head_dim": 8,
        "cross_attention_dim": 128,
        "encoder_hid_dim": 64,
        "norm_num_groups": 32,
        "addition_embed_type_num_heads": 4,
    }


def _tiny(stage: str) -> dict:
    config = _published_small(stage)
    if stage == "stage_i":
        return config | {
            "down_block_types": ["ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"],
            "up_block_types": ["SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"],
            "block_out_channels": [32, 64],
            "layers_per_block": 1,
            "cross_attention_dim": 64,
            "encoder_hid_dim": 48,
        }
    return config | {
        "down_block_types": ["ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"],
        "up_block_types": ["SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"],
        "block_out_channels": [32, 64],
        "layers_per_block": [1, 1],
        "cross_attention_dim": 64,
        "encoder_hid_dim": 48,
        "time_embedding_dim": 64,
    }


def _random_pair(config: dict):
    reference = diffusers.UNet2DConditionModel(**config).eval()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    ours = UNet2DConditionModel(config).eval()
    ours.load_checkpoint(reference.state_dict())
    return reference, ours


@pytest.mark.parametrize("stage", ("stage_i", "stage_ii"))
@pytest.mark.parametrize("kind", ("tiny", "published_small"))
def test_if_unet_random_weights_match_reference(stage: str, kind: str, record) -> None:
    config = _tiny(stage) if kind == "tiny" else _published_small(stage)
    reference, ours = _random_pair(config)
    generator = torch.Generator().manual_seed(1)
    size = 32 if stage == "stage_ii" and kind == "published_small" else 8
    sample = torch.randn(2, config["in_channels"], size, size, generator=generator)
    context = torch.randn(2, 7, config["encoder_hid_dim"], generator=generator)
    labels = torch.tensor([15, 250]) if stage == "stage_ii" else None
    with torch.inference_mode():
        expected = reference(
            sample, 713, encoder_hidden_states=context, class_labels=labels, return_dict=False
        )[0]
        actual = ours(sample, torch.tensor(713), context, class_labels=labels)
    assert torch.equal(actual, expected)
    record(f"deepfloyd_if_{stage}_{kind}", exact=True, shape=list(actual.shape))


@pytest.mark.parametrize("stage", ("stage_i", "stage_ii"))
def test_if_ddpm_dynamic_threshold_learned_range_matches_reference(stage: str, record) -> None:
    config = json.loads((FIXTURES / stage / "scheduler_config.json").read_text())
    reference = diffusers.DDPMScheduler.from_config(config)
    ours = DDPMScheduler(config)
    reference.set_timesteps(4)
    ours.set_timesteps(4)
    generator = torch.Generator().manual_seed(2)
    actual = expected = torch.randn(2, 3, 6, 10, generator=generator) * 4
    outputs = [torch.randn(2, 6, 6, 10, generator=generator) * 3 for _ in ours.timesteps]
    expected_generator = torch.Generator().manual_seed(3)
    actual_generator = torch.Generator().manual_seed(3)
    for timestep, output in zip(ours.timesteps, outputs, strict=True):
        expected = reference.step(
            output, timestep, expected, generator=expected_generator
        ).prev_sample
        actual = ours.step(output, timestep, actual, generator=actual_generator)
        assert torch.equal(actual, expected)
    record(f"deepfloyd_if_{stage}_ddpm", exact=True)


def test_if_stage_ii_image_noising_matches_reference(record) -> None:
    config = json.loads((FIXTURES / "stage_ii" / "image_noising_scheduler_config.json").read_text())
    reference = diffusers.DDPMScheduler.from_config(config)
    ours = DDPMScheduler(config)
    generator = torch.Generator().manual_seed(4)
    image = torch.randn(2, 3, 7, 9, generator=generator)
    noise = torch.randn(image.shape, generator=generator)
    timesteps = torch.tensor([0, 250])
    expected = reference.add_noise(image, noise, timesteps)
    actual = ours.add_noise(image, noise, timesteps)
    assert torch.equal(actual, expected)
    record("deepfloyd_if_stage_ii_image_noise", exact=True)
