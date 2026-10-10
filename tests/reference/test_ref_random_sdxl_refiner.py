"""Random-weight parity for the SDXL refiner's U-Net (five time ids) and image-to-image sampler."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.euler import EulerDiscreteScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

FIXTURES = Path(__file__).parents[1] / "fixtures"
PUBLISHED_UNET = json.loads((FIXTURES / "sdxl_refiner_configs/unet/config.json").read_text())
SCHEDULER = json.loads(
    (FIXTURES / "sdxl_refiner_configs/scheduler/scheduler_config.json").read_text()
)
TINY_UNET = PUBLISHED_UNET | {
    "block_out_channels": [32, 64, 64, 64],
    "attention_head_dim": [2, 4, 4, 4],
    "transformer_layers_per_block": 1,
    "cross_attention_dim": 32,
    "addition_time_embed_dim": 8,
    "projection_class_embeddings_input_dim": 72,
    "layers_per_block": 1,
    "norm_num_groups": 8,
}


@pytest.mark.parametrize(
    ("config", "sizes"),
    [(TINY_UNET, [(16, 16), (13, 9)]), (PUBLISHED_UNET, [(16, 16), (13, 9)])],
    ids=["tiny", "published"],
)
def test_refiner_unet_random_weights(config, sizes):
    torch.manual_seed(0)
    reference = diffusers.UNet2DConditionModel.from_config(config).eval()
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.02)
    ours = UNet2DConditionModel(config).eval()
    ours.load_checkpoint(reference.state_dict())
    context = torch.randn(2, 7, config["cross_attention_dim"])
    text_dim = (
        config["projection_class_embeddings_input_dim"] - 5 * config["addition_time_embed_dim"]
    )
    text_embeds = torch.randn(2, text_dim)
    time_ids = torch.randn(2, 5)
    for size in sizes:
        sample = torch.randn(2, 4, *size)
        for timestep in (torch.tensor(981), torch.tensor(1)):
            with torch.inference_mode():
                actual = ours(sample, timestep, context, text_embeds=text_embeds, time_ids=time_ids)
                expected = reference(
                    sample,
                    timestep,
                    context,
                    added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                ).sample
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("steps", "strength"), [(50, 0.3), (50, 1.0), (30, 0.5), (7, 0.7), (4, 0.25)]
)
def test_euler_image_to_image_trajectory_matches_diffusers(steps, strength):
    expected = diffusers.EulerDiscreteScheduler.from_config(SCHEDULER)
    ours = EulerDiscreteScheduler(SCHEDULER)
    expected.set_timesteps(steps)
    ours.set_timesteps(steps)
    init_timestep = min(int(steps * strength), steps)
    start = max(steps - init_timestep, 0)
    timesteps = expected.timesteps[start:]
    expected.set_begin_index(start)
    ours.set_begin_index(start)
    assert torch.equal(ours.timesteps[start:], timesteps)
    generator = torch.Generator().manual_seed(0)
    original = torch.randn(2, 4, 8, 8, generator=generator)
    noise = torch.randn(2, 4, 8, 8, generator=generator)
    batch = timesteps[:1].repeat(2)
    expected_latents = expected.add_noise(original, noise, batch)
    latents = ours.add_noise(original, noise, batch)
    torch.testing.assert_close(latents, expected_latents, rtol=0, atol=0)
    for timestep in timesteps:
        model_output = torch.randn(2, 4, 8, 8, generator=generator)
        scaled = expected.scale_model_input(expected_latents, timestep)
        torch.testing.assert_close(
            ours.scale_model_input(latents, timestep), scaled, rtol=0, atol=0
        )
        expected_latents = expected.step(model_output, timestep, expected_latents).prev_sample
        latents = ours.step(model_output, timestep, latents)
        torch.testing.assert_close(latents, expected_latents, rtol=0, atol=0)
