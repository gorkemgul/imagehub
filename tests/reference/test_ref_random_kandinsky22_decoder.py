"""Random-weight parity for the Kandinsky 2.2 decoder components."""

import json
import struct
from pathlib import Path

import pytest
import torch

from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.models.vq_model import VQModel
from imagehub.schedulers.ddpm import DDPMScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")


TINY_UNET = {
    "sample_size": 8,
    "in_channels": 4,
    "out_channels": 8,
    "down_block_types": ["ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"],
    "up_block_types": ["SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"],
    "block_out_channels": [32, 64],
    "layers_per_block": 1,
    "attention_head_dim": 8,
    "cross_attention_dim": 24,
    "encoder_hid_dim": 40,
    "encoder_hid_dim_type": "image_proj",
    "addition_embed_type": "image",
    "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
    "norm_num_groups": 8,
    "resnet_time_scale_shift": "scale_shift",
}
MEDIUM_UNET = TINY_UNET | {
    "block_out_channels": [64, 96],
    "layers_per_block": 2,
}

TINY_MOVQ = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ["AttnDownEncoderBlock2D"],
    "up_block_types": ["AttnUpDecoderBlock2D"],
    "block_out_channels": [32],
    "layers_per_block": 1,
    "latent_channels": 4,
    "num_vq_embeddings": 16,
    "norm_num_groups": 32,
    "vq_embed_dim": 4,
    "norm_type": "spatial",
}
MEDIUM_MOVQ = TINY_MOVQ | {
    "down_block_types": ["DownEncoderBlock2D", "AttnDownEncoderBlock2D"],
    "up_block_types": ["AttnUpDecoderBlock2D", "UpDecoderBlock2D"],
    "block_out_channels": [32, 64],
    "num_vq_embeddings": 64,
}


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("config", [TINY_UNET, MEDIUM_UNET], ids=["tiny", "medium"])
def test_image_conditioned_unet_random_weights(config):
    torch.manual_seed(0)
    reference = diffusers.UNet2DConditionModel(**config).eval()
    ours = UNet2DConditionModel(config).eval()
    ours.load_checkpoint(reference.state_dict())
    sample = torch.randn(2, 4, 8, 12)
    timestep = torch.tensor(37)
    image_embeds = torch.randn(2, config["encoder_hid_dim"])
    with torch.inference_mode():
        expected = reference(
            sample,
            timestep,
            encoder_hidden_states=None,
            added_cond_kwargs={"image_embeds": image_embeds},
        ).sample
        actual = ours(sample, timestep, None, image_embeds=image_embeds)
    exact(actual, expected)


@pytest.mark.parametrize("config", [TINY_MOVQ, MEDIUM_MOVQ], ids=["tiny", "medium"])
def test_movq_decode_encode_random_weights(config):
    torch.manual_seed(0)
    reference = diffusers.VQModel(**config).eval()
    ours = VQModel(config).eval()
    ours.load_checkpoint(reference.state_dict())
    latents = torch.randn(2, config["vq_embed_dim"], 5, 7)
    image = torch.randn(2, 3, 10, 14)
    with torch.inference_mode():
        expected_decode = reference.decode(latents, force_not_quantize=True).sample
        actual_decode = ours.decode(latents, force_not_quantize=True)
        expected_encode = reference.encode(image).latents
        actual_encode = ours.encode(image)
    exact(actual_decode, expected_decode)
    exact(actual_encode, expected_encode)


def test_kandinsky_ddpm_100_step_trajectory():
    config = {
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "linear",
        "variance_type": "learned_range",
        "clip_sample": True,
        "clip_sample_range": 1.0,
        "prediction_type": "epsilon",
        "timestep_spacing": "leading",
        "steps_offset": 0,
    }
    reference = diffusers.DDPMScheduler(**config)
    ours = DDPMScheduler(config)
    reference.set_timesteps(100)
    ours.set_timesteps(100)
    exact(ours.timesteps, reference.timesteps)

    output_generator = torch.Generator().manual_seed(1)
    actual = torch.randn(2, 4, 5, 7, generator=output_generator)
    expected = actual.clone()
    outputs = [torch.randn(2, 8, 5, 7, generator=output_generator) for _ in ours.timesteps]
    actual_generator = torch.Generator().manual_seed(42)
    expected_generator = torch.Generator().manual_seed(42)
    for timestep, model_output in zip(ours.timesteps, outputs, strict=True):
        expected = reference.step(
            model_output, timestep, expected, generator=expected_generator
        ).prev_sample
        actual = ours.step(model_output, timestep, actual, generator=actual_generator)
        exact(actual, expected)


def test_published_unet_layout():
    root = Path("checkpoints/kandinsky-2.2/unet")
    checkpoint = root / "diffusion_pytorch_model.safetensors"
    if not checkpoint.exists():
        pytest.skip("Kandinsky 2.2 decoder U-Net checkpoint is not present")
    config = json.loads(
        Path("tests/fixtures/kandinsky22_configs/unet_config.json").read_text(encoding="utf-8")
    )
    with torch.device("meta"):
        model = UNet2DConditionModel(config)
    with checkpoint.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    published = {
        name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
    }
    actual = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    assert actual == published


def test_published_movq_layout():
    root = Path("checkpoints/kandinsky-2.2/movq")
    checkpoint = root / "diffusion_pytorch_model.safetensors"
    if not checkpoint.exists():
        pytest.skip("Kandinsky 2.2 MoVQ checkpoint is not present")
    config = json.loads(
        Path("tests/fixtures/kandinsky22_configs/movq_config.json").read_text(encoding="utf-8")
    )
    with torch.device("meta"):
        model = VQModel(config)
    with checkpoint.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    published = {
        name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
    }
    actual = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    assert actual == published
