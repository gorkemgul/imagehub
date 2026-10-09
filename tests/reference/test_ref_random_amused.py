"""aMUSEd U-ViT, scheduler and decoder exact parity with random weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.uvit_2d import UVit2DModel
from imagehub.models.vq_model import VQModel
from imagehub.schedulers.amused import AmusedScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

FIXTURES = Path(__file__).parents[1] / "fixtures" / "amused_configs"
TINY = {
    "hidden_size": 32,
    "cond_embed_dim": 12,
    "micro_cond_encode_dim": 8,
    "micro_cond_embed_dim": 40,
    "encoder_hidden_size": 16,
    "vocab_size": 24,
    "codebook_size": 20,
    "in_channels": 16,
    "block_out_channels": 16,
    "num_res_blocks": 2,
    "block_num_heads": 2,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 48,
    "sample_size": 4,
}


def _config(size, name):
    config = json.loads((FIXTURES / size / name).read_text(encoding="utf-8"))
    return {k: v for k, v in config.items() if not k.startswith("_")}


def _compare_uvit(config, batch, height, width, text_len, record, name):
    torch.manual_seed(0)
    reference = diffusers.UVit2DModel(**config).eval()
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.1)
    model = UVit2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())
    ids = torch.randint(0, config["vocab_size"], (batch, height, width))
    text = torch.randn(batch, text_len, config["encoder_hidden_size"])
    pooled = torch.randn(batch, config["cond_embed_dim"])
    micro = torch.tensor([[256.0, 256.0, 0.0, 0.0, 6.0]]).expand(batch, -1)
    with torch.inference_mode():
        expected = reference(ids, text, pooled, micro)
        actual = model(ids, text, pooled, micro)
    record(name, max_abs_diff=float((actual - expected).abs().max()))
    assert torch.equal(actual, expected)


def test_uvit_tiny_exact(record):
    _compare_uvit(TINY, 2, 4, 4, 5, record, "amused_uvit_tiny")


def test_uvit_tiny_bias_resampling_exact(record):
    config = TINY | {"use_bias": True, "downsample": True, "upsample": True}
    _compare_uvit(config, 2, 4, 6, 5, record, "amused_uvit_tiny_bias_resampling")


def test_uvit_published_256_config_exact(record):
    _compare_uvit(
        _config("256", "transformer_config.json"), 2, 16, 16, 77, record, "amused_uvit_256"
    )


def _scheduler_pair(**config):
    base = {"mask_token_id": 20, "masking_schedule": "cosine"} | config
    return diffusers.AmusedScheduler(**base), AmusedScheduler(base)


@pytest.mark.parametrize(
    ("schedule", "temperature"),
    [("cosine", (2, 0)), ("cosine", 3), ("linear", (2, 0)), ("linear", [1.5, 0.5])],
)
def test_scheduler_trajectory_exact(schedule, temperature, record):
    reference, ours = _scheduler_pair(masking_schedule=schedule)
    steps = 7
    reference.set_timesteps(steps, temperature, "cpu")
    ours.set_timesteps(steps, temperature, "cpu")
    assert torch.equal(reference.timesteps, ours.timesteps)
    assert torch.equal(reference.temperatures, ours.temperatures)
    ref_gen = torch.Generator("cpu").manual_seed(5)
    our_gen = torch.Generator("cpu").manual_seed(5)
    logit_gen = torch.Generator("cpu").manual_seed(11)
    expected = torch.full((3, 4, 5), 20, dtype=torch.long)
    actual = expected.clone()
    for timestep in reference.timesteps:
        logits = torch.randn(3, 20, 4, 5, generator=logit_gen) * 2
        ref_out = reference.step(logits, timestep, expected, generator=ref_gen)
        our_out = ours.step(logits, timestep, actual, generator=our_gen)
        expected, actual = ref_out.prev_sample, our_out.prev_sample
        assert torch.equal(ref_out.prev_sample, our_out.prev_sample)
        assert torch.equal(ref_out.pred_original_sample, our_out.pred_original_sample)
    assert not (actual == 20).any()
    record(f"amused_scheduler_{schedule}", trajectory_equal=True)


def test_scheduler_without_generator_state_matches():
    reference, ours = _scheduler_pair()
    reference.set_timesteps(4)
    ours.set_timesteps(4)
    logits = torch.randn(2, 20, 3, 3)
    sample = torch.full((2, 3, 3), 20)
    torch.manual_seed(3)
    expected = reference.step(logits, reference.timesteps[0], sample).prev_sample
    torch.manual_seed(3)
    actual = ours.step(logits, ours.timesteps[0], sample).prev_sample
    assert torch.equal(actual, expected)


def _vq_state_pair(config):
    torch.manual_seed(0)
    reference = diffusers.VQModel(**config).eval()
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.05)
    model = VQModel(config).eval()
    model.load_checkpoint(reference.state_dict())
    return reference, model


TINY_VQ = {
    "block_out_channels": [32, 32, 64],
    "down_block_types": ["DownEncoderBlock2D"] * 3,
    "up_block_types": ["UpDecoderBlock2D"] * 3,
    "latent_channels": 8,
    "layers_per_block": 1,
    "norm_num_groups": 8,
    "norm_type": "group",
    "num_vq_embeddings": 16,
    "lookup_from_codebook": True,
    "mid_block_add_attention": False,
}


def test_vq_group_norm_codebook_decode_tiny_exact(record):
    reference, model = _vq_state_pair(TINY_VQ)
    ids = torch.randint(0, 16, (2, 5, 6))
    shape = (2, 5, 6, 8)
    with torch.inference_mode():
        expected = reference.decode(ids, force_not_quantize=True, shape=shape).sample
        actual = model.decode(ids, force_not_quantize=True, shape=shape)
    record("amused_vq_tiny", max_abs_diff=float((actual - expected).abs().max()))
    assert torch.equal(actual, expected)


def test_vq_group_norm_attention_and_quantizing_decode_exact():
    config = TINY_VQ | {"mid_block_add_attention": True, "lookup_from_codebook": False}
    reference, model = _vq_state_pair(config)
    latents = torch.randn(1, 8, 4, 4)
    with torch.inference_mode():
        expected = reference.decode(latents).sample
        actual = model.decode(latents)
    assert torch.equal(actual, expected)


def test_vq_published_config_exact(record):
    config = _config("256", "vqvae_config.json")
    reference, model = _vq_state_pair(config)
    ids = torch.randint(0, 8192, (1, 16, 16))
    shape = (1, 16, 16, config["latent_channels"])
    with torch.inference_mode():
        expected = reference.decode(ids, force_not_quantize=True, shape=shape).sample
        actual = model.decode(ids, force_not_quantize=True, shape=shape)
    record("amused_vq_256", max_abs_diff=float((actual - expected).abs().max()))
    assert torch.equal(actual, expected)
