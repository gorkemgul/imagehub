import json
from pathlib import Path

import pytest
import torch

from imagehub.exceptions import CheckpointError
from imagehub.models.infinity import InfinityTransformer
from imagehub.models.infinity_vae import InfinityVAE

FIXTURES = Path(__file__).parent / "fixtures" / "infinity_configs"
TINY_TRANSFORMER = {
    "depth": 2,
    "embed_dim": 64,
    "num_heads": 4,
    "mlp_ratio": 4,
    "text_channels": 32,
    "text_max_length": 8,
    "bit_channels": 4,
    "block_chunks": 1,
    "level_count": 4,
}
TINY_VAE = {
    "base_channels": 32,
    "channel_multipliers": (1, 1),
    "num_res_blocks": 1,
    "latent_channels": 4,
    "patch_size": 2,
}


def _published(name):
    return json.loads((FIXTURES / f"{name}_tensors.json").read_text())["tensors"]


def test_published_transformer_layout():
    published = _published("transformer")
    with torch.device("meta"):
        model = InfinityTransformer()
    actual = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
        for name, value in model.state_dict().items()
    }
    assert actual == published


def test_published_vae_layout_after_training_only_tensors_are_removed():
    published = _published("vae")
    with torch.device("meta"):
        model = InfinityVAE()
    state = {name: torch.empty(value["shape"], device="meta") for name, value in published.items()}
    prepared = model.prepare_state_dict(state)
    assert set(prepared) == set(model.state_dict())
    assert all(prepared[name].shape == value.shape for name, value in model.state_dict().items())
    assert len(published) == 302 and len(prepared) == 268


def _condition(channels=32):
    compact = torch.randn(5, channels)
    return compact, [5], torch.tensor([0, 5], dtype=torch.int32), 5


def test_tiny_transformer_runs_two_cached_scales():
    torch.manual_seed(0)
    model = InfinityTransformer(TINY_TRANSFORMER).eval()
    schedule = ((1, 1, 1), (1, 2, 2))
    stage, adaptive, pooled, condition = model.begin_generation(_condition(), schedule, True)
    with torch.no_grad():
        stage, first = model.stage_logits(stage, adaptive, pooled, condition, schedule, 0)
        assert first.shape == (2, 1, 8)
        next_stage = model.next_stage(torch.randn(1, 4, 4), 2)
        _, second = model.stage_logits(next_stage, adaptive, pooled, condition, schedule, 1)
    model.end_generation()
    assert second.shape == (2, 4, 8)
    assert all(block.sa.cached_k is None for block in model.blocks_in_order())


def test_tiny_vae_bit_codes_and_decode_shapes():
    model = InfinityVAE(TINY_VAE).eval()
    bits = torch.randint(0, 2, (2, 1, 3, 4, 4))
    codes = model.bits_to_codes(bits)
    assert codes.shape == (2, 4, 1, 3, 4)
    assert torch.equal(codes.square().sum(1), torch.ones(2, 1, 3, 4))
    with torch.no_grad():
        image = model.decode(codes.squeeze(2))
    assert image.shape == (2, 3, 6, 8)
    assert image.min() >= -1 and image.max() <= 1


def test_strict_loading_rejects_a_missing_tensor():
    model = InfinityTransformer(TINY_TRANSFORMER)
    state = model.state_dict()
    state.pop("head.weight")
    with pytest.raises(CheckpointError, match="head.weight"):
        InfinityTransformer(TINY_TRANSFORMER).load_checkpoint(state)
