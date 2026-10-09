import json
from pathlib import Path

import pytest
import torch
from helpers import meta_shapes

from imagehub.models.uvit_2d import UVit2DModel

FIXTURES = Path(__file__).parent / "fixtures" / "amused_configs"
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


def _fixture(size: str, name: str):
    return json.loads((FIXTURES / size / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("size", ["256", "512"])
def test_matches_published_tensor_layout(size):
    config = _fixture(size, "transformer_config.json")
    assert meta_shapes(UVit2DModel, config) == _fixture(size, "transformer_tensors.json")


def _inputs(batch, height, width, down=False):
    ids = torch.randint(0, 24, (batch, height, width))
    text = torch.randn(batch, 5, TINY["encoder_hidden_size"])
    pooled = torch.randn(batch, TINY["cond_embed_dim"])
    micro = torch.tensor([[64.0, 64.0, 0.0, 0.0, 6.0]]).expand(batch, -1)
    return ids, text, pooled, micro


@pytest.mark.parametrize("extra", [{}, {"downsample": True, "upsample": True}, {"use_bias": True}])
def test_tiny_forward_shape(extra):
    torch.manual_seed(0)
    model = UVit2DModel(TINY | extra).eval()
    ids, text, pooled, micro = _inputs(2, 4, 4)
    with torch.inference_mode():
        logits = model(ids, text, pooled, micro)
    assert logits.shape == (2, 20, 4, 4)
    assert torch.isfinite(logits).all()


def test_conditioning_changes_the_logits():
    torch.manual_seed(0)
    model = UVit2DModel(TINY).eval()
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.2)
    ids, text, pooled, micro = _inputs(1, 4, 4)
    with torch.inference_mode():
        base = model(ids, text, pooled, micro)
        assert not torch.equal(base, model(ids, text + 1, pooled, micro))
        assert not torch.equal(base, model(ids, text, pooled + 1, micro))
        assert not torch.equal(base, model(ids, text, pooled, micro + 1))


def test_unknown_option_rejected():
    with pytest.raises(ValueError, match="Unsupported UVit2D setting"):
        UVit2DModel(TINY | {"bogus": 1})
