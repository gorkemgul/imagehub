from __future__ import annotations

import pytest
import torch

from imagehub.backends.emu3 import Emu3Backend, generate_tokens
from imagehub.models.emu3 import Emu3ForCausalLM
from imagehub.models.emu3_vq import Emu3VisionVQ
from imagehub.types import T2IRequest

GENERATOR_CONFIG = {
    "vocab_size": 48,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "attention_dropout": 0.0,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "hidden_act": "silu",
    "rope_scaling": None,
}

VQ_CONFIG = {
    "ch": 32,
    "ch_mult": [1, 1],
    "num_res_blocks": 1,
    "attn_resolutions": [1],
    "dropout": 0.0,
    "in_channels": 3,
    "out_channels": 3,
    "z_channels": 4,
    "embed_dim": 4,
    "codebook_size": 16,
    "double_z": False,
    "temporal_downsample_factor": 1,
}


def test_generator_cache_matches_full_sequence():
    torch.manual_seed(0)
    model = Emu3ForCausalLM(GENERATOR_CONFIG).eval()
    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[5]])
    with torch.no_grad():
        _, cache = model(first)
        cached, _ = model(second, cache)
        full, _ = model(torch.cat((first, second), 1))
    assert torch.allclose(cached[:, -1], full[:, -1], atol=1e-6, rtol=0)


def test_vq_decode_shape_and_repeatability():
    torch.manual_seed(1)
    model = Emu3VisionVQ(VQ_CONFIG).eval()
    tokens = torch.tensor([[[1, 2], [3, 4]]])
    with torch.no_grad():
        first = model.decode(tokens)
        second = model.decode(tokens)
    assert first.shape == (1, 3, 4, 4)
    assert torch.equal(first, second)


class _Tokenizer:
    visual_start = 8
    visual_end = 11
    eol_id = 4
    eof_id = 5
    eoi_id = 6


class _PredictableModel:
    def __call__(self, ids, cache=None):
        logits = torch.full((1, ids.shape[1], 12), -100.0)
        logits[..., 9] = 10.0
        length = ids.shape[1] if cache is None else cache[0][0].shape[2] + ids.shape[1]
        present = [(torch.zeros(1, 1, length, 1), torch.zeros(1, 1, length, 1))]
        return logits, present


def test_constrained_generation_builds_grid():
    tokens = generate_tokens(
        _PredictableModel(),  # type: ignore[arg-type]
        _Tokenizer(),  # type: ignore[arg-type]
        torch.tensor([[1, 2, 3]]),
        None,
        height=2,
        width=3,
        guidance=1.0,
        temperature=1.0,
        top_k=1,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(tokens, torch.ones(1, 2, 3, dtype=torch.long))


def test_options_are_validated_before_loading(tmp_path):
    backend = Emu3Backend(tmp_path)
    request = T2IRequest("an emu")
    with pytest.raises(TypeError, match="Unknown Emu3 options"):
        backend.infer(request, typo=True)
    with pytest.raises(ValueError, match="multiple of 8"):
        backend.infer(request, width=10)
    assert not backend.is_loaded


def test_options_accept_tiny_generation(tmp_path):
    backend = Emu3Backend(tmp_path)
    values = backend.validate_options(
        T2IRequest("an emu"),
        {"width": 16, "height": 24, "seed": 0, "guidance": 1.0, "top_k": 1},
    )
    assert values["width"] == 16
    assert values["height"] == 24
