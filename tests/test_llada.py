import json
from pathlib import Path

import pytest
import torch

from imagehub.models.llada import LLaDAForMultiModalGeneration
from imagehub.models.vq_model import VQModel

FIXTURES = Path(__file__).parent / "fixtures" / "lumina_dimoo_configs"
TINY = {
    "d_model": 32,
    "embedding_size": 128,
    "vocab_size": 128,
    "n_heads": 4,
    "n_kv_heads": 4,
    "n_layers": 2,
    "mlp_hidden_size": 48,
    "max_sequence_length": 64,
}


def test_matches_published_tensor_layout():
    config = json.loads((FIXTURES / "model/config.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURES / "model/header.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = LLaDAForMultiModalGeneration(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == expected
    assert len(actual) == 291


def test_vqvae_matches_published_tensor_layout():
    config = json.loads((FIXTURES / "vqvae/config.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURES / "vqvae/header.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = VQModel(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == expected
    assert len(actual) == 277


def test_tiny_bidirectional_forward_is_repeatable():
    torch.manual_seed(0)
    model = LLaDAForMultiModalGeneration(TINY).eval()
    ids = torch.randint(0, 128, (2, 11))
    changed = ids.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 128
    with torch.inference_mode():
        first = model(ids)
        again = model(ids)
        other = model(changed)
    assert first.shape == (2, 11, 128)
    assert torch.equal(first, again)
    assert not torch.equal(first[:, 0], other[:, 0])


@pytest.mark.parametrize(
    "override",
    [
        {"block_type": "sequential"},
        {"activation_type": "gelu"},
        {"n_heads": 3},
        {"bogus": True},
    ],
)
def test_unsupported_settings_are_rejected(override):
    with pytest.raises(ValueError, match="Unsupported LLaDA"):
        LLaDAForMultiModalGeneration(TINY | override)
