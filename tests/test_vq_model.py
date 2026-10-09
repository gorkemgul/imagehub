import json
from pathlib import Path

import pytest
import torch
from helpers import TINY_AMUSED_VQVAE, meta_shapes

from imagehub.models.vq_model import VQModel

FIXTURES = Path(__file__).parent / "fixtures" / "amused_configs"


@pytest.mark.parametrize("size", ["256", "512"])
def test_amused_vqvae_matches_published_tensor_layout(size):
    config = json.loads((FIXTURES / size / "vqvae_config.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURES / size / "vqvae_tensors.json").read_text(encoding="utf-8"))
    assert meta_shapes(VQModel, config) == expected


def test_group_norm_codebook_decode_shape():
    torch.manual_seed(0)
    model = VQModel(TINY_AMUSED_VQVAE).eval()
    ids = torch.randint(0, 20, (2, 2, 3))
    with torch.inference_mode():
        image = model.decode(ids, force_not_quantize=True, shape=(2, 2, 3, 8))
    assert image.shape == (2, 3, 32, 48)


def test_codebook_lookup_returns_embeddings():
    model = VQModel(TINY_AMUSED_VQVAE)
    ids = torch.tensor([[[3, 7]]])
    entries = model.quantize.get_codebook_entry(ids, (1, 1, 2, 8))
    assert entries.shape == (1, 8, 1, 2)
    assert torch.equal(entries[0, :, 0, 1], model.quantize.embedding.weight[7])


def test_unsupported_norm_type_is_rejected():
    with pytest.raises(ValueError, match="norm_type"):
        VQModel(TINY_AMUSED_VQVAE | {"norm_type": "layer"})
