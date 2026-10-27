import json

import torch
from fixture_files import fixture_path

from imagehub.models.paella_vq import PaellaVQModel

FIXTURES = fixture_path("stable_cascade_configs")
TINY = {
    "levels": 2,
    "bottleneck_blocks": 2,
    "embed_dim": 16,
    "latent_channels": 4,
    "num_vq_embeddings": 8,
}


def test_published_layout_matches_tensor_list():
    config = json.loads((FIXTURES / "vqgan_config.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURES / "vqgan_tensors.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = PaellaVQModel(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == expected
    assert len(actual) == 122
    assert model.scale_factor == 0.3764


def test_tiny_encode_decode_shapes():
    model = PaellaVQModel(TINY).eval()
    with torch.inference_mode():
        latents = model.encode(torch.randn(1, 3, 32, 32))
        assert latents.shape == (1, 4, 8, 8)
        assert model.decode(latents).shape == (1, 3, 32, 32)
