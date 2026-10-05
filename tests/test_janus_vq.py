import json
from pathlib import Path

import pytest
import torch

from imagehub.models.janus_vq import JanusGenerationModel, postprocess_janus

FIXTURES = Path(__file__).parent / "fixtures"
VQ_CONFIG = json.loads((FIXTURES / "janus_pro_1b_config.json").read_text())["vq_config"]


def test_published_generation_layout():
    fixture = json.loads(
        (FIXTURES / "janus_pro_1b_generation_tensors.json").read_text(encoding="utf-8")
    )
    with torch.device("meta"):
        model = JanusGenerationModel(VQ_CONFIG)
    actual = {"model." + name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == fixture["tensors"]
    assert fixture["excluded_prefixes"] == [
        "model.vision_model.",
        "model.aligner.",
        "lm_head.",
        "model.vqmodel.encoder.",
        "model.vqmodel.quant_conv.",
    ]


def test_generation_shapes_and_image_token_count():
    config = VQ_CONFIG | {
        "base_channels": 32,
        "channel_multiplier": [1, 2],
        "embed_dim": 4,
        "image_token_embed_dim": 16,
        "latent_channels": 32,
        "num_embeddings": 64,
        "num_hidden_layers": 2,
        "num_patches": 2,
        "num_res_blocks": 1,
        "projection_dim": 16,
    }
    model = JanusGenerationModel(config).eval()
    tokens = torch.randint(0, 64, (2, 4))
    assert model.decode_image_tokens(tokens).shape == (2, 4, 4, 3)
    assert model.generation_aligner(model.generation_embeddings(tokens)).shape == (2, 4, 16)
    assert model.generation_head(torch.randn(2, 3, 16)).shape == (2, 3, 64)
    with pytest.raises(ValueError, match="Expected `image_tokens`"):
        model.decode_image_tokens(tokens[:, :-1])


def test_postprocess_janus_clips_and_converts_to_uint8():
    decoded = torch.tensor([[[[-2.0, -1.0, 0.0, 1.0, 2.0]]]]).permute(0, 2, 3, 1)
    decoded = decoded.expand(-1, -1, -1, 3)
    pixels = postprocess_janus(decoded)
    assert pixels.dtype == torch.uint8
    assert pixels[0, :, :, 0].tolist() == [[0, 0, 127, 254, 255]]
