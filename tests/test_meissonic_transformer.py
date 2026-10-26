import json
from pathlib import Path

import torch

from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.meissonic_transformer import MeissonicTransformer2DModel
from imagehub.models.vq_model import VQModel

FIXTURES = Path(__file__).parent / "fixtures" / "meissonic_configs"


def _layout(model, fixture):
    expected = json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
    actual = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    assert actual == expected


def test_published_tensor_layouts_on_meta():
    with torch.device("meta"):
        _layout(
            MeissonicTransformer2DModel(
                json.loads((FIXTURES / "transformer_config.json").read_text())
            ),
            "transformer_tensors.json",
        )
        _layout(
            CLIPTextModelWithProjection(
                json.loads((FIXTURES / "text_encoder_config.json").read_text())
            ),
            "text_encoder_tensors.json",
        )
        _layout(
            VQModel(json.loads((FIXTURES / "vqvae_config.json").read_text())),
            "vqvae_tensors.json",
        )


def test_tiny_forward_shape():
    from helpers import TINY_MEISSONIC_TRANSFORMER

    model = MeissonicTransformer2DModel(TINY_MEISSONIC_TRANSFORMER).eval()
    ids = torch.randint(0, 24, (2, 4, 6))
    text = torch.randn(2, 5, 32)
    pooled = torch.randn(2, 16)
    micro = torch.tensor([[64.0, 64.0, 0.0, 0.0, 6.0]]).expand(2, -1)
    image_ids = torch.zeros(24, 3)
    text_ids = torch.zeros(5, 3)
    with torch.inference_mode():
        logits = model(ids, text, pooled, torch.tensor([1, 1]), image_ids, text_ids, micro)
    assert logits.shape == (2, 20, 4, 6)
