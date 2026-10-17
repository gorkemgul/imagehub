from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from imagehub.backends.joyai_image import JoyAIImageBackend
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl_wan import AutoencoderKLWan
from imagehub.models.joyimage_transformer import JoyImageEditTransformer3DModel
from imagehub.models.qwen3_vl import Qwen3VLForConditionalGeneration
from imagehub.types import T2IRequest

_CONFIG = {
    "patch_size": [1, 2, 2],
    "in_channels": 4,
    "out_channels": 4,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "text_dim": 12,
    "mlp_width_ratio": 2.0,
    "num_layers": 2,
    "rope_dim_list": [2, 2, 0],
    "rope_type": "rope",
    "theta": 10000,
}
_FIXTURES = Path(__file__).parent / "fixtures/joyai_image_configs"


def test_joy_transformer_preserves_single_and_multi_item_shapes() -> None:
    model = JoyImageEditTransformer3DModel(_CONFIG).eval()
    text = torch.randn(1, 3, 12)
    assert model(torch.randn(1, 4, 1, 4, 4), torch.tensor([1.0]), text).shape == (1, 4, 1, 4, 4)
    assert model(torch.randn(1, 2, 4, 1, 4, 4), torch.tensor([1.0]), text).shape == (
        1,
        2,
        4,
        1,
        4,
        4,
    )


def test_joy_options_are_validated_before_loading(tmp_path) -> None:
    backend = JoyAIImageBackend(tmp_path, model_id="joyai-image-edit-plus")
    with pytest.raises(ValueError, match="image is required"):
        backend.validate_options(T2IRequest("edit"), {})
    with pytest.raises(TypeError, match="Unknown JoyAI-Image options"):
        backend.validate_options(T2IRequest("edit"), {"unknown": 1})
    assert not backend.is_loaded


@pytest.mark.parametrize("variant", ("base", "plus"))
def test_published_text_and_vision_headers_match_strict_model(variant: str) -> None:
    root = _FIXTURES / variant
    published = {}
    for path in root.glob("text_encoder_*.header.json"):
        published |= {
            name: shape
            for name, shape in json.loads(path.read_text()).items()
            if name != "__metadata__"
        }
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(read_json(root / "text_encoder/config.json"))
    expected = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert published == expected


@pytest.mark.parametrize("variant", ("base", "plus"))
def test_published_transformer_headers_match_strict_model(variant: str) -> None:
    root = _FIXTURES / variant
    published = {}
    for path in root.glob("transformer_*.header.json"):
        published |= {
            name: shape
            for name, shape in json.loads(path.read_text()).items()
            if name != "__metadata__"
        }
    with torch.device("meta"):
        model = JoyImageEditTransformer3DModel(read_json(root / "transformer/config.json"))
    expected = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert published == expected


@pytest.mark.parametrize("variant", ("base", "plus"))
def test_published_vae_headers_match_strict_model(variant: str) -> None:
    root = _FIXTURES / variant
    published = {
        name: shape
        for name, shape in json.loads(
            (root / "vae_diffusion_pytorch_model.safetensors.header.json").read_text()
        ).items()
        if name != "__metadata__"
    }
    with torch.device("meta"):
        model = AutoencoderKLWan(read_json(root / "vae/config.json"))
    expected = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert published == expected
