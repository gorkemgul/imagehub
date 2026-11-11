"""Published-layout checks for Qwen-Image-Edit without checkpoint weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.autoencoder_kl_qwen_image import AutoencoderKLQwenImage
from imagehub.models.qwen2_5_vl_text import Qwen25VLForConditionalGeneration
from imagehub.models.qwen_image_transformer import QwenImageTransformer2DModel

pytestmark = pytest.mark.reference
FIXTURES = Path(__file__).parents[1] / "fixtures/qwen_image_edit_configs"
PUBLISHED = FIXTURES / "published"


def _header_shapes(prefix: str) -> dict[str, tuple[int, ...]]:
    result = {}
    for path in sorted((FIXTURES / "headers").glob(f"{prefix}*.safetensors.json")):
        header = json.loads(path.read_text(encoding="utf-8"))
        result.update(
            {
                name: tuple(entry["shape"] if isinstance(entry, dict) else entry)
                for name, entry in header.items()
                if name != "__metadata__"
            }
        )
    return result


@pytest.mark.parametrize(
    ("model_class", "config_name", "header_prefix"),
    [
        (Qwen25VLForConditionalGeneration, "text_encoder_config.json", "text_encoder_"),
        (QwenImageTransformer2DModel, "transformer_config.json", "transformer_"),
        (AutoencoderKLQwenImage, "vae_config.json", "vae_"),
    ],
)
def test_qwen_image_edit_published_headers_match_meta_model(
    model_class, config_name, header_prefix
):
    config = json.loads((PUBLISHED / config_name).read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = model_class(config)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    assert _header_shapes(header_prefix) == expected
