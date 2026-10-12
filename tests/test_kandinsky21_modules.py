import json
from pathlib import Path

import pytest
import torch

from imagehub.checkpoints import read_json
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.clip_vision import CLIPVisionModelWithProjection
from imagehub.models.multilingual_clip import MultilingualCLIP
from imagehub.models.prior_transformer import PriorTransformer
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.models.vq_model import VQModel

FIXTURES = Path(__file__).parent / "fixtures/kandinsky21_configs"


def _header_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {name: tuple(shape) for name, shape in data.items()}


def _check_header(name: str, model, header_path: Path) -> None:
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    published = _header_shapes(header_path)
    prepared = model.prepare_state_dict(
        {key: torch.empty(shape, device="meta") for key, shape in published.items()}
    )
    actual = {key: tuple(value.shape) for key, value in prepared.items()}
    assert actual == expected, f"{name} header mismatch"


@pytest.mark.parametrize(
    "config_path, header_path, cls",
    [
        (
            FIXTURES / "prior/text_encoder/config.json",
            FIXTURES / "prior_text_encoder_header.json",
            CLIPTextModelWithProjection,
        ),
        (
            FIXTURES / "prior/image_encoder/config.json",
            FIXTURES / "prior_image_encoder_header.json",
            CLIPVisionModelWithProjection,
        ),
        (
            FIXTURES / "prior/prior/config.json",
            FIXTURES / "prior_diffusion_header.json",
            PriorTransformer,
        ),
        (
            FIXTURES / "decoder/text_encoder/config.json",
            FIXTURES / "decoder_text_encoder_header.json",
            MultilingualCLIP,
        ),
        (
            FIXTURES / "decoder/unet/config.json",
            FIXTURES / "decoder_unet_header.json",
            UNet2DConditionModel,
        ),
        (
            FIXTURES / "decoder/movq/config.json",
            FIXTURES / "decoder_movq_header.json",
            VQModel,
        ),
    ],
)
def test_published_layout(config_path, header_path, cls):
    with torch.device("meta"):
        model = cls(read_json(config_path))
    _check_header(config_path.parent.name, model, header_path)
