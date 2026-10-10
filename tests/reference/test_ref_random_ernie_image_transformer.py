"""ERNIE-Image transformer structural parity with random weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.ernie_image_transformer import ErnieImageTransformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

TINY = {
    "hidden_size": 32,
    "num_attention_heads": 4,
    "num_layers": 2,
    "ffn_hidden_size": 48,
    "in_channels": 8,
    "out_channels": 8,
    "patch_size": 1,
    "text_in_dim": 24,
    "rope_theta": 256,
    "rope_axes_dim": [2, 2, 4],
    "eps": 1e-6,
    "qk_layernorm": True,
}
SAME_WIDTH = TINY | {"text_in_dim": 32}
PUBLISHED = json.loads(
    (
        Path(__file__).parents[1]
        / "fixtures"
        / "ernie_image_configs"
        / "transformer"
        / "config.json"
    ).read_text()
)
PUBLISHED_SMALL = {key: value for key, value in PUBLISHED.items() if not key.startswith("_")} | {
    "num_layers": 2,
    "hidden_size": 256,
    "ffn_hidden_size": 384,
    "num_attention_heads": 2,
    "text_in_dim": 96,
    "in_channels": 16,
    "out_channels": 16,
}


def _randomise(model):
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=gen) * 0.05)


@pytest.mark.parametrize("config", [TINY, SAME_WIDTH], ids=["tiny", "no-text-proj"])
@pytest.mark.parametrize("lens", [(7, 7), (7, 4)], ids=["equal", "ragged"])
def test_transformer_random_weights_exact(config, lens):
    torch.manual_seed(0)
    reference = diffusers.ErnieImageTransformer2DModel(**config).eval()
    _randomise(reference)
    model = ErnieImageTransformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    batch, height, width = 2, 3, 5
    hidden = torch.randn(batch, config["in_channels"], height, width)
    text = torch.randn(batch, max(lens), config["text_in_dim"])
    text_lens = torch.tensor(lens)
    timestep = torch.tensor([999.0, 250.5])
    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden,
            timestep=timestep,
            text_bth=text,
            text_lens=text_lens,
            return_dict=False,
        )[0]
        actual = model(hidden, timestep, text, text_lens)
    assert torch.equal(actual, expected)


def test_transformer_random_weights_published_shape_exact():
    torch.manual_seed(0)
    reference = diffusers.ErnieImageTransformer2DModel(**PUBLISHED_SMALL).eval()
    _randomise(reference)
    model = ErnieImageTransformer2DModel(PUBLISHED_SMALL).eval()
    model.load_checkpoint(reference.state_dict())
    hidden = torch.randn(2, 16, 4, 6)
    text = torch.randn(2, 9, 96)
    text_lens = torch.tensor([9, 5])
    timestep = torch.tensor([1000.0, 3.0])
    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden,
            timestep=timestep,
            text_bth=text,
            text_lens=text_lens,
            return_dict=False,
        )[0]
        actual = model(hidden, timestep, text, text_lens)
    assert torch.equal(actual, expected)
