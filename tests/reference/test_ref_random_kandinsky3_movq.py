"""Kandinsky 3 MoVQ decoder parity with random weights at the published configuration."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.vq_model import VQModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

CONFIG = {
    key: value
    for key, value in json.loads(
        (Path(__file__).parents[1] / "fixtures/kandinsky3_configs/movq_config.json").read_text()
    ).items()
    if not key.startswith("_")
}


@pytest.mark.parametrize("shape", [(1, 4, 4, 6), (2, 4, 5, 3)], ids=["wide", "odd"])
def test_published_movq_decode_matches_reference(shape):
    torch.manual_seed(0)
    reference = diffusers.VQModel(**CONFIG).eval()
    generator = torch.Generator().manual_seed(0)
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.05, generator=generator)
    ours = VQModel(CONFIG).eval()
    ours.load_checkpoint(reference.state_dict())
    latents = torch.randn(*shape, generator=generator)
    with torch.inference_mode():
        expected = reference.decode(latents, force_not_quantize=True)["sample"]
        actual = ours.decode(latents, force_not_quantize=True)
    assert actual.shape == (shape[0], 3, shape[2] * 8, shape[3] * 8)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
