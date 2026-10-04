import math

import pytest
import torch
from helpers import TINY_UNET_CONFIG, meta_shapes, published_shapes

from imagehub.models.layers import timestep_embedding
from imagehub.models.unet_2d import UNet2DConditionModel


def test_matches_published_tensor_layout():
    assert meta_shapes(UNet2DConditionModel, {}) == published_shapes("unet")


def test_timestep_embedding_closed_form():
    emb = timestep_embedding(torch.tensor([3]), 4, flip_sin_to_cos=True, shift=0)
    freqs = [1.0, math.exp(-math.log(10000) / 2)]
    expected = [math.cos(3 * f) for f in freqs] + [math.sin(3 * f) for f in freqs]
    torch.testing.assert_close(emb[0], torch.tensor(expected))


@pytest.mark.parametrize("hw", [(8, 8), (9, 7)])
def test_forward_shapes(hw):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(TINY_UNET_CONFIG).eval()
    sample = torch.randn(2, 4, *hw)
    context = torch.randn(2, 77, 32)
    with torch.inference_mode():
        out = unet(sample, torch.tensor(981), context)
        batched_t = unet(sample, torch.tensor([981, 981]), context)
    assert out.shape == sample.shape
    torch.testing.assert_close(out, batched_t, rtol=0, atol=0)
