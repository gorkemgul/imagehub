"""Structural parity with random weights: no checkpoint download needed."""

import pytest
import torch
from helpers import TINY_TEXT_CONFIG, text_state_for_ours

from imagehub.models.clip_text import CLIP_TEXT_DEFAULTS, CLIPTextModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")
diffusers = pytest.importorskip("diffusers")


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("config", [TINY_TEXT_CONFIG, {}], ids=["tiny", "sd1"])
def test_clip_text_random_weights(config):
    torch.manual_seed(0)
    ref = transformers.CLIPTextModel(
        transformers.CLIPTextConfig(**(CLIP_TEXT_DEFAULTS | config))
    ).eval()
    ours = CLIPTextModel(config).eval()
    ours.load_checkpoint(text_state_for_ours(ref.state_dict()))
    ids = torch.randint(0, ours.config["vocab_size"], (3, 77))
    with torch.inference_mode():
        exact(ours(ids), ref(ids)[0])
