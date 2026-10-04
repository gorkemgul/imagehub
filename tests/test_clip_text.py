import pytest
import torch
from helpers import TINY_TEXT_CONFIG, meta_shapes, published_shapes

from imagehub.models.clip_text import CLIPTextModel


def test_matches_published_tensor_layout():
    expected = published_shapes("text_encoder")
    expected.pop("text_model.embeddings.position_ids")
    assert meta_shapes(CLIPTextModel, {}) == expected


def test_forward_shape_and_causality():
    torch.manual_seed(0)
    model = CLIPTextModel(TINY_TEXT_CONFIG).eval()
    ids = torch.randint(0, 517, (2, 77))
    changed = ids.clone()
    changed[:, 10] = (changed[:, 10] + 1) % 517
    with torch.inference_mode():
        a, b = model(ids), model(changed)
    assert a.shape == (2, 77, 32)
    torch.testing.assert_close(a[:, :10], b[:, :10])
    assert not torch.equal(a[:, 10:], b[:, 10:])


def test_position_ids_are_dropped_on_load():
    model = CLIPTextModel(TINY_TEXT_CONFIG)
    state = dict(model.state_dict())
    state["text_model.embeddings.position_ids"] = torch.arange(77)[None]
    model.load_checkpoint(state)


def test_rejects_other_activations():
    with pytest.raises(ValueError, match="activation"):
        CLIPTextModel(TINY_TEXT_CONFIG | {"hidden_act": "gelu"})
