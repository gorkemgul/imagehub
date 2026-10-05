import json
from pathlib import Path

import pytest
import torch
from helpers import (
    TINY_TEXT_CONFIG,
    meta_shapes,
    published_shapes,
    sdxl_published_shapes,
    ssd1b_published_shapes,
)

from imagehub.models.clip_text import CLIPTextModel, CLIPTextModelWithProjection

FIXTURES = Path(__file__).parent / "fixtures"


def test_matches_published_tensor_layout():
    expected = published_shapes("text_encoder")
    expected.pop("text_model.embeddings.position_ids")
    assert meta_shapes(CLIPTextModel, {}) == expected


def test_matches_sdxl_text_encoder_layouts():
    first = sdxl_published_shapes("text_encoder")
    assert meta_shapes(CLIPTextModel, {}) == first
    second_config = {
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "num_hidden_layers": 32,
        "num_attention_heads": 20,
        "hidden_act": "gelu",
        "projection_dim": 1280,
        "eos_token_id": 2,
    }
    assert meta_shapes(CLIPTextModelWithProjection, second_config) == sdxl_published_shapes(
        "text_encoder_2"
    )


@pytest.mark.parametrize(
    ("model_cls", "config_path", "component", "family"),
    [
        (CLIPTextModel, "sdxl_configs/text_encoder.json", "text_encoder", "sdxl"),
        (
            CLIPTextModelWithProjection,
            "sdxl_configs/text_encoder_2.json",
            "text_encoder_2",
            "sdxl",
        ),
        (CLIPTextModel, "sd1_configs/v1-5/text_encoder.json", "text_encoder", "sd1"),
        (CLIPTextModel, "sd1_configs/v1-1/text_encoder.json", "text_encoder", "sd1"),
        (CLIPTextModel, "ssd1b_configs/text_encoder.json", "text_encoder", "ssd1b"),
        (
            CLIPTextModelWithProjection,
            "ssd1b_configs/text_encoder_2.json",
            "text_encoder_2",
            "ssd1b",
        ),
    ],
)
def test_real_release_configs_match_published_tensor_layout(
    model_cls, config_path, component, family
):
    config = json.loads((FIXTURES / config_path).read_text(encoding="utf-8"))
    if family == "sdxl":
        expected = sdxl_published_shapes(component)
    elif family == "ssd1b":
        expected = ssd1b_published_shapes(component)
    else:
        expected = published_shapes(component)
    expected.pop("text_model.embeddings.position_ids", None)
    assert meta_shapes(model_cls, config) == expected


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


def test_projection_position_ids_are_dropped_on_load():
    config = TINY_TEXT_CONFIG | {"hidden_act": "gelu", "projection_dim": 24, "eos_token_id": 2}
    model = CLIPTextModelWithProjection(config)
    state = dict(model.state_dict())
    state["text_model.embeddings.position_ids"] = torch.arange(77)[None]
    model.load_checkpoint(state)


def test_rejects_other_activations():
    with pytest.raises(ValueError, match="activation"):
        CLIPTextModel(TINY_TEXT_CONFIG | {"hidden_act": "relu"})


def test_returns_penultimate_hidden_state_and_projected_eos():
    config = TINY_TEXT_CONFIG | {
        "hidden_act": "gelu",
        "projection_dim": 24,
        "eos_token_id": 2,
    }
    model = CLIPTextModelWithProjection(config).eval()
    ids = torch.tensor([[7, 2, 2, 0], [9, 8, 2, 0]])
    text_embeds, last_hidden_state, hidden_states = model(ids, output_hidden_states=True)
    assert text_embeds.shape == (2, 24)
    assert last_hidden_state.shape == (2, 4, 32)
    assert len(hidden_states) == config["num_hidden_layers"] + 1
    pooled = last_hidden_state[torch.arange(2), ids.argmax(dim=-1)]
    torch.testing.assert_close(text_embeds, model.text_projection(pooled), rtol=0, atol=0)
    assert hidden_states[-2].shape == last_hidden_state.shape
