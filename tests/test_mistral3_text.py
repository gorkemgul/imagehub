import json
from pathlib import Path

import pytest
import torch

from imagehub.exceptions import CheckpointError
from imagehub.models.mistral3_text import Mistral3TextModel, rope_inverse_frequencies

FIXTURES = Path(__file__).parent / "fixtures" / "ernie_image_configs"


def test_language_model_matches_published_tensor_names_on_the_meta_device():
    config = json.loads((FIXTURES / "text_encoder" / "config.json").read_text(encoding="utf-8"))
    published = json.loads((FIXTURES / "text_encoder_tensors.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = Mistral3TextModel(config)
    expected = {
        name: shape
        for name, shape in published.items()
        if not name.startswith(("vision_tower.", "multi_modal_projector."))
    }
    assert {name: list(value.shape) for name, value in model.state_dict().items()} == expected
    state = {name: torch.empty(shape, device="meta") for name, shape in published.items()}
    assert set(model.prepare_state_dict(state)) == set(expected)
    # The language model alone has 3.4B parameters; the vision tower is not part of it.
    assert sum(p.numel() for p in model.parameters()) == 3_429_006_336


def test_meta_construction_survives_and_buffers_are_rebuilt():
    config = json.loads((FIXTURES / "text_encoder" / "config.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = Mistral3TextModel(config)
    model.init_non_persistent_buffers()
    rotary = model.language_model.model.rotary_emb
    assert rotary.inv_freq.device.type == "cpu" and rotary.inv_freq.shape == (64,)
    assert rotary.attention_scaling == 1.0


def test_yarn_frequencies_blend_interpolation_and_extrapolation():
    parameters = json.loads((FIXTURES / "text_encoder" / "config.json").read_text())["text_config"][
        "rope_parameters"
    ]
    inv_freq, scaling = rope_inverse_frequencies(128, 262144, parameters)
    plain, _ = rope_inverse_frequencies(128, 262144, {"rope_theta": 1_000_000.0})
    assert scaling == 1.0
    assert inv_freq[0] == plain[0]  # high frequencies are extrapolated unchanged
    assert inv_freq[-1] == pytest.approx(float(plain[-1]) / 16.0, rel=1e-6)  # low ones interpolated


def test_unsupported_settings_are_rejected():
    with pytest.raises(ValueError, match="rope type"):
        rope_inverse_frequencies(8, 64, {"rope_theta": 1.0, "rope_type": "linear"})
    with pytest.raises(ValueError, match="sliding window"):
        Mistral3TextModel({"text_config": {"sliding_window": 4}})


def test_missing_language_tensors_are_reported():
    config = {
        "text_config": {
            "vocab_size": 8,
            "hidden_size": 8,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,
        }
    }
    model = Mistral3TextModel(config)
    with pytest.raises(CheckpointError):
        model.load_checkpoint({"vision_tower.x": torch.zeros(1)})
