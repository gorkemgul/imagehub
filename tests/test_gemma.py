import json
from pathlib import Path

import pytest
import torch

from imagehub.models.gemma import GemmaModel

FIXTURES = Path(__file__).parent / "fixtures" / "lumina_next_configs"
TINY = {
    "vocab_size": 16,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
}


def _published_config():
    return json.loads((FIXTURES / "text_encoder_config.json").read_text(encoding="utf-8"))


def test_matches_published_tensor_names_and_shapes():
    with torch.device("meta"):
        model = GemmaModel(_published_config())
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == json.loads((FIXTURES / "text_encoder_tensors.json").read_text())
    assert actual["embed_tokens.weight"] == [256000, 2048]
    assert actual["layers.0.self_attn.k_proj.weight"] == [256, 2048]
    assert "layers.0.pre_feedforward_layernorm.weight" not in actual


def test_loads_on_meta_then_moves():
    config = TINY
    with torch.device("meta"):
        reference = GemmaModel(config)
    state = {name: torch.randn(value.shape) for name, value in reference.state_dict().items()}
    model = GemmaModel(config)
    model.load_checkpoint(state)
    model.init_non_persistent_buffers()
    model.to("cpu")
    ids = torch.tensor([[1, 2, 3, 4]])
    assert model(ids).shape == (1, 4, 16)


def test_legacy_gelu_name_selects_the_tanh_approximation():
    GemmaModel(TINY | {"hidden_act": "gelu", "hidden_activation": "gelu_pytorch_tanh"})
    GemmaModel(TINY | {"hidden_act": "gelu"})
    with pytest.raises(ValueError, match="Unsupported Gemma activation"):
        GemmaModel(TINY | {"hidden_act": "relu"})


def test_padding_does_not_change_earlier_positions():
    torch.manual_seed(0)
    model = GemmaModel(TINY).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
    with torch.inference_mode():
        padded = model(ids, mask)
        short = model(ids[:, :5])
    assert torch.allclose(padded[:, :5], short, atol=1e-5)


def test_hidden_state_index_selects_layer_outputs():
    torch.manual_seed(0)
    model = GemmaModel(TINY).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        final = model(ids)
        assert torch.equal(model(ids, hidden_state_index=-1), final)
        before_last = model(ids, hidden_state_index=-2)
        assert torch.equal(model(ids, hidden_state_index=2), before_last)
        assert not torch.equal(before_last, final)
