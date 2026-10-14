"""T5Gemma encoder parity with random weights."""

import pytest
import torch

from imagehub.models.t5gemma import T5GemmaEncoder

pytestmark = pytest.mark.reference


def _config(*, published_shape: bool = False) -> dict[str, object]:
    return {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_logit_softcapping": 50.0,
        "query_pre_attn_scalar": 256 if published_shape else 8,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "hidden_activation": "gelu_pytorch_tanh",
        "dropout_rate": 0.0,
        "pad_token_id": 0,
    }


@pytest.mark.parametrize("published_shape", [False, True])
def test_t5gemma_encoder_exact_parity(record, published_shape: bool) -> None:
    transformers = pytest.importorskip("transformers")
    modeling = pytest.importorskip("transformers.models.t5gemma.modeling_t5gemma")
    config = _config(published_shape=published_shape)
    reference_config = transformers.T5GemmaModuleConfig(**config)
    reference = modeling.T5GemmaEncoder(reference_config).eval()
    model = T5GemmaEncoder(config).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    input_ids = torch.tensor([[2, 7, 9, 0, 0], [2, 5, 6, 8, 1]])
    attention_mask = input_ids.ne(0)
    with torch.no_grad():
        expected = reference(input_ids, attention_mask=attention_mask).last_hidden_state
        actual = model(input_ids, attention_mask=attention_mask)
    difference = float((actual - expected).abs().max())
    record(
        f"t5gemma_{'published' if published_shape else 'tiny'}",
        max_abs_diff=difference,
        tolerance="exact",
    )
    assert torch.equal(actual, expected)
