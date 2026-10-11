"""Llama text encoder parity with random weights (exact, CPU float32), llama3 rope scaling."""

import pytest
import torch

from imagehub.models.llama_text import LlamaTextModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

CONFIG = {
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 64,
        "rope_type": "llama3",
    },
}


@pytest.mark.parametrize("scaling", [True, False], ids=["llama3-rope", "default-rope"])
@pytest.mark.parametrize("padded", [True, False], ids=["padded", "unpadded"])
def test_hidden_states_match_transformers_exactly(scaling, padded):
    config = CONFIG if scaling else CONFIG | {"rope_scaling": None}
    torch.manual_seed(0)
    reference = transformers.LlamaForCausalLM(transformers.LlamaConfig(**config)).eval()
    reference.config._attn_implementation = "eager"
    model = LlamaTextModel(config).eval()
    state = reference.state_dict()
    model.load_checkpoint(state)
    ids = torch.randint(0, 64, (3, 40), generator=torch.Generator().manual_seed(1))
    mask = torch.ones(3, 40, dtype=torch.long)
    if padded:
        mask[0, 25:] = 0
        mask[1, 9:] = 0
    with torch.no_grad():
        expected = reference(ids, attention_mask=mask, output_hidden_states=True).hidden_states
    expected = torch.stack(expected[1:])
    actual = model(ids, mask.bool())
    assert torch.equal(actual, expected)
