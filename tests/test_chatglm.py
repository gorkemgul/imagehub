import json
from pathlib import Path

import pytest
import torch

from imagehub.models.chatglm import ChatGLMModel

FIXTURES = Path(__file__).parent / "fixtures" / "kolors_configs"
TINY = {
    "num_layers": 3,
    "padded_vocab_size": 64,
    "hidden_size": 32,
    "ffn_hidden_size": 48,
    "kv_channels": 8,
    "num_attention_heads": 4,
    "seq_length": 64,
    "add_qkv_bias": True,
    "multi_query_attention": True,
    "multi_query_group_num": 2,
}


def test_chatglm_meta_construction_matches_published_tensors():
    config = json.loads((FIXTURES / "text_encoder_config.json").read_text())
    published = json.loads((FIXTURES / "text_encoder_tensors.json").read_text())
    with torch.device("meta"):
        state = ChatGLMModel(config).state_dict()

    assert len(published) == 200
    assert {name: list(value.shape) for name, value in state.items()} == published


def test_chatglm_outputs_and_padding_mask():
    torch.manual_seed(0)
    model = ChatGLMModel(TINY).eval()
    input_ids = torch.randint(1, 64, (2, 12))
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[0, :4] = False
    positions = torch.arange(12).repeat(2, 1)
    positions[0] = torch.cat((torch.zeros(4, dtype=torch.long), torch.arange(8)))

    with torch.inference_mode():
        embeds, pooled = model(input_ids, positions, mask)
        shifted, shifted_pooled = model(input_ids[:1, 4:])

    assert embeds.shape == (2, 12, 32)
    assert pooled.shape == (2, 32)
    # Left padding is invisible to the real tokens of the padded row.
    torch.testing.assert_close(embeds[0, 4:], shifted[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pooled[0], shifted_pooled[0], atol=1e-5, rtol=1e-5)


def test_chatglm_rejects_prefix_encoder():
    with pytest.raises(ValueError):
        ChatGLMModel(TINY | {"pre_seq_len": 4})
