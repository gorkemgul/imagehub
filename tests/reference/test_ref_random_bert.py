"""BERT encoder structural parity with random weights."""

import pytest
import torch

from imagehub.models.bert import BertModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

TINY_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "layer_norm_eps": 1e-12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
}
MEDIUM_CONFIG = TINY_CONFIG | {
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "intermediate_size": 1024,
}


@pytest.mark.parametrize("config", [TINY_CONFIG, MEDIUM_CONFIG], ids=["tiny", "medium"])
def test_bert_encoder_random_weights(config):
    torch.manual_seed(0)
    reference = transformers.BertModel(transformers.BertConfig(**config)).eval()
    ours = BertModel(config).eval()
    ours.load_checkpoint(reference.state_dict())

    input_ids = torch.randint(1, config["vocab_size"], (2, 77))
    attention_mask = torch.ones(2, 77, dtype=torch.long)
    attention_mask[0, 37:] = 0
    attention_mask[1, 71:] = 0
    input_ids.masked_fill_(attention_mask == 0, 0)

    with torch.inference_mode():
        expected = reference(input_ids, attention_mask=attention_mask).last_hidden_state
        actual = ours(input_ids, attention_mask)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
