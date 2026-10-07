import torch

from imagehub.models.bert import BertModel

PUBLISHED_CONFIG = {
    "vocab_size": 47020,
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 4096,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "layer_norm_eps": 1e-12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
}


def test_bert_published_layout_on_meta():
    with torch.device("meta"):
        model = BertModel(PUBLISHED_CONFIG)

    state = model.state_dict()
    assert len(state) == 391
    assert state["embeddings.word_embeddings.weight"].shape == (47020, 1024)
    assert state["embeddings.position_embeddings.weight"].shape == (512, 1024)
    assert state["embeddings.token_type_embeddings.weight"].shape == (2, 1024)
    assert state["encoder.layer.23.attention.self.query.weight"].shape == (1024, 1024)
    assert state["encoder.layer.23.intermediate.dense.weight"].shape == (4096, 1024)
    assert state["pooler.dense.weight"].shape == (1024, 1024)
