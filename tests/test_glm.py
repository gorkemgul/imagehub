import torch

from imagehub.models.glm import GlmModel

CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 64,
    "rms_norm_eps": 1.5625e-7,
    "rope_theta": 10_000.0,
    "partial_rotary_factor": 0.5,
    "attention_bias": True,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "pad_token_id": 63,
}


def test_glm_published_tensor_names_and_shapes():
    with torch.device("meta"):
        state = GlmModel(CONFIG).state_dict()

    assert tuple(state["embed_tokens.weight"].shape) == (64, 32)
    assert tuple(state["layers.0.self_attn.q_proj.weight"].shape) == (32, 32)
    assert tuple(state["layers.0.self_attn.k_proj.weight"].shape) == (16, 32)
    assert tuple(state["layers.0.self_attn.q_proj.bias"].shape) == (32,)
    assert tuple(state["layers.0.mlp.gate_up_proj.weight"].shape) == (96, 32)
    assert tuple(state["layers.0.mlp.down_proj.weight"].shape) == (32, 48)


def test_glm_returns_pipeline_hidden_state():
    torch.manual_seed(0)
    model = GlmModel(CONFIG).eval()
    input_ids = torch.randint(0, 64, (2, 16))

    with torch.inference_mode():
        hidden = model(input_ids)

    assert hidden.shape == (2, 16, 32)
    assert torch.isfinite(hidden).all()


def test_glm_non_persistent_rotary_buffer_survives_meta_move():
    with torch.device("meta"):
        model = GlmModel(CONFIG)
    model.to_empty(device="cpu")
    model.init_non_persistent_buffers()

    torch.testing.assert_close(model.rotary_emb.inv_freq, torch.tensor([1.0, 0.01]))
