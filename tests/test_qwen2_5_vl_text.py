import torch

from imagehub.models.qwen2_5_vl_text import Qwen25VLTextModel

TINY = {
    "text_config": {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "rope_scaling": {"type": "default", "mrope_section": [2, 1, 1]},
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "use_sliding_window": False,
        "tie_word_embeddings": False,
    }
}


def test_qwen25_vl_text_published_layout() -> None:
    with torch.device("meta"):
        state = Qwen25VLTextModel(TINY).state_dict()

    assert "model.language_model.embed_tokens.weight" in state
    assert "model.language_model.layers.0.self_attn.q_proj.bias" in state
    assert "model.language_model.layers.0.self_attn.k_proj.bias" in state
    assert "model.language_model.layers.0.self_attn.v_proj.bias" in state
    assert "model.language_model.layers.0.self_attn.o_proj.bias" not in state
    assert "lm_head.weight" in state


def test_qwen25_vl_text_ignores_vision_tensors() -> None:
    model = Qwen25VLTextModel(TINY)
    state = dict(model.state_dict())
    state["model.visual.patch_embed.proj.weight"] = torch.empty(1)

    assert "model.visual.patch_embed.proj.weight" not in model.prepare_state_dict(state)


def test_qwen25_vl_text_generation_stops_at_eos() -> None:
    torch.manual_seed(0)
    model = Qwen25VLTextModel(TINY).eval()
    input_ids = torch.tensor([[4, 5, 6], [4, 5, 6]])
    attention_mask = torch.ones_like(input_ids)
    hidden_states = model(input_ids, attention_mask)
    final = hidden_states[-1][:, -1]
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[2].copy_(final[0] * 100)

    generated = model.generate(
        input_ids,
        attention_mask,
        max_new_tokens=20,
        generation_config={
            "do_sample": True,
            "top_k": 1,
            "top_p": 0.001,
            "temperature": 0.1,
            "repetition_penalty": 1.05,
            "eos_token_id": [2, 3],
            "pad_token_id": 0,
        },
    )

    assert generated.shape == (2, 4)
    assert generated[:, -1].tolist() == [2, 2]
