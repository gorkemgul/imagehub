"""Qwen2.5-VL text model and rewrite-generation parity with random weights."""

from typing import Any

import pytest
import torch

from imagehub.models.qwen2_5_vl_text import Qwen25VLTextModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

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
        "bos_token_id": 1,
        "eos_token_id": [2, 3],
        "pad_token_id": 0,
    },
    "vision_config": {
        "depth": 1,
        "embed_dim": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 4,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "in_channels": 3,
    },
    "image_token_id": 60,
    "video_token_id": 61,
    "bos_token_id": 1,
    "eos_token_id": [2, 3],
    "pad_token_id": 0,
}
MEDIUM = {
    **TINY,
    "text_config": {
        **TINY["text_config"],
        "vocab_size": 128,
        "hidden_size": 96,
        "intermediate_size": 192,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "rope_scaling": {"type": "default", "mrope_section": [2, 2, 2]},
    },
}
GENERATION_CONFIG = {
    "do_sample": True,
    "top_k": 1,
    "top_p": 0.001,
    "temperature": 0.1,
    "repetition_penalty": 1.05,
    "eos_token_id": [2, 3],
    "pad_token_id": 0,
}


def _reference(model_config: dict[str, Any] = TINY) -> Any:
    config = transformers.Qwen2_5_VLConfig(**model_config)
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    return transformers.Qwen2_5_VLForConditionalGeneration(config).eval()


def _generate_with_logits(
    reference: Any,
    model: Qwen25VLTextModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generation_config: dict[str, Any],
    max_new_tokens: int,
) -> tuple[Any, torch.Tensor, list[torch.Tensor]]:
    actual_logits = []

    def capture_logits(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        actual_logits.append(output[:, -1].float().clone())

    handle = model.lm_head.register_forward_hook(capture_logits)
    try:
        with torch.inference_mode():
            expected = reference.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                generation_config=transformers.GenerationConfig(
                    **generation_config,
                    output_logits=True,
                    return_dict_in_generate=True,
                ),
            )
            actual = model.generate(
                input_ids,
                attention_mask,
                max_new_tokens=max_new_tokens,
                generation_config=generation_config,
            )
    finally:
        handle.remove()
    return expected, actual, actual_logits


def test_qwen25_vl_random_hidden_states_and_logits_exact() -> None:
    torch.manual_seed(0)
    reference = _reference()
    model = Qwen25VLTextModel(TINY).eval()
    model.load_checkpoint(reference.state_dict())
    input_ids = torch.tensor([[0, 0, 4, 5, 6, 7], [8, 9, 10, 11, 0, 0]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])

    with torch.inference_mode():
        expected = reference(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        actual_hidden_states = model(input_ids, attention_mask)
        actual_logits = model.lm_head(actual_hidden_states[-1])

    torch.testing.assert_close(actual_hidden_states[-1], expected.hidden_states[-1], atol=0, rtol=0)
    torch.testing.assert_close(actual_logits, expected.logits, atol=0, rtol=0)


def test_qwen25_vl_rewrite_generation_exact() -> None:
    torch.manual_seed(1)
    reference = _reference()
    model = Qwen25VLTextModel(TINY).eval()
    model.load_checkpoint(reference.state_dict())
    input_ids = torch.tensor([[0, 0, 12, 13, 14], [15, 16, 17, 18, 19]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    reference_config = transformers.GenerationConfig(**GENERATION_CONFIG)

    with torch.inference_mode():
        expected = reference.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            generation_config=reference_config,
        )
        actual = model.generate(
            input_ids,
            attention_mask,
            max_new_tokens=20,
            generation_config=GENERATION_CONFIG,
        )

    assert expected.shape == (2, 25)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("model_config", [TINY, MEDIUM], ids=["tiny", "medium"])
def test_qwen25_vl_cached_generation_logits_exact(model_config: dict[str, Any]) -> None:
    torch.manual_seed(3)
    reference = _reference(model_config)
    model = Qwen25VLTextModel(model_config).eval()
    model.load_checkpoint(reference.state_dict())
    input_ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10]])
    attention_mask = torch.ones_like(input_ids)
    generation_config = {
        **GENERATION_CONFIG,
        "eos_token_id": model_config["text_config"]["vocab_size"] + 1,
    }

    expected, actual, actual_logits = _generate_with_logits(
        reference,
        model,
        input_ids,
        attention_mask,
        generation_config,
        max_new_tokens=32,
    )

    assert torch.equal(actual, expected.sequences)
    assert len(expected.logits) == len(actual_logits) == 32
    for actual_step, expected_step in zip(actual_logits, expected.logits, strict=True):
        torch.testing.assert_close(actual_step, expected_step, atol=0, rtol=0)


def test_qwen25_vl_rewrite_generation_early_eos_exact() -> None:
    torch.manual_seed(2)
    reference = _reference()
    model = Qwen25VLTextModel(TINY).eval()
    model.load_checkpoint(reference.state_dict())
    input_ids = torch.tensor([[4, 5, 6], [4, 5, 6]])
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = reference(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        reference.lm_head.weight.zero_()
        reference.lm_head.weight[2].copy_(output.hidden_states[-1][0, -1] * 100)
    model.load_checkpoint(reference.state_dict())
    expected, actual, actual_logits = _generate_with_logits(
        reference,
        model,
        input_ids,
        attention_mask,
        GENERATION_CONFIG,
        max_new_tokens=20,
    )

    assert expected.sequences.shape == (2, 4)
    assert torch.equal(actual, expected.sequences)
    assert len(expected.logits) == len(actual_logits) == 1
    torch.testing.assert_close(actual_logits[0], expected.logits[0], atol=0, rtol=0)
