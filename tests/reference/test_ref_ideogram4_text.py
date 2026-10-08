"""Ideogram 4 prompt packing and Qwen3-VL feature parity."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from imagehub.models.ideogram4_text import (
    build_text_inputs,
    encode_text,
    load_fp8_text_encoder,
    tokenize_prompt,
)
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
ideogram4 = pytest.importorskip("ideogram4")
pipeline_module = pytest.importorskip("ideogram4.pipeline_ideogram4")
quantized_loading = pytest.importorskip("ideogram4.quantized_loading")
transformers = pytest.importorskip("transformers")

FIXTURES = Path(__file__).parents[1] / "fixtures"
PROMPTS = (
    "a red fox in snow",
    '{"high_level_description":"A café sign",'
    '"compositional_deconstruction":{"background":"blue","elements":[]}}',
)
TINY = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5_000_000,
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
        "rope_type": "default",
    },
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
}


def _tokenizers(tmp_path):
    root = tmp_path / "tokenizer"
    shutil.copytree(FIXTURES / "qwen_tokenizer", root)
    tokenizer_path = root / "tokenizer.json"
    tokenizer_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_data["added_tokens"].extend(
        {
            "id": 34 + index,
            "content": prompt,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": False,
        }
        for index, prompt in enumerate(PROMPTS)
    )
    tokenizer_path.write_text(json.dumps(tokenizer_data), encoding="utf-8")
    config_path = root / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chat_template"] = (
        "{{ '<|im_start|>user\\n' + messages[0]['content'][0]['text'] + "
        "'<|im_end|>\\n<|im_start|>assistant\\n' }}"
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    reference = transformers.AutoTokenizer.from_pretrained(root, local_files_only=True)
    return Qwen2Tokenizer(root), reference


def _reference_pipeline(tokenizer):
    pipeline = object.__new__(pipeline_module.Ideogram4Pipeline)
    pipeline.text_tokenizer = tokenizer
    pipeline.config = pipeline_module.Ideogram4PipelineConfig(max_text_tokens=2048)
    pipeline.device = torch.device("cpu")
    return pipeline


def _fp8_state(model):
    state = dict(model.state_dict())
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_name = f"{name}.weight"
            state[weight_name], state[f"{weight_name}_scale"] = (
                quantized_loading.quantize_weight_to_fp8(state[weight_name])
            )
    return state


def test_ideogram4_tokenize_and_build_inputs_exact(tmp_path):
    tokenizer, reference_tokenizer = _tokenizers(tmp_path)
    reference = _reference_pipeline(reference_tokenizer)

    for prompt in PROMPTS:
        actual = tokenize_prompt(tokenizer, prompt)
        expected = reference._tokenize(prompt)
        assert actual[1] == expected[1]
        assert torch.equal(actual[0], expected[0])

    actual_inputs = build_text_inputs(tokenizer, list(PROMPTS), 32, 48)
    expected_inputs = reference._build_inputs(list(PROMPTS), 32, 48)
    assert actual_inputs.keys() == expected_inputs.keys()
    for key in actual_inputs:
        actual = actual_inputs[key]
        expected = expected_inputs[key]
        if isinstance(actual, torch.Tensor):
            assert torch.equal(actual, expected)
        else:
            assert actual == expected


def test_ideogram4_fp8_text_encoder_features_exact(monkeypatch):
    torch.manual_seed(0)
    config = transformers.Qwen3VLTextConfig(**TINY)
    config._attn_implementation = "sdpa"
    reference_model = transformers.Qwen3VLTextModel(config).eval()
    state = _fp8_state(reference_model)
    quantized_loading.swap_linears_to_fp8(reference_model, state, torch.float32)
    quantized_loading.load_fp8_state_dict(
        reference_model,
        state,
        torch.device("cpu"),
        torch.float32,
        assign=True,
    )
    model = load_fp8_text_encoder(TINY, state, torch.device("cpu"), torch.float32)

    monkeypatch.setattr(pipeline_module, "QWEN3_VL_ACTIVATION_LAYERS", (0, 1))
    reference = object.__new__(pipeline_module.Ideogram4Pipeline)
    reference.text_encoder = SimpleNamespace(language_model=reference_model)

    token_ids = torch.randint(0, TINY["vocab_size"], (2, 7))
    indicator = torch.tensor([[0, 0, 3, 3, 3, 3, 3], [0, 3, 3, 3, 3, 3, 3]], dtype=torch.long)
    text_position_ids = torch.zeros(2, 7, 3, dtype=torch.long)
    text_position_ids[0, 2:] = torch.arange(5).view(-1, 1).expand(-1, 3)
    text_position_ids[1, 1:] = torch.arange(6).view(-1, 1).expand(-1, 3)
    attention_mask = (indicator == 3).long()
    with torch.inference_mode():
        selected = reference._get_qwen3_vl_embeddings(
            token_ids, attention_mask, text_position_ids[..., 0]
        )
        expected = torch.stack(selected).permute(1, 2, 3, 0).reshape(2, 7, -1)
        expected = expected * attention_mask.unsqueeze(-1)
        actual = encode_text(
            model,
            token_ids,
            text_position_ids,
            indicator,
            activation_layers=(0, 1),
        )

    torch.testing.assert_close(actual, expected.float(), atol=0, rtol=0)
    assert actual.shape == (2, 7, TINY["hidden_size"] * 2)
    assert all(
        module.weight.dtype == torch.float8_e4m3fn
        for module in model.modules()
        if module.__class__.__name__ == "Fp8Linear"
    )
