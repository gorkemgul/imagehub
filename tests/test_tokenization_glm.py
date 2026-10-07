import json

import torch

from imagehub.tokenization_glm import GlmTokenizer, _pretokenize


def _write_tokenizer(path):
    vocabulary = {
        "a": 0,
        "b": 1,
        "c": 2,
        "Ġ": 3,
        "Ġa": 4,
        "ab": 5,
        "abc": 6,
        "1": 7,
        "2": 8,
        "3": 9,
        "4": 10,
    }
    data = {
        "model": {
            "type": "BPE",
            "vocab": vocabulary,
            "merges": [["a", "b"], ["ab", "c"]],
            "ignore_merges": True,
        },
        "added_tokens": [
            {"id": 11, "content": "<|endoftext|>", "normalized": False},
            {"id": 12, "content": "[gMASK]", "normalized": False},
            {"id": 13, "content": "<sop>", "normalized": False},
        ],
    }
    path.mkdir()
    (path / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps({"pad_token": "<|endoftext|>", "padding_side": "left"}), encoding="utf-8"
    )


def test_glm_pretokenizer_groups_one_to_three_digits():
    assert _pretokenize("1234567") == ["123", "456", "7"]


def test_glm_special_prefix_ignore_merges_and_pipeline_padding(tmp_path):
    path = tmp_path / "tokenizer"
    _write_tokenizer(path)
    tokenizer = GlmTokenizer(path)

    assert tokenizer.tokenize("abc") == [12, 13, 6]
    input_ids, attention_mask = tokenizer(["abc", "a"], max_length=16)

    assert input_ids.shape == (2, 16)
    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.bool
    assert input_ids[0, -3:].tolist() == [12, 13, 6]
    assert input_ids[1, -3:].tolist() == [12, 13, 0]
    assert attention_mask.sum(dim=1).tolist() == [3, 3]
