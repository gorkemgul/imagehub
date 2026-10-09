import json

import pytest
import torch

from imagehub.tokenization_auraflow import AuraFlowTokenizer

PIECES = ["<unk>", "<s>", "</s>", "<x>"]
SYMBOLS = ["▁", "a", "b", "▁a", "ab", "▁ab"]
BYTES = [f"<0x{byte:02X}>" for byte in range(256)]
EOS_TEMPLATE = {
    "type": "TemplateProcessing",
    "single": [
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "</s>", "type_id": 0}},
    ],
    "special_tokens": {"</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}},
}


def _write(directory, post_processor=EOS_TEMPLATE, byte_tokens=True):
    symbols = [*SYMBOLS, *(BYTES if byte_tokens else [])]
    vocab = {token: index for index, token in enumerate([*PIECES, *symbols])}
    added = [{"id": vocab[t], "content": t, "special": True, "normalized": False} for t in PIECES]
    data = {
        "added_tokens": added,
        "post_processor": post_processor,
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [["▁", "a"], ["a", "b"], ["▁a", "b"]],
        },
    }
    (directory / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    config = {"add_bos_token": False, "add_eos_token": True, "bos_token": "<s>"}
    config |= {"eos_token": "</s>", "pad_token": "<s>", "unk_token": "<unk>"}
    (directory / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    return vocab


def test_first_segment_gets_a_prefix_and_eos_is_appended(tmp_path):
    vocab = _write(tmp_path)
    tokenizer = AuraFlowTokenizer(tmp_path)

    assert tokenizer.tokenize("") == [2]
    assert tokenizer.tokenize("a") == [vocab["▁a"], 2]
    assert tokenizer.tokenize("a ab") == [vocab["▁a"], vocab["▁ab"], 2]
    assert tokenizer.tokenize(" a") == [vocab["▁a"], 2]
    assert tokenizer.tokenize("  a") == [vocab["▁"], vocab["▁a"], 2]


def test_added_tokens_split_text_and_only_the_first_segment_is_prefixed(tmp_path):
    vocab = _write(tmp_path)
    tokenizer = AuraFlowTokenizer(tmp_path)

    assert tokenizer.tokenize("<x>a") == [vocab["<x>"], vocab["a"], 2]
    assert tokenizer.tokenize("a<x>a b") == [
        vocab["▁a"],
        vocab["<x>"],
        vocab["a"],
        vocab["▁"],
        vocab["b"],
        2,
    ]
    assert tokenizer.tokenize("<x>") == [vocab["<x>"], 2]


def test_special_tokens_come_from_the_post_processor_not_the_config(tmp_path):
    vocab = _write(tmp_path, post_processor=None)
    tokenizer = AuraFlowTokenizer(tmp_path)
    assert tokenizer.tokenize("a") == [vocab["▁a"]]


def test_unknown_characters_use_bytes_or_are_dropped(tmp_path):
    vocab = _write(tmp_path)
    assert AuraFlowTokenizer(tmp_path).tokenize("a é") == [
        vocab["▁a"],
        vocab["▁"],
        vocab["<0xC3>"],
        vocab["<0xA9>"],
        2,
    ]
    other = tmp_path / "nobytes"
    other.mkdir()
    vocab = _write(other, byte_tokens=False)
    assert AuraFlowTokenizer(other).tokenize("a é") == [vocab["▁a"], vocab["▁"], 2]


def test_truncation_keeps_the_final_eos(tmp_path):
    vocab = _write(tmp_path)
    tokenizer = AuraFlowTokenizer(tmp_path)
    assert tokenizer.tokenize("a ab a ab", max_length=3) == [vocab["▁a"], vocab["▁ab"], 2]
    assert tokenizer.tokenize("a", max_length=1) == [2]


def test_right_padding_with_the_pad_token_by_default(tmp_path):
    vocab = _write(tmp_path)
    tokenizer = AuraFlowTokenizer(tmp_path)
    ids, mask = tokenizer(["a", "a ab"], max_length=5)
    one = vocab["<s>"]
    assert ids.tolist() == [
        [vocab["▁a"], 2, one, one, one],
        [vocab["▁a"], vocab["▁ab"], 2, one, one],
    ]
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, True, False, False, False], [True, True, True, False, False]]


def test_unsupported_post_processor_is_rejected(tmp_path):
    _write(tmp_path, post_processor={"type": "ByteLevel"})
    with pytest.raises(ValueError, match="post-processor"):
        AuraFlowTokenizer(tmp_path)
