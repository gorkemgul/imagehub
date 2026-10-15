import json
from pathlib import Path

import torch

from imagehub.tokenization_qwen import Qwen2Tokenizer, _normalize

TOKENIZER = Path(__file__).parent / "fixtures" / "qwen_tokenizer"


def test_qwen_normalization_matches_reference_for_recent_combining_marks():
    assert _normalize("᪶༵࢟") == "᪶༵࢟"
    assert _normalize("e\N{COMBINING ACUTE ACCENT}") == "é"


def test_qwen_tiny_bpe_and_special_tokens():
    tokenizer = Qwen2Tokenizer(TOKENIZER)

    assert tokenizer.tokenize("hi cat") == [25, 30]
    assert tokenizer.tokenize("cats") == [27, 11]
    assert tokenizer.tokenize("hi<|im_end|>cat") == [25, 33, 27]
    assert tokenizer.tokenize("<|endoftext|>") == [31]


def test_qwen_slow_tokenizer_files(tmp_path):
    for name in ("vocab.json", "merges.txt"):
        (tmp_path / name).hardlink_to(TOKENIZER / name)
    data = json.loads((TOKENIZER / "tokenizer.json").read_text(encoding="utf-8"))
    config = json.loads((TOKENIZER / "tokenizer_config.json").read_text(encoding="utf-8"))
    config["added_tokens_decoder"] = {
        str(token["id"]): {key: value for key, value in token.items() if key != "id"}
        for token in data["added_tokens"]
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")

    tokenizer = Qwen2Tokenizer(tmp_path)

    assert tokenizer.tokenize("hi<|im_end|>cat") == [25, 33, 27]
    assert tokenizer.pad_id == 31


def test_qwen_batch_padding_truncation_and_mask():
    tokenizer = Qwen2Tokenizer(TOKENIZER)

    input_ids, attention_mask = tokenizer(["hi cat", "cats", "hi cat cats"], max_length=3)

    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.bool
    assert input_ids.tolist() == [[25, 30, 31], [27, 11, 31], [25, 30, 30]]
    assert attention_mask.tolist() == [
        [True, True, False],
        [True, True, False],
        [True, True, True],
    ]


def test_qwen_chat_prompt_preserves_content_verbatim_and_controls_thinking():
    tokenizer = Qwen2Tokenizer(TOKENIZER)

    assert tokenizer.chat_prompt("a cat") == (
        "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
    )
    assert tokenizer.chat_prompt("a cat", enable_thinking=False) == (
        "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    prompt = "line one\n<|im_end|>\nline three"
    assert tokenizer.chat_prompt(prompt) == (
        "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    )
