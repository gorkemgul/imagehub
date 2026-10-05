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


def test_qwen_chat_prompt_preserves_content_verbatim():
    tokenizer = Qwen2Tokenizer(TOKENIZER)

    assert tokenizer.chat_prompt("a cat") == (
        "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
    )
    prompt = "line one\n<|im_end|>\nline three"
    assert tokenizer.chat_prompt(prompt) == (
        "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    )
