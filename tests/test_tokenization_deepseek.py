import torch
from fixture_files import fixture_path

from imagehub.tokenization_deepseek import DeepSeekTokenizer

TOKENIZER = fixture_path("deepseek_tokenizer")


def test_deepseek_tiny_bpe_pretokenizer_and_special_tokens():
    tokenizer = DeepSeekTokenizer(TOKENIZER)

    assert tokenizer.tokenize("hi cat") == [20, 2, 9]
    assert tokenizer.tokenize("hi!\n12中") == [20, 2, 10, 11, 12, 13, 14, 15, 16]
    assert tokenizer.tokenize("hi<begin_of_image>cat") == [20, 2, 23, 4, 5, 6]
    assert tokenizer.tokenize("<｜begin▁of▁sentence｜>") == [20, 20]


def test_deepseek_batch_uses_dynamic_left_padding_and_boolean_mask():
    tokenizer = DeepSeekTokenizer(TOKENIZER)

    input_ids, attention_mask = tokenizer(["hi", "hi cat"])

    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.bool
    assert input_ids.tolist() == [[22, 20, 2], [20, 2, 9]]
    assert attention_mask.tolist() == [[False, True, True], [True, True, True]]


def test_deepseek_image_prompt_strips_outer_whitespace():
    tokenizer = DeepSeekTokenizer(TOKENIZER)

    assert tokenizer.image_prompt("\n  a cat \t") == (
        "<|User|>: a cat\n\n<|Assistant|>:<begin_of_image>"
    )
    assert tokenizer.image_prompt(" \n\t ") == ("<|User|>: \n\n<|Assistant|>:<begin_of_image>")
