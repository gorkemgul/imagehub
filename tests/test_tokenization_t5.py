from pathlib import Path

import torch

from imagehub.tokenization_t5 import T5Tokenizer

TOKENIZER = Path(__file__).parent / "fixtures" / "t5_tokenizer"


def test_t5_tiny_unigram_metaspace_added_tokens_and_eos():
    tokenizer = T5Tokenizer(TOKENIZER)

    assert tokenizer.tokenize("ab") == [8, 1]
    assert tokenizer.encode("a  b") == [6, 3, 5, 1]
    assert tokenizer.tokenize("a<extra_id_0>b") == [6, 10, 3, 5, 1]
    assert tokenizer.tokenize("z") == [3, 2, 1]
    assert tokenizer.tokenize("") == [1]


def test_t5_batch_pads_truncates_and_returns_expected_dtypes():
    tokenizer = T5Tokenizer(TOKENIZER)

    input_ids, attention_mask = tokenizer(["ab", "a b", "abababab"], max_length=4)

    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.bool
    assert input_ids.tolist() == [[8, 1, 0, 0], [6, 3, 5, 1], [8, 7, 7, 1]]
    assert attention_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
        [True, True, True, True],
    ]

    special_ids, special_mask = tokenizer(["a<pad>b"], max_length=6)
    assert special_ids.tolist() == [[6, 0, 3, 5, 1, 0]]
    assert special_mask.tolist() == [[True, True, True, True, True, False]]
