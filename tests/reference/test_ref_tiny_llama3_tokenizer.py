"""Llama 3 tokenizer against the reference fast tokenizer, on a tiny byte-level vocabulary."""

import random

import pytest
import torch
from helpers import write_tiny_llama3_tokenizer

from imagehub.tokenization_llama3 import Llama3Tokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

TEXTS = [
    "",
    "a cat",
    "hi cat hi",
    "12345678 9",
    "  leading and trailing  ",
    "line1\nline2\r\n\r\n\n x",
    "it's I'M WE'LL they've",
    "<|begin_of_text|>hi<|eot_id|> cat",
    "tabs\t\tand   spaces   ",
    "naïve café 日本語 🐱🐱",
    "cat " * 200,
]


def test_tokens_and_masks_match_the_reference(tmp_path):
    path = write_tiny_llama3_tokenizer(tmp_path)
    reference = transformers.AutoTokenizer.from_pretrained(path)
    reference.pad_token = reference.eos_token
    ours = Llama3Tokenizer(path)
    generator = random.Random(0)
    alphabet = "abc hi cat 0123.,!?'\n\t日é🙂-_"
    texts = TEXTS + [
        "".join(generator.choice(alphabet) for _ in range(generator.randint(0, 60)))
        for _ in range(200)
    ]
    for text in texts:
        for length in (8, 128):
            expected = reference(
                [text],
                padding="max_length",
                max_length=length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            ids, mask = ours([text], length)
            assert torch.equal(ids, expected.input_ids), repr(text)
            assert torch.equal(mask, expected.attention_mask.bool()), repr(text)
