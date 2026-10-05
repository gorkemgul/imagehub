import os
import random
from pathlib import Path

import pytest
import torch

from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

FIXED_PROMPTS = [
    "",
    "a cat",
    "  leading and trailing  ",
    "Bir kedi, İstanbul'da ĞÜŞİÖÇ ığüşöç",
    "Ελληνικά Русский текст",
    "日本語のテキストと中文和한국어",
    "emoji 🐱🚀 and   multiple\tspaces\nnewline",
    "numbers 1234567890 3.14 1e-5 ½ ² Ⅷ ٣",
    "Ａ ﬁ ⁴ ① ㎒ ㍿",
    "x<extra_id_0>y </s> <extra_id_99>",
    "x<pad>y",
    "<extra_id_> <extra_id_0x> </sx>",
    "word " * 300,
]

FUZZ_PIECES = [
    "a",
    "Cat",
    "café",
    "é",
    "ＡＢＣ",
    "ﬁﬂ",
    "²⁴",
    "①ⒶⅧ",
    "㎒㍿",
    "İstanbul",
    "ΣΟΦΟΣ",
    "Кот",
    "日本",
    "中文",
    "한국어",
    "🐱",
    "🚀",
    "123456",
    "١٢٣",
    " ",
    "   ",
    "\t",
    "\n",
    "\r\n\n",
    "!?.",
    "<extra_id_0>",
    "<extra_id_99>",
    "</s>",
    "<extra_id_>",
    "<extra_id_0x>",
    "</sx>",
]


def _tokenizer_path():
    value = os.environ.get("IMAGEHUB_T5_TOKENIZER")
    if not value:
        pytest.skip("Set IMAGEHUB_T5_TOKENIZER to a local tokenizer snapshot")
    return Path(value)


def test_t5_tokenizer_fixed_prompts_and_batches_match_reference(record):
    path = _tokenizer_path()
    ours = T5Tokenizer(path)
    reference = transformers.AutoTokenizer.from_pretrained(path)

    assert [ours.tokenize(prompt) for prompt in FIXED_PROMPTS] == [
        reference.encode(prompt) for prompt in FIXED_PROMPTS
    ]
    for max_length in (256, 8):
        actual_ids, actual_mask = ours(FIXED_PROMPTS, max_length=max_length)
        expected = reference(
            FIXED_PROMPTS,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        assert torch.equal(actual_ids, expected["input_ids"])
        assert torch.equal(actual_mask, expected["attention_mask"].bool())
    record("t5_tokenizer", prompts=len(FIXED_PROMPTS), mismatches=0, tolerance="exact")


def test_t5_tokenizer_seeded_fuzz_matches_reference(record):
    path = _tokenizer_path()
    ours = T5Tokenizer(path)
    reference = transformers.AutoTokenizer.from_pretrained(path)
    rng = random.Random(0)
    prompts = [
        "".join(rng.choice(FUZZ_PIECES) for _ in range(rng.randint(0, 18))) for _ in range(3000)
    ]

    for start in range(0, len(prompts), 100):
        batch = prompts[start : start + 100]
        expected = reference(batch, add_special_tokens=True)["input_ids"]
        assert [ours.tokenize(prompt) for prompt in batch] == expected
    record("t5_tokenizer_fuzz", prompts=len(prompts), mismatches=0, tolerance="exact")


def test_t5_tokenizer_matches_reference_for_every_code_point(record):
    path = _tokenizer_path()
    ours = T5Tokenizer(path)
    reference = transformers.AutoTokenizer.from_pretrained(path)
    mismatches = 0
    checked = 0
    prompts = []

    def compare_batch():
        nonlocal mismatches, checked
        expected = reference(prompts, add_special_tokens=True)["input_ids"]
        actual = [ours.tokenize(prompt) for prompt in prompts]
        mismatches += sum(left != right for left, right in zip(actual, expected, strict=True))
        checked += len(prompts)
        prompts.clear()

    for code_point in range(0x20, 0x110000):
        if 0xD800 <= code_point <= 0xDFFF:
            continue
        character = chr(code_point)
        prompts.extend(("a" + character + "b", character + " x"))
        if len(prompts) >= 1024:
            compare_batch()
    if prompts:
        compare_batch()

    assert mismatches == 0
    record(
        "t5_tokenizer_exhaustive",
        prompts=checked,
        mismatches=mismatches,
        tolerance="exact",
    )
