"""CogView3-Plus T5 tokenizer parity (the published SentencePiece model, no weights needed)."""

import random
from pathlib import Path

import pytest
import torch

from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

TOKENIZER = Path(__file__).parents[1] / "fixtures" / "cogview3_plus_configs" / "tokenizer"
MAX_LENGTH = 224
FIXED_PROMPTS = [
    "",
    "a cat",
    "A photograph of an astronaut riding a horse on mars, 8k, highly detailed",
    "  leading and trailing  ",
    "Bir kedi, İstanbul'da ĞÜŞİÖÇ ığüşöç",
    "Ελληνικά Русский текст",
    "日本語のテキストと中文和한국어",
    "emoji 🐱🚀 and   multiple\tspaces\nnewline",
    "numbers 1234567890 3.14 1e-5 ½ ² Ⅷ ٣",
    "Ａ ﬁ ⁴ ① ㎒ ㍿",
    "x<extra_id_0>y </s> <extra_id_99>",
    "x<pad>y <unk>",
    "<extra_id_> <extra_id_0x> </sx>",
    "word " * 300,
]
FUZZ_PIECES = [
    "a",
    "Cat",
    "café",
    "ＡＢＣ",
    "ﬁﬂ",
    "²⁴",
    "①ⒶⅧ",
    "İstanbul",
    "Кот",
    "日本",
    "한국어",
    "🐱",
    "123456",
    "١٢٣",
    " ",
    "   ",
    "\t",
    "\n",
    "!?.",
    "<extra_id_0>",
    "<extra_id_99>",
    "</s>",
    "<pad>",
    "<unk>",
    "<extra_id_>",
    "</sx>",
]


@pytest.fixture(scope="module")
def tokenizers():
    return T5Tokenizer(TOKENIZER), transformers.AutoTokenizer.from_pretrained(TOKENIZER)


def test_fixed_prompts_match_the_pipeline_call(record, tokenizers):
    ours, reference = tokenizers
    for max_length in (MAX_LENGTH, 16):
        actual_ids, _ = ours(FIXED_PROMPTS, max_length=max_length)
        expected = reference(
            FIXED_PROMPTS,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        assert torch.equal(actual_ids, expected["input_ids"])
    assert [ours.tokenize(prompt) for prompt in FIXED_PROMPTS] == [
        reference.encode(prompt) for prompt in FIXED_PROMPTS
    ]
    record("cogview3_plus_tokenizer", prompts=len(FIXED_PROMPTS), mismatches=0, tolerance="exact")


def test_seeded_fuzz_matches_reference(record, tokenizers):
    ours, reference = tokenizers
    rng = random.Random(0)
    prompts = [
        "".join(rng.choice(FUZZ_PIECES) for _ in range(rng.randint(0, 18))) for _ in range(3000)
    ]
    for start in range(0, len(prompts), 100):
        batch = prompts[start : start + 100]
        expected = reference(batch, add_special_tokens=True)["input_ids"]
        assert [ours.tokenize(prompt) for prompt in batch] == expected
    record("cogview3_plus_tokenizer_fuzz", prompts=len(prompts), mismatches=0, tolerance="exact")


def test_every_code_point_matches_reference(record, tokenizers):
    ours, reference = tokenizers
    mismatches = checked = 0
    prompts: list[str] = []

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
    record("cogview3_plus_tokenizer_exhaustive", prompts=checked, mismatches=0, tolerance="exact")
