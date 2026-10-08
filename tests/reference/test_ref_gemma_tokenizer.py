"""Gemma tokenizer parity (Sana text encoder tokenizer)."""

import random

import pytest
import torch
from helpers import snapshot_dir

from imagehub.tokenization_gemma import GemmaTokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

FIXED_PROMPTS = [
    "",
    "a cat",
    "A photo of a cat, sitting on a red sofa; it's 3.14 o'clock!",
    "numbers 1234567890 3.14 1e-5 ½ ² Ⅷ ٣",
    "multiple   spaces and    more",
    "line one\nline two\n\n\nline four\r\n\ttabbed",
    "  leading and trailing  ",
    " ",
    "Bir kedi, İstanbul'da ĞÜŞİÖÇ ığüşöç café naïve é",
    "日本語のテキスト与中文和한국어",
    "emoji 🐱🚀 and \U0010ffff ퟿",
    "text <eos> inside <bos> and <pad><unk> <start_of_turn>user\nhi<end_of_turn>\n<unused5>",
    "<mask> [@BOS@] <2mass> <table><td>",
    "▁ already▁spaced ▁▁",
    "word " * 400,
]
FUZZ_PIECES = [
    "a",
    "Cat",
    "İstanbul",
    "日本",
    "中文",
    "한국어",
    "🐱",
    "🚀",
    "1234567",
    "١٢٣",
    "é",
    "é",
    "'s",
    "don't",
    " ",
    "   ",
    "\t",
    "\n",
    "\n\n\n",
    "\r\n",
    "!?.",
    "▁",
    "<eos>",
    "<bos>",
    "<pad>",
    "<start_of_turn>",
    "<unused3>",
    "<mask>",
]


def _loaded():
    path = snapshot_dir("sana-1600m-1024") / "tokenizer"
    ours = GemmaTokenizer(path)
    reference = transformers.AutoTokenizer.from_pretrained(path)
    return ours, reference


def _check(ours, reference, prompts, max_length, side):
    reference.padding_side = side
    actual_ids, actual_mask = ours(prompts, max_length=max_length, padding_side=side)
    expected = reference(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    assert torch.equal(actual_ids, expected.input_ids)
    assert torch.equal(actual_mask, expected.attention_mask.bool())
    for prompt in prompts:
        assert ours.tokenize(prompt) == reference.encode(prompt, add_special_tokens=True)
        short = reference.encode(prompt, add_special_tokens=True, truncation=True, max_length=10)
        assert ours.tokenize(prompt, max_length=10) == short


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("max_length", [300, 10])
def test_gemma_tokenizer_fixed_prompts_match_reference(record, side, max_length):
    ours, reference = _loaded()
    _check(ours, reference, FIXED_PROMPTS, max_length, side)
    record("gemma_tokenizer", prompts=len(FIXED_PROMPTS), mismatches=0, tolerance="exact")


def test_gemma_tokenizer_seeded_fuzz_matches_reference(record):
    ours, reference = _loaded()
    rng = random.Random(0)
    prompts = [
        "".join(rng.choice(FUZZ_PIECES) for _ in range(rng.randint(0, 18))) for _ in range(600)
    ]
    for start in range(0, len(prompts), 100):
        _check(ours, reference, prompts[start : start + 100], 64, "right")
    record("gemma_tokenizer_fuzz", prompts=len(prompts), mismatches=0, tolerance="exact")
