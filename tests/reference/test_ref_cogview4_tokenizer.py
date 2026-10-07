"""CogView4 GLM tokenizer parity."""

import random

import pytest
import torch
from helpers import snapshot_dir

from imagehub.tokenization_glm import GlmTokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

FIXED_PROMPTS = [
    "",
    "a cat",
    "Bir kedi, İstanbul'da ĞÜŞİÖÇ ığüşöç",
    "日本語のテキスト与中文和한국어",
    "emoji 🐱🚀 and   multiple\tspaces\nnewline",
    "numbers 1234567890 3.14 1e-5 ½ ² Ⅷ ٣",
    "x[gMASK]y <sop> <|endoftext|>",
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
    "Ⅷ½",
    "'s",
    "'RE",
    "don't",
    " ",
    "   ",
    "\t",
    "\n",
    "\r\n\n",
    "!?.",
    "[gMASK]",
    "<sop>",
    "<|endoftext|>",
]


def _reference_batch(reference, prompts, max_length):
    expected = reference(
        prompts,
        padding="longest",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    pad_length = (-expected.input_ids.shape[1]) % 16
    if pad_length:
        ids = torch.full((len(prompts), pad_length), reference.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(prompts), pad_length), dtype=torch.long)
        expected.input_ids = torch.cat((ids, expected.input_ids), dim=1)
        expected.attention_mask = torch.cat((mask, expected.attention_mask), dim=1)
    return expected


def _assert_batch_matches(ours, reference, prompts, max_length):
    actual_ids, actual_mask = ours(prompts, max_length=max_length)
    expected = _reference_batch(reference, prompts, max_length)
    assert torch.equal(actual_ids, expected.input_ids)
    assert torch.equal(actual_mask, expected.attention_mask.bool())
    for prompt in prompts:
        assert ours.tokenize(prompt) == reference.encode(prompt, add_special_tokens=True)


def test_cogview4_tokenizer_fixed_prompts_match_reference(record):
    path = snapshot_dir("cogview4-6b") / "tokenizer"
    ours = GlmTokenizer(path)
    reference = transformers.PreTrainedTokenizerFast.from_pretrained(path)

    _assert_batch_matches(ours, reference, FIXED_PROMPTS, max_length=1024)
    record("cogview4_tokenizer", prompts=len(FIXED_PROMPTS), mismatches=0, tolerance="exact")


def test_cogview4_tokenizer_seeded_fuzz_matches_reference(record):
    path = snapshot_dir("cogview4-6b") / "tokenizer"
    ours = GlmTokenizer(path)
    reference = transformers.PreTrainedTokenizerFast.from_pretrained(path)
    rng = random.Random(0)
    prompts = [
        "".join(rng.choice(FUZZ_PIECES) for _ in range(rng.randint(0, 18))) for _ in range(2000)
    ]

    for start in range(0, len(prompts), 100):
        _assert_batch_matches(ours, reference, prompts[start : start + 100], max_length=128)
    record("cogview4_tokenizer_fuzz", prompts=len(prompts), mismatches=0, tolerance="exact")
