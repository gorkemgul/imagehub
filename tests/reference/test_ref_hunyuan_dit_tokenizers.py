import random

import pytest
import torch
from helpers import snapshot_dir

from imagehub.tokenization_bert import BertTokenizer
from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

FIXED_PROMPTS = [
    "",
    "a cute cat on the grass",
    "一只可爱的猫在草地上",
    "一只 cute 猫 sitting on 绿色 grass",
    "  leading and trailing  ",
    "Bir kedi, İstanbul'da ĞÜŞİÖÇ ığüşöç",
    "Ελληνικά Русский текст",
    "日本語のテキスト、中文和한국어",
    "emoji 🐱🚀 and   multiple\tspaces\nnewline",
    "numbers 1234567890 3.14 1e-5 ½ ² Ⅷ ٣",
    "ＡＢＣ１２３，ﬁ ⁴ ① ㎒ ㍿",
    "punctuation: !?.,;:'\"()[]{}—…",
    "x[CLS]y [SEP] [MASK] [PAD] [UNK]",
    "x<pad>y </s> <unk>",
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
    "一只可爱的猫",
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
    "!?.,;:'\"()[]{}—…",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "[PAD]",
    "[UNK]",
    "<pad>",
    "</s>",
    "<unk>",
]


def _assert_batch_matches(ours, reference, prompts, max_length):
    actual_ids, actual_mask = ours(prompts, max_length=max_length)
    expected = reference(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    assert torch.equal(actual_ids, expected["input_ids"])
    assert torch.equal(actual_mask, expected["attention_mask"].bool())


def test_hunyuan_dit_tokenizers_fixed_prompts_match_reference(record):
    root = snapshot_dir("hunyuan-dit-v1.1-distilled")
    ours_bert = BertTokenizer(root / "tokenizer")
    reference_bert = transformers.BertTokenizer.from_pretrained(root / "tokenizer")
    ours_t5 = T5Tokenizer(root / "tokenizer_2")
    reference_t5 = transformers.T5Tokenizer.from_pretrained(root / "tokenizer_2")

    _assert_batch_matches(ours_bert, reference_bert, FIXED_PROMPTS, max_length=77)
    _assert_batch_matches(ours_t5, reference_t5, FIXED_PROMPTS, max_length=256)
    record(
        "hunyuan_dit_tokenizers",
        prompts=len(FIXED_PROMPTS),
        mismatches=0,
        tolerance="exact",
    )


def test_hunyuan_dit_tokenizers_seeded_fuzz_match_reference(record):
    root = snapshot_dir("hunyuan-dit-v1.1-distilled")
    ours_bert = BertTokenizer(root / "tokenizer")
    reference_bert = transformers.BertTokenizer.from_pretrained(root / "tokenizer")
    ours_t5 = T5Tokenizer(root / "tokenizer_2")
    reference_t5 = transformers.T5Tokenizer.from_pretrained(root / "tokenizer_2")
    rng = random.Random(0)
    prompts = [
        "".join(rng.choice(FUZZ_PIECES) for _ in range(rng.randint(0, 18))) for _ in range(2500)
    ]

    for start in range(0, len(prompts), 100):
        batch = prompts[start : start + 100]
        _assert_batch_matches(ours_bert, reference_bert, batch, max_length=77)
        _assert_batch_matches(ours_t5, reference_t5, batch, max_length=256)
    record(
        "hunyuan_dit_tokenizers_fuzz",
        prompts=len(prompts),
        mismatches=0,
        tolerance="exact",
    )
