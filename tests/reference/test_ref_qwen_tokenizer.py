import random

import pytest
import torch
from helpers import snapshot_dir

from imagehub.tokenization_qwen import Qwen2Tokenizer, _normalize

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

FIXED_PROMPTS = [
    "a cat",
    "",
    "Bir kedi, İstanbul'da ĞÜŞİÖÇ ığüşöç",
    "日本語のテキストと中文和한국어",
    "emoji 🐱🚀 and   multiple\tspaces\nnewline",
    "DON'T Don't don't they're I'VE I've i'm we'll he'd",
    "numbers 1234567890 3.14 1e-5 ½ ² Ⅷ ٣",
    "x<|im_end|>y <|endoftext|> <|im_start|>",
    "<|im_starp|> <|im_end> <|endoftext|x>",
    " \r\n\n\t trailing   ",
    "word " * 200,
]

FUZZ_PIECES = [
    "a",
    "Cat",
    "İstanbul",
    "ΣΟΦΟΣ",
    "日本",
    "中文",
    "한국어",
    "🐱",
    "🚀",
    "123456",
    "١٢٣",
    "Ⅷ½",
    "'s",
    "'T",
    "'RE",
    "'Ve",
    "'M",
    "'LL",
    "'d",
    "don't",
    "WON'T",
    " ",
    "   ",
    "\t",
    "\n",
    "\r\n\n",
    "!?.",
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|im_end>",
    "<|im_starp|>",
    "<|endoftext|x>",
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
    for prompt in prompts:
        expected_chat = reference.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        assert ours.chat_prompt(prompt) == expected_chat


def test_qwen_tokenizer_fixed_prompts_match_reference(record):
    path = snapshot_dir("z-image-turbo") / "tokenizer"
    ours = Qwen2Tokenizer(path)
    reference = transformers.AutoTokenizer.from_pretrained(path)

    _assert_batch_matches(ours, reference, FIXED_PROMPTS, max_length=64)
    for prompt in FIXED_PROMPTS:
        assert ours.tokenize(prompt) == reference.encode(prompt, add_special_tokens=True)
    record("qwen_tokenizer", prompts=len(FIXED_PROMPTS), mismatches=0, tolerance="exact")


def test_qwen_tokenizer_seeded_fuzz_matches_reference(record):
    path = snapshot_dir("z-image-turbo") / "tokenizer"
    ours = Qwen2Tokenizer(path)
    reference = transformers.AutoTokenizer.from_pretrained(path)
    rng = random.Random(0)
    prompts = [
        "".join(rng.choice(FUZZ_PIECES) for _ in range(rng.randint(0, 18))) for _ in range(3000)
    ]

    for start in range(0, len(prompts), 100):
        batch = prompts[start : start + 100]
        _assert_batch_matches(ours, reference, batch, max_length=96)
        for prompt in batch:
            assert ours.tokenize(prompt) == reference.encode(prompt, add_special_tokens=True)
    record("qwen_tokenizer_fuzz", prompts=len(prompts), mismatches=0, tolerance="exact")


def test_qwen_tokenizer_matches_reference_for_every_code_point(record):
    path = snapshot_dir("z-image-turbo") / "tokenizer"
    ours = Qwen2Tokenizer(path)
    reference = transformers.AutoTokenizer.from_pretrained(path)
    reference_normalize = reference.backend_tokenizer.normalizer.normalize_str
    normalization_mismatches = 0
    tokenization_mismatches = 0

    for code_point in range(0x20, 0x110000):
        if 0xD800 <= code_point <= 0xDFFF:
            continue
        character = chr(code_point)
        normalization_prompts = [
            character,
            "a" + character,
            character + "̴",
            "̴" + character,
            character + "́",
            "༵" + character + "̴",
            character + "ְ",
        ]
        normalization_mismatches += sum(
            _normalize(prompt) != reference_normalize(prompt) for prompt in normalization_prompts
        )
        tokenization_prompts = [
            "a" + character + "b",
            character + "́",
            " " + character + "1",
            "༵" + character,
        ]
        tokenization_mismatches += sum(
            ours.tokenize(prompt) != reference.encode(prompt, add_special_tokens=True)
            for prompt in tokenization_prompts
        )

    assert normalization_mismatches == 0
    assert tokenization_mismatches == 0
    record(
        "qwen_tokenizer_exhaustive",
        code_points=0x110000 - 0x20 - 0x800,
        normalization_mismatches=normalization_mismatches,
        tokenization_mismatches=tokenization_mismatches,
        tolerance="exact",
    )
