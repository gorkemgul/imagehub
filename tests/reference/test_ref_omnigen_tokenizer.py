"""OmniGen tokenizer parity with transformers' Llama tokenizer on the published files."""

import random
from pathlib import Path

import pytest

from imagehub.tokenization_omnigen import OmniGenTokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

FILES = Path(__file__).parents[1] / "fixtures" / "omnigen_tokenizer"
FIXED_PROMPTS = [
    "",
    " ",
    "a cat",
    "A photo of a cat, sitting on a red sofa; it's 3.14 o'clock!",
    "multiple   spaces and    more",
    "line one\nline two\n\n\nline four\r\n\ttabbed",
    "  leading and trailing  ",
    "Bir kedi, İstanbul'da ĞÜŞİÖÇ ığüşöç café naïve é",
    "日本語のテキスト与中文和한국어",
    "emoji 🐱🚀 and \U0010ffff ퟿",
    "<|user|>\nGenerate an image according to the following instructions\nA cat<|end|>\n"
    "<|assistant|>\n<|diffusion|>",
    "a <img>",
    " </img> hi",
    "<img><|image_1|></img>",
    "x</s>  y<|assistant|>   z<|placeholder3|>\n\nw",
    "<s><s> <|endoftext|><unk> <|system|>",
    "word " * 400,
]
FUZZ_PIECES = [
    "a",
    "Cat",
    "İstanbul",
    "日本",
    "🐱",
    "1234567",
    "é",
    "'s",
    " ",
    "   ",
    "\t",
    "\n",
    "\n\n",
    "\r\n",
    " ",
    " ",
    "!?.",
    "▁",
    "<|user|>",
    "<|end|>",
    "<|assistant|>",
    "<|system|>",
    "<|diffusion|>",
    "<img>",
    "</img>",
    "<|image_1|>",
    "</s>",
    "<s>",
    "<|endoftext|>",
    "<|placeholder1|>",
    "<|image|>",
    "<cfg>",
]


def _loaded():
    return OmniGenTokenizer(FILES), transformers.LlamaTokenizer.from_pretrained(FILES)


def test_omnigen_tokenizer_fixed_prompts_match_reference(record):
    ours, reference = _loaded()
    for prompt in FIXED_PROMPTS:
        assert ours.tokenize(prompt) == reference(prompt).input_ids, prompt
    record("omnigen_tokenizer", prompts=len(FIXED_PROMPTS), mismatches=0, tolerance="exact")


def test_omnigen_tokenizer_seeded_fuzz_matches_reference(record):
    ours, reference = _loaded()
    rng = random.Random(0)
    prompts = [
        "".join(rng.choice(FUZZ_PIECES) for _ in range(rng.randint(0, 18))) for _ in range(1500)
    ]
    for prompt in prompts:
        assert ours.tokenize(prompt) == reference(prompt).input_ids, repr(prompt)
    record("omnigen_tokenizer_fuzz", prompts=len(prompts), mismatches=0, tolerance="exact")


def test_omnigen_tokenizer_random_unicode_matches_reference():
    ours, reference = _loaded()
    rng = random.Random(1)
    for _ in range(300):
        text = "".join(
            chr(rng.choice([rng.randint(32, 126), rng.randint(0xA0, 0x2FFF), rng.randint(0x1F300, 0x1F64F)]))
            for _ in range(rng.randint(0, 20))
        )
        assert ours.tokenize(text) == reference(text).input_ids, repr(text)
