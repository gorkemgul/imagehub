"""AuraFlow tokenizer parity: a synthetic Llama-style folder and the published folder."""

import json
import random

import pytest
import torch
from helpers import snapshot_dir

from imagehub.tokenization_auraflow import AuraFlowTokenizer

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
    "Bir kedi, İstanbul'da ĞÜŞİÖÇ ığüşöç café naïve é",
    "日本語のテキスト与中文和한국어",
    "emoji 🐱🚀 and \U0010ffff ퟿",
    "<extra_id_0> sentinel <extra_id_99>text<extra_id_10><extra_id_1>",
    "</s> inside <s> and <unk><pad>",
    "▁ already▁spaced ▁▁",
    "word " * 400,
]
FUZZ_PIECES = [
    "a",
    "Cat",
    "İstanbul",
    "日本",
    "한국어",
    "🐱",
    "1234567",
    "é",
    "'s",
    "don't",
    " ",
    "   ",
    "\t",
    "\n",
    "\r\n",
    "!?.",
    "▁",
    "<extra_id_0>",
    "<extra_id_10>",
    "<extra_id_99>",
    "</s>",
    "<s>",
    "<unk>",
]


def _check(ours, reference, prompts, max_length):
    actual_ids, actual_mask = ours(prompts, max_length=max_length)
    expected = reference(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    assert torch.equal(actual_ids, expected.input_ids)
    assert torch.equal(actual_mask, expected.attention_mask.bool())
    for prompt in prompts:
        assert ours.tokenize(prompt) == reference(prompt).input_ids
        short = reference(prompt, truncation=True, max_length=10).input_ids
        assert ours.tokenize(prompt, max_length=10) == short


def _fuzz_prompts(count):
    rng = random.Random(0)
    return [
        "".join(rng.choice(FUZZ_PIECES) for _ in range(rng.randint(0, 18))) for _ in range(count)
    ]


def _run(path, record, name):
    ours = AuraFlowTokenizer(path)
    reference = transformers.AutoTokenizer.from_pretrained(path)
    assert reference.padding_side == "right"
    for max_length in (256, 10):
        _check(ours, reference, FIXED_PROMPTS, max_length)
    prompts = _fuzz_prompts(600)
    for start in range(0, len(prompts), 100):
        _check(ours, reference, prompts[start : start + 100], 64)
    record(name, prompts=len(FIXED_PROMPTS) + len(prompts), mismatches=0, tolerance="exact")


def test_auraflow_tokenizer_matches_published_tokenizer(record):
    _run(snapshot_dir("auraflow-v0.3") / "tokenizer", record, "auraflow_tokenizer")


def _write_synthetic(directory):
    """A small Llama-style BPE folder with the published configuration."""
    from tokenizers import AddedToken, Tokenizer, pre_tokenizers, processors
    from tokenizers.models import BPE

    sentinels = [f"<extra_id_{i}>" for i in range(99, -1, -1)]
    letters = list("abcdeinorstl ") + ["é", "İ"]
    symbols = ["▁" if c == " " else c for c in letters] + ["."]
    merges = [
        ("▁", "a"),
        ("a", "b"),
        ("▁a", "b"),
        ("c", "a"),
        ("ca", "t"),
        ("▁", "c"),
        ("a", "t"),
        ("▁c", "at"),
    ]
    merged = ["".join(pair) for pair in merges]
    pieces = ["<unk>", "<s>", "</s>"]
    vocab = {t: i for i, t in enumerate([*pieces, *dict.fromkeys(symbols + merged)])}
    for byte in range(256):
        vocab[f"<0x{byte:02X}>"] = len(vocab)
    base = len(vocab)
    vocab.update({token: base + i for i, token in enumerate(sentinels)})
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, fuse_unk=True, byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="first", split=False
    )
    tokenizer.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in pieces])
    tokenizer.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in sentinels])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A </s>", pair="$A </s> $B </s>", special_tokens=[("</s>", 2)]
    )
    tokenizer.save(str(directory / "tokenizer.json"))
    config = {
        "tokenizer_class": "LlamaTokenizer",
        "add_bos_token": False,
        "add_eos_token": True,
        "add_prefix_space": True,
        "legacy": False,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<s>",
        "padding_side": "right",
        "additional_special_tokens": sentinels,
        "model_max_length": 512,
    }
    (directory / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")


def test_auraflow_tokenizer_matches_reference_on_synthetic_folder(tmp_path, record):
    pytest.importorskip("tokenizers")
    _write_synthetic(tmp_path)
    _run(tmp_path, record, "auraflow_tokenizer_synthetic")
