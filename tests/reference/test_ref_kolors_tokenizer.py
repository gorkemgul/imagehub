"""Kolors ChatGLM tokenizer parity."""

import random
from pathlib import Path

import pytest
import torch
from helpers import snapshot_dir
from test_tokenization_chatglm import write_model

from imagehub.tokenization_chatglm import ChatGLMTokenizer

pytestmark = pytest.mark.reference
pytest.importorskip("transformers")
sentencepiece = pytest.importorskip("sentencepiece")
reference_module = pytest.importorskip("diffusers.pipelines.kolors.tokenizer")

FIXED_PROMPTS = [
    "",
    "a cat",
    "A photo of an astronaut riding a horse on mars, 4k",
    "一张室内的猫和狗的照片，高清",
    "Bir kedi, İstanbul'da ĞÜŞİÖÇ ığüşöç",
    "日本語のテキストと한국어",
    "emoji \U0001f431\U0001f680 and   multiple\tspaces\nnewline\r\n",
    "numbers 1234567890 3.14 1e-5 ½ ² Ⅷ ٣",
    "x[gMASK]y sop eop <|user|> <|assistant|> <unk> </s> <pad> [MASK]",
    "   leading and trailing   ",
    "word " * 400,
]
FUZZ_PIECES = [
    "a",
    "Cat",
    "猫",
    "中文",
    "日本",
    "\U0001f431",
    "1234567",
    "Ⅷ½",
    "'s",
    " ",
    "   ",
    "\t",
    "\n",
    "\r\n\n",
    "!?.",
    " ",
    "　",
    "▁",
    "[gMASK]",
    "[MASK]",
    "sop",
    "eop",
    "<|user|>",
    "<unk>",
    "</s>",
    "<pad>",
    "<0x0A>",
]


def _assert_matches(ours, reference, prompts, max_length):
    ids, mask, positions = ours(prompts, max_length=max_length)
    expected = reference(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    assert torch.equal(ids, expected["input_ids"])
    assert torch.equal(mask, expected["attention_mask"].bool())
    assert torch.equal(positions, expected["position_ids"])


def _assert_unpadded_ids(ours, reference, prompts):
    for prompt in prompts:
        expected = reference.build_inputs_with_special_tokens(
            reference.convert_tokens_to_ids(reference.tokenize(prompt))
        )
        assert ours.tokenize(prompt) == expected


def _fuzz_prompts(seed, count):
    rng = random.Random(seed)
    return [
        "".join(rng.choice(FUZZ_PIECES) for _ in range(rng.randint(0, 16))) for _ in range(count)
    ]


def test_kolors_tokenizer_fixed_prompts_match_reference(record):
    path = snapshot_dir("kolors") / "tokenizer" / "tokenizer.model"
    ours = ChatGLMTokenizer(path)
    reference = reference_module.ChatGLMTokenizer(str(path))

    _assert_matches(ours, reference, FIXED_PROMPTS, max_length=256)
    _assert_unpadded_ids(ours, reference, FIXED_PROMPTS)
    record("kolors_tokenizer", prompts=len(FIXED_PROMPTS), mismatches=0, tolerance="exact")


def test_kolors_tokenizer_seeded_fuzz_matches_reference(record):
    path = snapshot_dir("kolors") / "tokenizer" / "tokenizer.model"
    ours = ChatGLMTokenizer(path)
    reference = reference_module.ChatGLMTokenizer(str(path))
    prompts = _fuzz_prompts(0, 2000)

    for start in range(0, len(prompts), 100):
        _assert_matches(ours, reference, prompts[start : start + 100], max_length=64)
    record("kolors_tokenizer_fuzz", prompts=len(prompts), mismatches=0, tolerance="exact")


def _train(directory: Path, name: str, **options) -> Path:
    rng = random.Random(0)
    words = "the quick brown fox cat 你好 世界 中文 abc 123 éa ｱ ①".split()
    lines = [" ".join(rng.choice(words) for _ in range(rng.randint(3, 12))) for _ in range(800)]
    prefix = directory / name
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(lines),
        model_prefix=str(prefix),
        model_type="bpe",
        character_coverage=1.0,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        hard_vocab_limit=False,
        **options,
    )
    return prefix.with_suffix(".model")


@pytest.fixture(scope="module")
def small_models(tmp_path_factory):
    directory = tmp_path_factory.mktemp("chatglm_models")
    return {
        "byte": _train(
            directory,
            "byte",
            vocab_size=420,
            byte_fallback=True,
            allow_whitespace_only_pieces=True,
            remove_extra_whitespaces=False,
            normalization_rule_name="identity",
        ),
        "nfkc": _train(directory, "nfkc", vocab_size=420, byte_fallback=True),
        "plain": _train(
            directory,
            "plain",
            vocab_size=170,
            byte_fallback=False,
            normalization_rule_name="identity",
        ),
        "stripped": _train(
            directory,
            "stripped",
            vocab_size=400,
            byte_fallback=True,
            add_dummy_prefix=False,
            normalization_rule_name="nmt_nfkc_cf",
        ),
        "handmade": write_model(directory / "handmade.model"),
    }


@pytest.mark.parametrize("name", ["byte", "nfkc", "plain", "stripped", "handmade"])
def test_small_sentencepiece_models_match_reference(small_models, name):
    path = small_models[name]
    ours = ChatGLMTokenizer(path)
    reference = reference_module.ChatGLMTokenizer(str(path))
    prompts = FIXED_PROMPTS[:-1] + _fuzz_prompts(1, 600)

    for start in range(0, len(prompts), 100):
        _assert_matches(ours, reference, prompts[start : start + 100], max_length=32)
    _assert_unpadded_ids(ours, reference, prompts[:100])
