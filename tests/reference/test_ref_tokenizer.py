import json
import random
import shutil
from pathlib import Path

import pytest
from helpers import checkpoint_dir, snapshot_dir, write_tiny_tokenizer

from imagehub.tokenization import CLIPTokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

PROMPTS = [
    "a photo of an astronaut riding a horse on mars",
    "Bir kedi, yağlı boya; İSTANBUL'da ĞÜŞİÖÇ ığüşöç",
    "emoji 🐱🚀 and   multiple\tspaces\nnewline",
    "don't won't they're I've I'm we'll he'd 's '' '''s",
    "numbers 12345 3.14 1e-5 ½ ² Ⅷ ٣",
    "punct!!! ??? ... --- (a) [b] {c} <d> @#$%^&*",
    "x!<|endoftext|>y <|startoftext|> mid",
    "x <|ENDOFTEXT|> y",
    "<|StartOfText|>hi",
    "a b\u3000c d e",
    "Ünïcödé NFC é ﬁ ligature Ωmega ΣΟΦΟΣ",
    "日本語のテキストと中文和한국어",
    "word " * 100,
    "",
]

TURBO_CONFIGS = Path(__file__).parents[1] / "fixtures" / "sdxl_turbo_configs"


def test_tiny_tokenizer_matches_transformers(tmp_path):
    path = write_tiny_tokenizer(tmp_path)
    ours = CLIPTokenizer(path, max_length=8)
    ref = transformers.CLIPTokenizer.from_pretrained(path, model_max_length=8)
    for text in ["hi cat", "cats", "Hi CAT!", "a b c d e f g h", ""]:
        expected = ref(text, padding="max_length", max_length=8, truncation=True)["input_ids"]
        assert ours(text) == expected, text


def test_tiny_tokenizer_special_case_matches_transformers(tmp_path):
    path = write_tiny_tokenizer(tmp_path)
    ours = CLIPTokenizer(path, max_length=40)
    ref = transformers.CLIPTokenizer.from_pretrained(path, model_max_length=40)
    for text in ["x <|ENDOFTEXT|> y", "<|StartOfText|>hi", "x<|endoftext|>y"]:
        expected = ref(text, padding="max_length", max_length=40, truncation=True)["input_ids"]
        assert ours(text) == expected, text


def test_tiny_tokenizer_configured_pad_matches_transformers(tmp_path):
    path = write_tiny_tokenizer(tmp_path)
    config = {
        "unk_token": "<|endoftext|>",
        "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "!",
        "model_max_length": 40,
        "do_lower_case": True,
    }
    (path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "special_tokens_map.json").unlink()
    ours = CLIPTokenizer(path, max_length=40)
    ref = transformers.CLIPTokenizer.from_pretrained(path, model_max_length=40)
    for text in ["hi!!! cat", "a!b", "!!", "x <|endoftext|> y!", "Hi CAT!"]:
        expected = ref(text, padding="max_length", max_length=40, truncation=True)["input_ids"]
        assert ours(text) == expected, text


def test_sd1_tokenizer_matches_transformers(record):
    path = checkpoint_dir(4) / "tokenizer"
    ours = CLIPTokenizer(path)
    ref = transformers.CLIPTokenizer.from_pretrained(path)
    for text in PROMPTS:
        expected = ref(text, padding="max_length", max_length=77, truncation=True)["input_ids"]
        assert ours(text) == expected, text
    record("tokenizer", prompts=len(PROMPTS), mismatches=0, tolerance="exact")


SPECIAL_PROMPTS = [
    "<|StartOfText|>hi",
    "x <|EndOfText|> y",
    "<|STARTOFTEXT|><|endoftext|>",
    "a<|startoftext|>b",
]


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_sd1_tokenizers_match_transformers_exhaustively(version):
    path = checkpoint_dir(version) / "tokenizer"
    ours = CLIPTokenizer(path)
    ref = transformers.CLIPTokenizer.from_pretrained(path)
    for text in [*PROMPTS, *SPECIAL_PROMPTS]:
        expected = ref(text, padding="max_length", max_length=77, truncation=True)["input_ids"]
        assert ours(text) == expected, (version, text)


FUZZ_PIECES = [
    "<|startoftext|>", "<|endoftext|>", "<|StartOfText|>", "<|ENDOFTEXT|>", "<|EndOfText|>",
    "<|STARTOFTEXT|>", "a", "Cat", "'s", " ", "  ", "!", "é", "İ", "日本", "🐱", " ", "12",
    "x<|", "|>", "don't", "'''s", "é",
]  # fmt: skip


def _fuzz(ours, ref, count=1000):
    rng = random.Random(0)
    for _ in range(count):
        text = "".join(rng.choice(FUZZ_PIECES) for _ in range(rng.randint(1, 12)))
        expected = ref(text, padding="max_length", max_length=77, truncation=True)["input_ids"]
        assert ours(text) == expected, f"first mismatching prompt: {text!r}"


def test_tiny_tokenizer_fuzz_matches_transformers(tmp_path):
    path = write_tiny_tokenizer(tmp_path)
    ours = CLIPTokenizer(path, max_length=77)
    ref = transformers.CLIPTokenizer.from_pretrained(path, model_max_length=77)
    _fuzz(ours, ref)


@pytest.mark.parametrize("name", ["tokenizer", "tokenizer_2"])
def test_sdxl_turbo_fixture_tokenizers_fuzz_match_transformers(tmp_path, name):
    source = snapshot_dir("sdxl-turbo") / name
    path = tmp_path / name
    path.mkdir()
    for filename in ("vocab.json", "merges.txt"):
        shutil.copyfile(source / filename, path / filename)
    for filename in ("tokenizer_config.json", "special_tokens_map.json"):
        shutil.copyfile(TURBO_CONFIGS / name / filename, path / filename)
    ours = CLIPTokenizer(path)
    ref = transformers.CLIPTokenizer.from_pretrained(path)
    _fuzz(ours, ref)


@pytest.mark.parametrize(
    "config",
    [
        {
            "added_tokens_decoder": {
                "515": {"content": "<|startoftext|>", "normalized": True, "special": True},
                "516": {"content": "<|endoftext|>", "normalized": True, "special": True},
            },
            "unk_token": "<|endoftext|>",
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
        },
        {
            "added_tokens_decoder": {
                "515": {"content": "<|startoftext|>", "normalized": True, "special": True},
                "516": {"content": "<|endoftext|>", "normalized": True, "special": True},
            },
            "unk_token": {
                "__type": "AddedToken",
                "content": "<|endoftext|>",
                "normalized": False,
            },
            "bos_token": "<|startoftext|>",
            "eos_token": {
                "__type": "AddedToken",
                "content": "<|endoftext|>",
                "normalized": False,
            },
            "pad_token": "<|endoftext|>",
        },
        {
            "added_tokens_decoder": {
                "515": {"content": "<|startoftext|>", "normalized": True, "special": True},
            },
            "unk_token": "<|endoftext|>",
            "bos_token": "<|startoftext|>",
            "eos_token": {
                "__type": "AddedToken",
                "content": "<|endoftext|>",
                "normalized": True,
            },
            "pad_token": "<|endoftext|>",
        },
    ],
)
def test_mixed_added_token_decoder_configs_fuzz_match_transformers(tmp_path, config):
    path = write_tiny_tokenizer(tmp_path)
    (path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    ours = CLIPTokenizer(path)
    ref = transformers.CLIPTokenizer.from_pretrained(path)
    _fuzz(ours, ref)


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_sd1_tokenizers_fuzz_matches_transformers(version):
    path = checkpoint_dir(version) / "tokenizer"
    ours = CLIPTokenizer(path)
    ref = transformers.CLIPTokenizer.from_pretrained(path)
    _fuzz(ours, ref)
