import pytest
from helpers import checkpoint_dir, write_tiny_tokenizer

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
    "Ünïcödé NFC é ﬁ ligature Ωmega ΣΟΦΟΣ",
    "日本語のテキストと中文和한국어",
    "word " * 100,
    "",
]


def test_tiny_tokenizer_matches_transformers(tmp_path):
    path = write_tiny_tokenizer(tmp_path)
    ours = CLIPTokenizer(path, max_length=8)
    ref = transformers.CLIPTokenizer.from_pretrained(path, model_max_length=8)
    for text in ["hi cat", "cats", "Hi CAT!", "a b c d e f g h", ""]:
        expected = ref(text, padding="max_length", max_length=8, truncation=True)["input_ids"]
        assert ours(text) == expected, text


def test_sd1_tokenizer_matches_transformers(record):
    path = checkpoint_dir(4) / "tokenizer"
    ours = CLIPTokenizer(path)
    ref = transformers.CLIPTokenizer.from_pretrained(path)
    for text in PROMPTS:
        expected = ref(text, padding="max_length", max_length=77, truncation=True)["input_ids"]
        assert ours(text) == expected, text
    record("tokenizer", prompts=len(PROMPTS), mismatches=0, tolerance="exact")
