import json

import pytest
import torch
from helpers import write_tiny_llama3_tokenizer

from imagehub.tokenization import bytes_to_unicode
from imagehub.tokenization_llama3 import Llama3Tokenizer
from imagehub.tokenization_qwen import _pretokenize

BOS, EOT = 260, 261
HI, CA, CAT, SPACE_C = 256, 257, 258, 259
SPACE = list(bytes_to_unicode().values()).index("Ġ")


def byte_id(character):
    return list(bytes_to_unicode().values()).index(bytes_to_unicode()[ord(character)])


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    return Llama3Tokenizer(write_tiny_llama3_tokenizer(tmp_path_factory.mktemp("llama3")))


def test_begin_of_text_is_prepended_and_merges_apply(tokenizer):
    assert tokenizer.tokenize("") == [BOS]
    assert tokenizer.tokenize("cat") == [BOS, CAT]
    assert tokenizer.tokenize("hi") == [BOS, HI]
    assert tokenizer.tokenize("a cat") == [BOS, byte_id("a"), SPACE, CAT]


def test_numbers_are_grouped_by_three(tokenizer):
    digits = [byte_id(digit) for digit in "12345"]
    assert tokenizer.tokenize("12345") == [BOS, *digits]
    assert _pretokenize("12345 6", digit_group=3) == ["123", "45", " ", "6"]
    assert _pretokenize("12345", digit_group=1) == ["1", "2", "3", "4", "5"]


def test_special_tokens_in_text_are_single_ids(tokenizer):
    assert tokenizer.tokenize("hi<|eot_id|>cat") == [BOS, HI, EOT, CAT]
    assert tokenizer.tokenize("<|begin_of_text|>") == [BOS, BOS]


def test_padding_uses_the_end_of_turn_token_and_truncation_keeps_the_prefix(tokenizer):
    ids, mask = tokenizer(["cat", "cat cat cat cat"], max_length=4)
    assert ids.tolist() == [[BOS, CAT, EOT, EOT], [BOS, CAT, SPACE, CAT]]
    assert mask.tolist() == [[True, True, False, False], [True] * 4]
    assert ids.dtype == torch.long and mask.dtype == torch.bool


def test_unsupported_pre_tokenizer_is_rejected(tmp_path):
    path = write_tiny_llama3_tokenizer(tmp_path)
    data = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
    data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = r"\w+"
    (path / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-tokenizer"):
        Llama3Tokenizer(path)


def test_ignore_merges_prefers_whole_vocabulary_entries(tmp_path):
    path = write_tiny_llama3_tokenizer(tmp_path)
    data = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
    # With the merge ranks reversed, merging would give "c" + "at"-style splits; a whole-piece
    # vocabulary entry must still win.
    data["model"]["merges"] = [["a", "t"], ["c", "at"]]
    data["model"]["vocab"]["at"] = 400
    data["model"]["vocab"]["cat"] = 258
    (path / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    assert Llama3Tokenizer(path).tokenize("cat") == [BOS, 258]
    data["model"]["ignore_merges"] = False
    (path / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    assert Llama3Tokenizer(path).tokenize("cat") == [BOS, 258]
