import json

import pytest
from helpers import ERNIE_TINY_PRETOKENIZER_REGEX, write_tiny_ernie_tokenizer

from imagehub.tokenization import bytes_to_unicode
from imagehub.tokenization_mistral import MistralTokenizer, python_regex


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    return MistralTokenizer(write_tiny_ernie_tokenizer(tmp_path_factory.mktemp("tok")))


def _ids(tokenizer, text):
    return tokenizer(text)


def test_bos_is_prepended_and_empty_text_is_only_bos(tokenizer):
    assert _ids(tokenizer, "") == [1]
    assert _ids(tokenizer, "a")[0] == 1


def test_merges_and_ignore_merges(tokenizer):
    vocab = tokenizer.encoder
    assert _ids(tokenizer, "cat") == [1, vocab["cat"]]
    # " the" is reached through merges ("Ġ"+"t", "h"+"e", "Ġt"+"he") and ends as one token.
    assert _ids(tokenizer, " the") == [1, vocab["Ġthe"]]
    assert _ids(tokenizer, "hi")[1:] == [vocab["h"], vocab["i"]]


def test_added_tokens_are_extracted_from_raw_text(tokenizer):
    vocab = tokenizer.encoder
    assert _ids(tokenizer, "[INST]hi[/INST]") == [
        1,
        vocab["[INST]"],
        vocab["h"],
        vocab["i"],
        vocab["[/INST]"],
    ]
    assert _ids(tokenizer, "<s></s>") == [1, 1, 2]


def test_bytes_cover_arbitrary_unicode(tokenizer):
    text = "日本語 🐱\n\n x"
    ids = tokenizer(text)
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    inverse = {i: t for t, i in tokenizer.encoder.items()}
    data = bytes(byte_decoder[ch] for i in ids[1:] for ch in inverse[i])
    assert data.decode("utf-8") == text


def test_truncation_keeps_the_bos_and_the_first_tokens(tokenizer):
    ids = tokenizer("cat " * 100)
    assert len(ids) == tokenizer.max_length == 64
    assert ids[0] == 1
    assert ids[1:] == tokenizer.tokenize("cat " * 100)[1 : 1 + 63]
    assert tokenizer("cat " * 100, max_length=5) == ids[:5]


def test_pretokenizer_regex_matches_the_reference_alternatives():
    pattern = python_regex(ERNIE_TINY_PRETOKENIZER_REGEX)
    text = "Hello,  World42!\n\n  see/x  "
    assert pattern.findall(text) == [
        "Hello",
        ",",
        " ",
        " World",
        "4",
        "2",
        "!\n\n",
        " ",
        " see",
        "/x",
        "  ",
    ]
    assert "".join(pattern.findall(text)) == text


@pytest.mark.parametrize("field", ["normalized", "lstrip", "rstrip", "single_word"])
def test_added_token_flags_beyond_plain_are_rejected(tmp_path, field):
    directory = write_tiny_ernie_tokenizer(tmp_path)
    data = json.loads((directory / "tokenizer.json").read_text())
    data["added_tokens"][0][field] = True
    (directory / "tokenizer.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="plain added tokens"):
        MistralTokenizer(directory)
