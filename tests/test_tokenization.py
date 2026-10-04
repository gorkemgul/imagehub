import json

import pytest
from helpers import write_tiny_tokenizer

from imagehub.tokenization import (
    CLIPTokenizer,
    bytes_to_unicode,
    normalize,
    pretokenize,
    special_token_flags,
)


def test_bytes_to_unicode_is_a_bijection():
    table = bytes_to_unicode()
    assert len(table) == 256 and len(set(table.values())) == 256
    assert table[ord("a")] == "a" and table[ord(" ")] == "Ġ"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello   World\t\nX", "hello world x"),
        ("ΣΟΦΟΣ", "σοφοσ"),  # per-character lowercase: no final-sigma rule
        ("é", "é"),  # NFC
        ("İ", "i̇"),
    ],
)
def test_normalize(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "don't stop!!! x2 <|endoftext|>",
            ["don", "'t", "stop", "!!!", "x", "2", "<|", "endoftext", "|>"],
        ),
        ("a1b22", ["a", "1", "b", "2", "2"]),
        ("hi...there", ["hi", "...", "there"]),
        ("x!<|endoftext|>y", ["x", "!<|", "endoftext", "|>", "y"]),
        ("'''s", ["'''", "s"]),
        ("<|endoftext|>🐱", ["<|", "endoftext", "|>", "🐱"]),
        ("<|startoftext|>'s", ["<|", "startoftext", "|>", "'s"]),
        ("<|endoftext|>|>", ["<|", "endoftext", "|>", "|>"]),
        ("<|endoftext|><|startoftext|>", ["<|", "endoftext", "|>", "<|", "startoftext", "|>"]),
        ("<|endofte", ["<|", "endofte"]),
        ("日本語 テスト", ["日本語", "テスト"]),
    ],
)
def test_pretokenize(text, expected):
    assert pretokenize(normalize(text)) == expected


def test_tiny_vocab_bpe(tmp_path):
    tok = CLIPTokenizer(write_tiny_tokenizer(tmp_path), max_length=8)
    assert (tok.bos_id, tok.eos_id, tok.pad_id) == (515, 516, 516)
    assert tok.tokenize("hi cat") == [512, 514]
    assert tok.tokenize("cats") == [513, 83, 338]
    assert tok.tokenize("Hi CAT!") == [512, 514, 256]
    assert tok("hi cat") == [515, 512, 514, 516, 516, 516, 516, 516]
    assert tok("a b c d e f g h") == [515, 320, 321, 322, 323, 324, 325, 516]
    assert tok("") == [515, 516, 516, 516, 516, 516, 516, 516]


def test_specials_are_exact_case_on_raw_text(tmp_path):
    tok = CLIPTokenizer(write_tiny_tokenizer(tmp_path), max_length=40)
    assert tok.tokenize("x<|endoftext|>y") == [343, 516, 344]
    assert tok.tokenize("<|startoftext|>hi") == [515, 512]
    assert tok.tokenize("x <|ENDOFTEXT|> y") == [
        343, 27, 347, 68, 77, 67, 78, 69, 83, 68, 87, 339, 91, 285, 344,
    ]  # fmt: skip
    assert tok.tokenize("<|StartOfText|>hi") == [
        27, 347, 82, 83, 64, 81, 83, 78, 69, 83, 68, 87, 339, 91, 285, 512,
    ]  # fmt: skip


BOS_T, EOS_T = "<|startoftext|>", "<|endoftext|>"


def _tiny_with_config(tmp_path, config):
    path = write_tiny_tokenizer(tmp_path)
    (path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    return path


def test_flags_plain_strings_are_not_normalized(tmp_path):
    path = write_tiny_tokenizer(tmp_path)
    assert special_token_flags(path) == {BOS_T: False, EOS_T: False}


def test_flags_follow_dicts_and_last_registration_wins(tmp_path):
    sd1_like = {
        "unk_token": {"content": BOS_T, "normalized": True},
        "bos_token": {"content": BOS_T, "normalized": True},
        "eos_token": {"content": EOS_T, "normalized": True},
        "pad_token": EOS_T,
    }
    path = _tiny_with_config(tmp_path / "a", sd1_like)
    assert special_token_flags(path) == {BOS_T: True, EOS_T: False}  # pad (plain) wins for eos
    shared = {"unk_token": {"content": EOS_T, "normalized": True}, "eos_token": EOS_T}
    path = _tiny_with_config(tmp_path / "b", shared)
    assert special_token_flags(path)[EOS_T] is False


def test_flags_fall_back_to_special_tokens_map_then_default(tmp_path):
    path = write_tiny_tokenizer(tmp_path)
    (path / "tokenizer_config.json").unlink()
    (path / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": {"content": BOS_T, "normalized": True}}), encoding="utf-8"
    )
    assert special_token_flags(path) == {BOS_T: True, EOS_T: False}
    (path / "special_tokens_map.json").unlink()
    assert special_token_flags(path) == {BOS_T: False, EOS_T: False}


def test_normalized_bos_matches_after_lowercasing(tmp_path):
    config = {
        "bos_token": {"content": BOS_T, "normalized": True},
        "eos_token": EOS_T,
        "pad_token": EOS_T,
        "unk_token": EOS_T,
    }
    tok = CLIPTokenizer(_tiny_with_config(tmp_path, config), max_length=40)
    assert tok.tokenize("<|StartOfText|>hi") == [515, 512]
    assert tok.tokenize("<|startoftext|>hi") == [515, 512]
    assert tok.tokenize("a<|STARTOFTEXT|>b") == [320, 515, 321]
    assert tok.tokenize("<|STARTOFTEXT|><|endoftext|>") == [515, 516]
    # eos stays raw-only: the uppercase variant is ordinary text
    assert 516 not in tok.tokenize("x <|ENDOFTEXT|> y")
    assert tok.tokenize("x<|endoftext|>y") == [343, 516, 344]
