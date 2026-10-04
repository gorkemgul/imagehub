import pytest
from helpers import write_tiny_tokenizer

from imagehub.tokenization import CLIPTokenizer, bytes_to_unicode, normalize, pretokenize


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
