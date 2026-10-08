import json

import pytest
import torch

from imagehub.tokenization_gemma import GemmaTokenizer

PIECES = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<2>"]
SYMBOLS = ["▁", "a", "b", "c", "ab", "▁a", "▁ab", "abc", "\n", "é"]
BYTES = [f"<0x{byte:02X}>" for byte in range(256)]


def _write(directory, **config):
    vocab = {token: index for index, token in enumerate([*PIECES, *SYMBOLS, *BYTES])}
    merges = [["a", "b"], ["▁", "a"], ["▁a", "b"], ["ab", "c"]]
    added = [
        {
            "id": vocab[token],
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": index < 4,
        }
        for index, token in enumerate(PIECES)
    ]
    data = {
        "added_tokens": added,
        "model": {
            "type": "BPE",
            "unk_token": "<unk>",
            "fuse_unk": True,
            "byte_fallback": True,
            "vocab": vocab,
            "merges": merges,
        },
    }
    (directory / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    base = {"add_bos_token": True, "add_eos_token": False}
    base |= {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}
    (directory / "tokenizer_config.json").write_text(json.dumps(base | config), encoding="utf-8")
    return vocab


def test_spaces_become_markers_and_merges_follow_rank(tmp_path):
    vocab = _write(tmp_path)
    tokenizer = GemmaTokenizer(tmp_path)

    assert tokenizer.tokenize("") == [vocab["<bos>"]]
    assert tokenizer.tokenize("abc") == [2, vocab["abc"]]
    assert tokenizer.tokenize(" ab") == [2, vocab["▁"], vocab["ab"]]
    assert tokenizer.tokenize(" a") == [2, vocab["▁a"]]
    assert tokenizer.tokenize("a b") == [2, vocab["a"], vocab["▁"], vocab["b"]]
    assert tokenizer.tokenize("  a") == [2, vocab["▁"], vocab["▁a"]]
    assert tokenizer.tokenize("\n") == [2, vocab["\n"]]


def test_byte_fallback_and_special_text(tmp_path):
    vocab = _write(tmp_path)
    tokenizer = GemmaTokenizer(tmp_path)

    assert tokenizer.tokenize("é") == [2, vocab["é"]]
    assert tokenizer.tokenize("日") == [2, vocab["<0xE6>"], vocab["<0x97>"], vocab["<0xA5>"]]
    assert tokenizer.tokenize("a<eos>b") == [2, vocab["a"], vocab["<eos>"], vocab["b"]]
    assert tokenizer.tokenize("<mask><2>") == [2, vocab["<mask>"], vocab["<2>"]]


def test_bos_and_eos_flags(tmp_path):
    vocab = _write(tmp_path, add_bos_token=False, add_eos_token=True)
    tokenizer = GemmaTokenizer(tmp_path)

    assert tokenizer.tokenize("ab") == [vocab["ab"], vocab["<eos>"]]
    assert tokenizer.tokenize("ab", max_length=1) == [vocab["<eos>"]]


@pytest.mark.parametrize("side", ["left", "right"])
def test_padding_truncation_and_mask(tmp_path, side):
    vocab = _write(tmp_path)
    tokenizer = GemmaTokenizer(tmp_path)

    ids, mask = tokenizer(["ab", "abc abc abc"], max_length=4, padding_side=side)

    assert ids.dtype == torch.long
    assert mask.dtype == torch.bool
    bos, pad, ab = vocab["<bos>"], vocab["<pad>"], vocab["ab"]
    if side == "left":
        assert ids[0].tolist() == [pad, pad, bos, ab]
        assert mask[0].tolist() == [False, False, True, True]
    else:
        assert ids[0].tolist() == [bos, ab, pad, pad]
        assert mask[0].tolist() == [True, True, False, False]
    assert ids[1].tolist()[0] == bos
    assert mask[1].tolist() == [True] * 4
