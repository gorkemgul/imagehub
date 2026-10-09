import struct
from pathlib import Path

import torch

from imagehub.tokenization_chatglm import ChatGLMTokenizer

NORMAL, UNKNOWN, CONTROL, BYTE = 1, 2, 3, 6
PIECES = [
    ("<unk>", 0.0, UNKNOWN),
    ("<s>", 0.0, CONTROL),
    ("</s>", 0.0, CONTROL),
    ("\u2581", -10.0, NORMAL),
    ("a", -11.0, NORMAL),
    ("b", -12.0, NORMAL),
    ("c", -13.0, NORMAL),
    *((f"<0x{byte:02X}>", 0.0, BYTE) for byte in range(256)),
    ("ab", -1.0, NORMAL),
    ("\u2581a", -2.0, NORMAL),
    ("\u2581ab", -3.0, NORMAL),
    ("bc", -1.5, NORMAL),
    ("abc", -4.0, NORMAL),
    ("sop", -20.0, NORMAL),
]
SIZE = len(PIECES)
NEWLINE, BYTE_C3, BYTE_A9 = 7 + 0x0A, 7 + 0xC3, 7 + 0xA9
SPACE_A, SPACE_AB, BC = 264, 265, 266
PREFIX = [SIZE + 1, SIZE + 3]
PREFIX_NO_BYTES = [SIZE - 255, SIZE - 253]


def _varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _field(number: int, payload: bytes) -> bytes:
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def _flag(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def write_model(path: Path, *, byte_fallback: bool = True, remove_extra: bool = False) -> Path:
    data = b""
    for piece, score, kind in PIECES:
        if kind == BYTE and not byte_fallback:
            continue
        body = _field(1, piece.encode())
        body += _varint(2 << 3 | 5) + struct.pack("<f", score)
        body += _flag(3, kind)
        data += _field(1, body)
    trainer = _flag(3, 2) + _flag(35, int(byte_fallback)) + _flag(40, 0) + _flag(41, 1)
    trainer += _flag(42, 2)
    data += _field(2, trainer)
    normalizer = _field(1, b"identity") + _flag(3, 1) + _flag(4, int(remove_extra)) + _flag(5, 1)
    data += _field(3, normalizer)
    path.write_bytes(data)
    return path


def test_bpe_merges_follow_scores(tmp_path):
    tokenizer = ChatGLMTokenizer(write_model(tmp_path / "tokenizer.model"))

    assert tokenizer.vocab_size == SIZE + 9
    assert tokenizer.tokenize("") == PREFIX
    # "ab" (-1) merges before "▁a" (-2), so "▁ab" is built from "▁" + "ab".
    assert tokenizer.tokenize("ab") == PREFIX + [SPACE_AB]
    assert tokenizer.tokenize("abc") == PREFIX + [SPACE_AB, 6]
    assert tokenizer.tokenize("bc") == PREFIX + [3, BC]
    assert tokenizer.tokenize("a a") == PREFIX + [SPACE_A, SPACE_A]
    assert tokenizer.tokenize("a a b c") == PREFIX + [SPACE_A, SPACE_A, 3, 5, 3, 6]


def test_byte_fallback_for_unknown_characters(tmp_path):
    tokenizer = ChatGLMTokenizer(write_model(tmp_path / "tokenizer.model"))

    assert tokenizer.tokenize("a\n") == PREFIX + [SPACE_A, NEWLINE]
    assert tokenizer.tokenize("\u00e9") == PREFIX + [3, BYTE_C3, BYTE_A9]
    assert tokenizer.tokenize("a\u00e9b") == PREFIX + [SPACE_A, BYTE_C3, BYTE_A9, 5]


def test_without_byte_fallback_unknown_runs_fuse(tmp_path):
    tokenizer = ChatGLMTokenizer(write_model(tmp_path / "tokenizer.model", byte_fallback=False))
    # Pieces: the three controls and base letters, then merges after the 256 missing bytes.
    space_a = tokenizer.piece_ids["\u2581a"]

    assert tokenizer.tokenize("a\u00e9\u00e9b") == PREFIX_NO_BYTES + [space_a, 0, 5]


def test_whitespace_removal_option(tmp_path):
    kept = ChatGLMTokenizer(write_model(tmp_path / "kept.model"))
    stripped = ChatGLMTokenizer(write_model(tmp_path / "stripped.model", remove_extra=True))

    assert kept.tokenize("  a ") == PREFIX + [3, 3, SPACE_A, 3]
    assert stripped.tokenize("  a ") == PREFIX + [SPACE_A]
    assert stripped.tokenize("   ") == PREFIX


def test_batch_is_left_padded_with_positions(tmp_path):
    tokenizer = ChatGLMTokenizer(write_model(tmp_path / "tokenizer.model"))
    long = PREFIX + [SPACE_A, SPACE_A, 3, 5, 3, 6]

    ids, mask, positions = tokenizer(["ab", "a a b c"], max_length=8)

    assert ids.dtype == torch.long
    assert mask.dtype == torch.bool
    assert ids.tolist() == [[0] * 5 + PREFIX + [SPACE_AB], long]
    assert mask.tolist() == [[False] * 5 + [True] * 3, [True] * 8]
    assert positions.tolist() == [[0] * 5 + [0, 1, 2], list(range(8))]

    ids, mask, positions = tokenizer(["a a b c"], max_length=4)
    assert ids.tolist() == [long[:4]]
    assert mask.all()
