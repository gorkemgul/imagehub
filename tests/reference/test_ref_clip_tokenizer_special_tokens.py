import json
from pathlib import Path

import pytest
from helpers import write_tiny_tokenizer

from imagehub.tokenization import CLIPTokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

BOS, EOS = "<|startoftext|>", "<|endoftext|>"
MAX_LENGTH = 24
PROMPTS = [
    "hi!!! cat",
    "Hi CAT!",
    "",
    "x<|endoftext|>y",
    "x <|ENDOFTEXT|> y",
    "<|startoftext|>hi <|endoftext|>",
    "<|StartOfText|>hi",
    "a<|STARTOFTEXT|>b",
    "<|STARTOFTEXT|><|endoftext|>!",
]


def _token(content, *, normalized=None):
    if normalized is None:
        return content
    return {
        "content": content,
        "lstrip": False,
        "normalized": normalized,
        "rstrip": False,
        "single_word": False,
    }


def _decoder(bos_normalized, eos_normalized):
    return {
        "515": {
            "content": BOS,
            "lstrip": False,
            "normalized": bos_normalized,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
        "516": {
            "content": EOS,
            "lstrip": False,
            "normalized": eos_normalized,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
    }


def _tokens(unk, bos, eos, pad):
    return {"unk_token": unk, "bos_token": bos, "eos_token": eos, "pad_token": pad}


SCENARIOS = {
    "config_only_string_pad": (_tokens(EOS, BOS, EOS, "!"), None),
    "config_only_dicts": (
        _tokens(
            _token(EOS, normalized=True),
            _token(BOS, normalized=True),
            _token(EOS, normalized=True),
            EOS,
        ),
        None,
    ),
    "map_only_string_pad": (None, _tokens(EOS, BOS, EOS, "!")),
    "map_only_dicts": (
        None,
        _tokens(
            _token(EOS, normalized=False),
            _token(BOS, normalized=True),
            _token(EOS, normalized=False),
            "!",
        ),
    ),
    "legacy_map_overrides_string_pad": (
        _tokens(EOS, BOS, EOS, EOS),
        _tokens(EOS, BOS, EOS, "!"),
    ),
    "legacy_dict_config_string_map": (
        _tokens(
            _token(EOS, normalized=True),
            _token(BOS, normalized=True),
            _token(EOS, normalized=True),
            EOS,
        ),
        _tokens(EOS, BOS, EOS, "!"),
    ),
    "legacy_string_config_dict_map_normalized": (
        _tokens(EOS, BOS, EOS, EOS),
        _tokens(
            _token(EOS, normalized=True),
            _token(BOS, normalized=True),
            _token(EOS, normalized=True),
            "!",
        ),
    ),
    "legacy_string_config_dict_map_raw": (
        _tokens(EOS, BOS, EOS, EOS),
        _tokens(
            _token(EOS, normalized=False),
            _token(BOS, normalized=False),
            _token(EOS, normalized=False),
            "!",
        ),
    ),
    "legacy_conflicting_bos_eos": (
        _tokens(EOS, BOS, EOS, EOS),
        {"bos_token": _token(BOS, normalized=True), "eos_token": EOS, "pad_token": "!"},
    ),
    "legacy_partial_map_keeps_config_keys": (
        _tokens(_token(EOS, normalized=True), _token(BOS, normalized=True), EOS, EOS),
        {"pad_token": "!"},
    ),
    "config_dict_without_normalized": (
        _tokens(EOS, {"content": BOS}, EOS, "!"),
        None,
    ),
    "map_dict_without_normalized": (
        None,
        _tokens(EOS, {"content": BOS}, {"content": EOS}, "!"),
    ),
    "config_unk_dict_registers_after_plain_eos": (
        {
            "bos_token": BOS,
            "eos_token": EOS,
            "unk_token": _token(EOS, normalized=True),
            "pad_token": "!",
        },
        None,
    ),
    "config_eos_dict_registers_before_plain_unk": (
        {
            "bos_token": BOS,
            "eos_token": _token(EOS, normalized=True),
            "unk_token": EOS,
            "pad_token": "!",
        },
        None,
    ),
    "modern_decoder_and_map": (
        {**_tokens(EOS, BOS, EOS, EOS), "added_tokens_decoder": _decoder(True, True)},
        _tokens(_token(EOS, normalized=False), _token(BOS, normalized=False), EOS, "!"),
    ),
    "modern_decoder_raw_and_dict_map": (
        {
            **_tokens(_token(EOS, normalized=True), BOS, EOS, "!"),
            "added_tokens_decoder": _decoder(False, False),
        },
        _tokens(EOS, _token(BOS, normalized=True), _token(EOS, normalized=True), EOS),
    ),
}


def _typed(config):
    """Dict values in tokenizer_config.json must carry the AddedToken marker, as saved files do."""
    return {
        key: {"__type": "AddedToken", **value}
        if key.endswith("_token") and isinstance(value, dict)
        else value
        for key, value in config.items()
    }


def _write(directory, config, special_map):
    path = write_tiny_tokenizer(directory)
    (path / "tokenizer_config.json").unlink()
    (path / "special_tokens_map.json").unlink()
    if config is not None:
        payload = {**_typed(config), "model_max_length": MAX_LENGTH, "do_lower_case": True}
        (path / "tokenizer_config.json").write_text(json.dumps(payload), encoding="utf-8")
    if special_map is not None:
        (path / "special_tokens_map.json").write_text(json.dumps(special_map), encoding="utf-8")
    return Path(path)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_special_tokens_match_transformers(tmp_path, name):
    config, special_map = SCENARIOS[name]
    path = _write(tmp_path, config, special_map)
    ours = CLIPTokenizer(path, max_length=MAX_LENGTH)
    ref = transformers.CLIPTokenizer.from_pretrained(path)
    assert ours.pad_id == ref.pad_token_id
    for text in PROMPTS:
        expected = ref(text, padding="max_length", max_length=MAX_LENGTH, truncation=True)
        ids = ours(text)
        assert ids == expected["input_ids"], (name, text)
        length = min(len(ours.tokenize(text)), MAX_LENGTH - 2) + 2
        mask = [1] * length + [0] * (MAX_LENGTH - length)
        assert mask == expected["attention_mask"], (name, text)
