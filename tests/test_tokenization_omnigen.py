from pathlib import Path

from imagehub.tokenization_omnigen import OmniGenTokenizer

FILES = Path(__file__).parent / "fixtures" / "omnigen_tokenizer"


def _ids(text):
    return OmniGenTokenizer(FILES).tokenize(text)


def test_instruction_prompt_ids():
    text = (
        "<|user|>\nGenerate an image according to the following instructions\nA cat<|end|>\n"
        "<|assistant|>\n<|diffusion|>"
    )
    assert _ids(text) == [
        1, 32010, 13, 5631, 403, 385, 1967, 5034, 304, 278, 1494, 11994, 13, 29909, 6635,
        32007, 13, 32001, 32048,
    ]  # fmt: skip


def test_only_the_first_piece_gets_a_prefix_space():
    assert _ids("a <img>") == [1, 263, 29871, 32045]
    assert _ids(" </img> hi") == [1, 29871, 32046, 7251]
    assert _ids("x</s>  y") == [1, 921, 2, 29891]


def test_rstrip_tokens_swallow_following_whitespace():
    assert _ids("<|assistant|>\n\n  <|diffusion|>") == [1, 32001, 32048]
    # <|end|> does not strip, so the newline stays (byte-fallback token for "\n").
    assert _ids("<|end|>\n<|diffusion|>") == [1, 32007, 13, 32048]


def test_empty_text_is_just_the_start_token():
    assert _ids("") == [1]
