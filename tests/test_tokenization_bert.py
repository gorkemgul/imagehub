import json

import torch

from imagehub.tokenization_bert import BertTokenizer

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "hello",
    "world",
    "##s",
    "cafe",
    "一",
    "只",
    "可",
    "爱",
    "的",
    "猫",
    ",",
    "!",
]


def _write_tokenizer(tmp_path):
    (tmp_path / "vocab.txt").write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "do_lower_case": True,
                "tokenize_chinese_chars": True,
                "strip_accents": None,
                "model_max_length": 77,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_bert_basic_tokenization_wordpiece_and_special_tokens(tmp_path):
    tokenizer = BertTokenizer(_write_tokenizer(tmp_path))

    assert tokenizer.tokenize("Héllo worlds, 一只猫!") == [2, 5, 6, 7, 15, 9, 10, 14, 16, 3]
    assert tokenizer.tokenize("unknown") == [2, 1, 3]
    assert tokenizer.tokenize("") == [2, 3]


def test_bert_batch_pads_truncates_and_returns_expected_dtypes(tmp_path):
    tokenizer = BertTokenizer(_write_tokenizer(tmp_path))

    input_ids, attention_mask = tokenizer(["hello", "hello worlds"], max_length=5)

    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.bool
    assert input_ids.tolist() == [[2, 5, 3, 0, 0], [2, 5, 6, 7, 3]]
    assert attention_mask.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]


def test_bert_wordpiece_rejects_words_longer_than_100_characters(tmp_path):
    tokenizer = BertTokenizer(_write_tokenizer(tmp_path))

    assert tokenizer.tokenize("a" * 101) == [2, 1, 3]
