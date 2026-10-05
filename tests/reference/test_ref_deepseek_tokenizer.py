import random

import pytest
import torch
from helpers import snapshot_dir

from imagehub.catalog import MODEL_CATALOG
from imagehub.tokenization_deepseek import DeepSeekTokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")
JANUS_MODEL_IDS = tuple(spec.id for spec in MODEL_CATALOG.values() if spec.family == "janus")

FIXED_PROMPTS = [
    "a cat",
    "",
    "  leading and trailing  ",
    "Bir kedi, İstanbul'da ĞÜŞİÖÇ ığüşöç",
    "Ελληνικά Русский текст",
    "日本語のテキストと中文和한국어",
    "emoji 🐱🚀 and   multiple\tspaces\nnewline",
    "DON'T Don't don't they're I'VE I've i'm we'll he'd",
    "numbers 1234567890 3.14 1e-5 ½ ² Ⅷ ٣",
    "x<begin_of_image>y <｜begin▁of▁sentence｜> <｜end▁of▁sentence｜>",
    "<begin_of_imag> <begin_of_imagex> <image_0x>",
    " \r\n\n\t trailing   ",
    "word " * 200,
]

FUZZ_PIECES = [
    "a",
    "Cat",
    "İstanbul",
    "ΣΟΦΟΣ",
    "Кот",
    "日本",
    "中文",
    "한국어",
    "🐱",
    "🚀",
    "123456",
    "١٢٣",
    "Ⅷ½",
    "'s",
    "'T",
    "'RE",
    "don't",
    " ",
    "   ",
    "\t",
    "\n",
    "\r\n\n",
    "!?.",
    "<begin_of_image>",
    "<image_0>",
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<begin_of_imag>",
    "<begin_of_imagex>",
    "<image_0x>",
]


def _processor_prompt(processor, prompt):
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        add_generation_prompt=True,
    )


def _assert_prompts_match(ours, processor, prompts):
    expected_prompts = [_processor_prompt(processor, prompt) for prompt in prompts]
    actual_prompts = [ours.image_prompt(prompt) for prompt in prompts]
    assert actual_prompts == [prompt + processor.image_start_token for prompt in expected_prompts]

    actual_ids, actual_mask = ours(actual_prompts)
    expected = processor(
        text=expected_prompts,
        generation_mode="image",
        padding=True,
        return_tensors="pt",
    )
    assert torch.equal(actual_ids, expected["input_ids"])
    assert torch.equal(actual_mask, expected["attention_mask"].bool())


def _record_key(key, model_id):
    return key if model_id == "janus-pro-1b" else f"{key}_{model_id}"


@pytest.mark.parametrize("model_id", JANUS_MODEL_IDS)
def test_deepseek_tokenizer_fixed_prompts_and_processor_match_reference(record, model_id):
    path = snapshot_dir(model_id)
    ours = DeepSeekTokenizer(path)
    reference = transformers.AutoTokenizer.from_pretrained(path)
    processor = transformers.JanusProcessor.from_pretrained(path)

    for prompt in FIXED_PROMPTS:
        assert ours.tokenize(prompt) == reference.encode(prompt)
    _assert_prompts_match(ours, processor, FIXED_PROMPTS)
    record(
        _record_key("deepseek_tokenizer", model_id),
        prompts=len(FIXED_PROMPTS),
        mismatches=0,
        tolerance="exact",
    )


@pytest.mark.parametrize("model_id", JANUS_MODEL_IDS)
def test_deepseek_tokenizer_seeded_fuzz_matches_reference(record, model_id):
    path = snapshot_dir(model_id)
    ours = DeepSeekTokenizer(path)
    reference = transformers.AutoTokenizer.from_pretrained(path)
    processor = transformers.JanusProcessor.from_pretrained(path)
    rng = random.Random(0)
    prompts = [
        "".join(rng.choice(FUZZ_PIECES) for _ in range(rng.randint(0, 18))) for _ in range(3000)
    ]

    for start in range(0, len(prompts), 100):
        batch = prompts[start : start + 100]
        expected = reference(batch, add_special_tokens=True)["input_ids"]
        assert [ours.tokenize(prompt) for prompt in batch] == expected
        _assert_prompts_match(ours, processor, batch)
    record(
        _record_key("deepseek_tokenizer_fuzz", model_id),
        prompts=len(prompts),
        mismatches=0,
        tolerance="exact",
    )


@pytest.mark.parametrize("model_id", JANUS_MODEL_IDS)
def test_deepseek_tokenizer_matches_reference_for_every_code_point(record, model_id):
    path = snapshot_dir(model_id)
    ours = DeepSeekTokenizer(path)
    reference = transformers.AutoTokenizer.from_pretrained(path)
    mismatches = 0
    checked = 0
    prompts = []

    def compare_batch():
        nonlocal mismatches, checked
        expected = reference(prompts, add_special_tokens=True)["input_ids"]
        actual = [ours.tokenize(prompt) for prompt in prompts]
        mismatches += sum(left != right for left, right in zip(actual, expected, strict=True))
        checked += len(prompts)
        prompts.clear()

    for code_point in range(0x20, 0x110000):
        if 0xD800 <= code_point <= 0xDFFF:
            continue
        character = chr(code_point)
        prompts.extend(("a" + character + "b", " " + character + "1", character + character))
        if len(prompts) >= 768:
            compare_batch()
    if prompts:
        compare_batch()

    assert mismatches == 0
    record(
        _record_key("deepseek_tokenizer_exhaustive", model_id),
        prompts=checked,
        mismatches=mismatches,
        tolerance="exact",
    )
