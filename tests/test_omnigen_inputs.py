from pathlib import Path

import pytest
import torch

from imagehub.backends.omnigen_inputs import (
    IMAGE_PLACEHOLDER,
    NEGATIVE_PROMPT,
    build_inputs,
    crop_image,
    preprocess_input_image,
    validate_prompt,
)
from imagehub.tokenization_omnigen import OmniGenTokenizer

PIL = pytest.importorskip("PIL.Image")
FILES = Path(__file__).parent / "fixtures" / "omnigen_tokenizer"


def _picture(width, height, mode="RGB"):
    return PIL.new(mode, (width, height), (10, 120, 230))


def test_placeholder_convention():
    validate_prompt(f"a hat on {IMAGE_PLACEHOLDER}", True)
    validate_prompt(f"{IMAGE_PLACEHOLDER} and {IMAGE_PLACEHOLDER}", True)
    validate_prompt("a cat", False)
    with pytest.raises(ValueError, match="placeholder"):
        validate_prompt("a hat on the cat", True)
    with pytest.raises(ValueError, match="placeholder"):
        validate_prompt("<|image_1|> without the img tags", True)
    with pytest.raises(ValueError, match="only one input image"):
        validate_prompt(f"{IMAGE_PLACEHOLDER} and <img><|image_2|></img>", True)
    with pytest.raises(ValueError, match="no image"):
        validate_prompt(f"a hat on {IMAGE_PLACEHOLDER}", False)
    with pytest.raises(ValueError, match="no image"):
        validate_prompt("<|image_3|>", False)


@pytest.mark.parametrize(
    ("size", "limit", "expected"),
    [
        ((64, 48), 1024, (64, 48)),
        ((70, 50), 1024, (64, 48)),
        ((2000, 1000), 1024, (1024, 512)),
        ((5, 5), 1024, (16, 16)),
        ((100, 100), 40, (32, 32)),
    ],
)
def test_crop_image_sizes(size, limit, expected):
    assert crop_image(_picture(*size), limit).size == expected


def test_preprocess_range_shape_and_modes():
    tensor = preprocess_input_image(_picture(70, 50))
    assert tensor.shape == (3, 48, 64) and tensor.dtype == torch.float32
    assert tensor.min() >= -1 and tensor.max() <= 1
    assert torch.equal(tensor, preprocess_input_image(_picture(70, 50, "RGBA")))


def test_text_only_layout():
    tokenizer = OmniGenTokenizer(FILES)
    inputs = build_inputs(tokenizer, "a cat", None, height=64, width=96, use_img_cfg=False)
    tokens = 64 * 96 // 256
    assert inputs.input_ids.shape[0] == 2
    length = inputs.input_ids.shape[1]
    assert inputs.attention_mask.shape == (2, length + tokens + 1, length + tokens + 1)
    assert inputs.position_ids.shape == (2, length + tokens + 1)
    assert inputs.input_pixel_values == [] and inputs.input_image_sizes == {}
    # The prompt is shorter than the negative prompt, so it is left padded with token 2.
    assert inputs.input_ids[0, 0] == 2 and inputs.input_ids[1, 0] == 1
    assert inputs.attention_mask[0, 0].min() == 1  # padded rows attend everywhere


def test_image_layout_places_placeholders_in_the_prompt_and_image_prompt():
    tokenizer = OmniGenTokenizer(FILES)
    pixels = preprocess_input_image(_picture(64, 48))
    inputs = build_inputs(
        tokenizer, f"edit {IMAGE_PLACEHOLDER}", [pixels], height=64, width=64, use_img_cfg=True
    )
    image_tokens = 48 * 64 // 256
    assert inputs.input_ids.shape[0] == 3
    assert len(inputs.input_pixel_values) == 2
    assert all(x.shape == (1, 3, 48, 64) for x in inputs.input_pixel_values)
    assert sorted(inputs.input_image_sizes) == [0, 2]
    for spans in inputs.input_image_sizes.values():
        assert len(spans) == 1 and spans[0][1] - spans[0][0] == image_tokens
    # Image tokens attend to each other and to everything before them.
    start, end = inputs.input_image_sizes[0][0]
    assert inputs.attention_mask[0, start:end, start:end].min() == 1
    assert inputs.attention_mask[0, start, end:].max() == 0  # causal: nothing after the image


def test_input_size_as_output_overrides_height_and_width():
    tokenizer = OmniGenTokenizer(FILES)
    pixels = preprocess_input_image(_picture(64, 48))
    inputs = build_inputs(
        tokenizer,
        f"edit {IMAGE_PLACEHOLDER}",
        [pixels],
        height=512,
        width=512,
        use_img_cfg=True,
        use_input_image_size_as_output=True,
    )
    assert (inputs.height, inputs.width) == (48, 64)


def test_negative_prompt_text():
    assert NEGATIVE_PROMPT.startswith("low quality, jpeg artifacts, ugly")
    assert NEGATIVE_PROMPT.endswith("too many fingers.")
