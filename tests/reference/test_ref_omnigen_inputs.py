"""OmniGen input handling parity: image preprocessing, token layout, masks and positions."""

from pathlib import Path

import pytest
import torch

from imagehub.backends.omnigen_inputs import (
    IMAGE_PLACEHOLDER,
    NEGATIVE_PROMPT,
    build_inputs,
    preprocess_input_image,
)
from imagehub.tokenization_omnigen import OmniGenTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
pytest.importorskip("torchvision")
PIL = pytest.importorskip("PIL.Image")
from diffusers.pipelines.omnigen.processor_omnigen import OmniGenMultiModalProcessor  # noqa: E402

FILES = Path(__file__).parents[1] / "fixtures" / "omnigen_tokenizer"
SIZES = [(64, 48), (100, 70), (333, 257), (1024, 1024), (1500, 900), (2500, 2100), (5, 5), (3, 400)]


def _image(size, seed, mode="RGB"):
    generator = torch.Generator().manual_seed(seed)
    data = torch.randint(0, 256, (size[1], size[0], 3), generator=generator, dtype=torch.uint8)
    return PIL.fromarray(data.numpy()).convert(mode)


def _reference():
    tokenizer = transformers.LlamaTokenizer.from_pretrained(FILES)
    return OmniGenMultiModalProcessor(tokenizer, max_image_size=1024)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("limit", [1024, 256])
def test_preprocess_matches_reference(record, size, limit):
    reference = _reference()
    reference.reset_max_image_size(limit)
    image = _image(size, 0)
    expected = reference.process_image(image)
    actual = preprocess_input_image(image, limit)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    record(f"omnigen_preprocess_{size[0]}x{size[1]}_{limit}", max_abs_diff=0.0)


def test_preprocess_converts_other_modes_to_rgb():
    image = _image((70, 50), 1, "RGBA")
    assert torch.equal(preprocess_input_image(image), preprocess_input_image(image.convert("RGB")))


def _same(actual, expected):
    assert torch.equal(actual.input_ids, expected["input_ids"])
    assert torch.equal(actual.attention_mask, expected["attention_mask"])
    assert torch.equal(actual.position_ids, expected["position_ids"])
    assert actual.input_image_sizes == expected["input_image_sizes"]
    assert len(actual.input_pixel_values) == len(expected["input_pixel_values"])
    for a, e in zip(actual.input_pixel_values, expected["input_pixel_values"], strict=True):
        assert torch.equal(a, e)


@pytest.mark.parametrize("prompt", ["a cat on a mat", "", "line\nbreak  and   spaces"])
@pytest.mark.parametrize("hw", [(64, 64), (96, 160)])
def test_text_only_inputs_match_reference(record, prompt, hw):
    reference = _reference()
    ours = OmniGenTokenizer(FILES)
    expected = reference([prompt], [None], height=hw[0], width=hw[1], use_img_cfg=False)
    actual = build_inputs(ours, prompt, None, height=hw[0], width=hw[1], use_img_cfg=False)
    _same(actual, expected)
    record("omnigen_inputs_text", mismatches=0)


@pytest.mark.parametrize(
    "prompt",
    [
        f"Put a hat on {IMAGE_PLACEHOLDER}.",
        f"{IMAGE_PLACEHOLDER} as a watercolor",
        f"{IMAGE_PLACEHOLDER}",
        f"make {IMAGE_PLACEHOLDER} look like {IMAGE_PLACEHOLDER}",
    ],
)
@pytest.mark.parametrize("size", [(64, 48), (333, 257), (2500, 2100)])
@pytest.mark.parametrize("input_size_as_output", [False, True])
def test_image_inputs_match_reference(record, prompt, size, input_size_as_output):
    reference = _reference()
    ours = OmniGenTokenizer(FILES)
    image = _image(size, 2)
    expected = reference(
        [prompt],
        [[image]],
        height=96,
        width=128,
        use_img_cfg=True,
        use_input_image_size_as_output=input_size_as_output,
    )
    pixels = preprocess_input_image(image)
    actual = build_inputs(
        ours,
        prompt,
        [pixels],
        height=96,
        width=128,
        use_img_cfg=True,
        use_input_image_size_as_output=input_size_as_output,
    )
    _same(actual, expected)
    if input_size_as_output:
        assert (actual.height, actual.width) == tuple(pixels.shape[-2:])
    record("omnigen_inputs_image", mismatches=0)


def test_negative_prompt_matches_reference_default():
    import inspect

    default = inspect.signature(OmniGenMultiModalProcessor.__call__).parameters["negative_prompt"]
    assert default.default == NEGATIVE_PROMPT
