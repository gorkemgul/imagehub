from pathlib import Path

import pytest
import torch
from glm_image_helpers import write_tiny_glm_image

from imagehub import pipeline
from imagehub.backends.glm_image import GlmImageBackend, glm_image_sigmas, glyph_texts
from imagehub.exceptions import CheckpointError
from imagehub.types import T2IRequest


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return write_tiny_glm_image(tmp_path_factory.mktemp("tiny-glm-image"))


def test_defaults_come_from_the_catalog():
    backend = GlmImageBackend(Path("missing"), model_id="glm-image")
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert (options["steps"], options["guidance"], options["max_sequence_length"]) == (50, 1.5, 2048)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"negative_prompt": "x"}, TypeError, "negative_prompt"),
        ({"image": object()}, TypeError, "image"),
        ({"width": 1000}, ValueError, "multiple of 32"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"max_sequence_length": 0}, ValueError, "max_sequence_length"),
        ({"latents": torch.zeros(1, 16, 64, 64)}, ValueError, "shape"),
    ],
)
def test_invalid_options_fail_before_loading(options, error, match):
    backend = GlmImageBackend(Path("missing"))
    with pytest.raises(error, match=match):
        backend.validate_options(T2IRequest("a cat"), options)
    assert not backend.is_loaded


def test_sigmas_match_the_pipeline_formula():
    assert glm_image_sigmas(4, 1000) == [
        1.0,
        0.75,
        0.5,
        0.25,
    ]
    assert glm_image_sigmas(3, 1000)[1] == pytest.approx(0.667, abs=1e-6)


def test_glyph_texts_collect_every_quote_style():
    prompt = "a sign 'one' with “two” and \"three\" and 「four」"
    assert glyph_texts(prompt) == ["one", "two", "three", "four"]
    assert glyph_texts("no quotes") == []


def test_incomplete_snapshot_is_reported(tmp_path):
    with pytest.raises(CheckpointError, match="not a GLM-Image snapshot"):
        GlmImageBackend(tmp_path).load()


OPTIONS = {"steps": 2, "width": 64, "height": 64}


def test_generates_with_metadata(tiny):
    result = pipeline("t2i", backend="glm-image", model=tiny)(
        "a sign saying 'hello' and \"x\"", seed=3, guidance=2.0, **OPTIONS
    )
    image = result.images[0]
    assert (image.width, image.height) == (64, 64)
    assert image.tensor.dtype == torch.uint8
    assert result.metadata["seed"] == 3
    assert result.metadata["guidance"] == 2.0
    assert result.metadata["device"] == "cpu"


def test_seed_is_reproducible_and_guidance_can_be_off(tiny):
    pipe = pipeline("t2i", backend="glm-image", model=tiny)
    a = pipe("a cat", seed=5, guidance=1.0, **OPTIONS)
    b = pipe("a cat", seed=5, guidance=1.0, **OPTIONS)
    c = pipe("a cat", seed=6, guidance=1.0, **OPTIONS)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert not torch.equal(a.images[0].tensor, c.images[0].tensor)


def test_non_square_output(tiny):
    result = pipeline("t2i", backend="glm-image", model=tiny)(
        "a cat", seed=1, steps=1, width=96, height=64
    )
    assert (result.images[0].width, result.images[0].height) == (96, 64)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny, grad_before):
    pipe = pipeline("t2i", backend="glm-image", model=tiny)
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        pipe("a cat", seed=1, steps=1, width=64, height=64)
        assert torch.is_grad_enabled() is grad_before
        assert not torch.is_inference_mode_enabled()
    finally:
        torch.set_grad_enabled(previous)
