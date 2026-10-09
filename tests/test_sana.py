import pytest
import torch
from helpers import write_tiny_sana

from imagehub import pipeline
from imagehub.backends.sana import (
    COMPLEX_HUMAN_INSTRUCTION,
    SanaBackend,
    classify_height_width_bin,
)
from imagehub.catalog import model_info
from imagehub.exceptions import CheckpointError
from imagehub.types import T2IRequest


@pytest.fixture(scope="module")
def tiny_sana(tmp_path_factory):
    return write_tiny_sana(tmp_path_factory.mktemp("tiny-sana") / "snapshot")


def test_generates_images_with_metadata(tiny_sana):
    result = pipeline("t2i", backend="sana", model=tiny_sana)(
        "a cat", seed=3, steps=2, width=512, height=512
    )
    assert len(result.images) == 1
    image = result.images[0]
    assert (image.width, image.height) == (512, 512)
    assert image.tensor.dtype == torch.uint8
    metadata = result.metadata
    assert metadata["seed"] == 3
    assert metadata["steps"] == 2
    assert metadata["guidance"] == 4.5
    assert metadata["negative_prompt"] == ""
    assert metadata["scheduler"] == "dpm-solver++"
    assert metadata["device"] == "cpu" and metadata["dtype"] == "float32"


def test_binned_request_is_returned_at_the_requested_size(tiny_sana):
    pipe = pipeline("t2i", backend="sana", model=tiny_sana)
    result = pipe("a cat", seed=1, steps=2, width=600, height=400, max_sequence_length=32)
    assert (result.images[0].width, result.images[0].height) == (600, 400)
    unbinned = pipe(
        "a cat",
        seed=1,
        steps=2,
        width=320,
        height=224,
        use_resolution_binning=False,
        max_sequence_length=32,
    )
    assert (unbinned.images[0].width, unbinned.images[0].height) == (320, 224)


def test_guidance_one_and_disabled_instruction(tiny_sana):
    pipe = pipeline("t2i", backend="sana", model=tiny_sana)
    options = {"seed": 1, "steps": 2, "width": 512, "height": 512, "max_sequence_length": 32}
    plain = pipe("a cat", guidance=1.0, complex_human_instruction=[], **options)
    guided = pipe("a cat", **options)
    assert plain.metadata["guidance"] == 1.0
    assert not torch.equal(plain.images[0].tensor, guided.images[0].tensor)


def test_seed_is_reproducible(tiny_sana):
    pipe = pipeline("t2i", backend="sana", model=tiny_sana)
    options = {"steps": 2, "width": 512, "height": 512, "max_sequence_length": 32}
    a = pipe("a cat", seed=11, **options)
    b = pipe("a cat", seed=11, **options)
    c = pipe("a cat", seed=12, **options)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert not torch.equal(a.images[0].tensor, c.images[0].tensor)


def test_requests_are_isolated(tiny_sana):
    pipe = pipeline("t2i", backend="sana", model=tiny_sana)
    options = {"width": 512, "height": 512, "max_sequence_length": 32}
    first = pipe("a cat", seed=1, steps=2, **options)
    pipe("a hi cat", seed=2, steps=3, guidance=2.0, negative_prompt="blurry", **options)
    again = pipe("a cat", seed=1, steps=2, **options)
    assert torch.equal(first.images[0].tensor, again.images[0].tensor)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_sana, grad_before):
    pipe = pipeline("t2i", backend="sana", model=tiny_sana)
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        result = pipe("a cat", seed=1, steps=1, width=512, height=512, max_sequence_length=32)
        assert torch.is_grad_enabled() is grad_before
        assert not torch.is_inference_mode_enabled()
        assert not result.images[0].tensor.requires_grad
    finally:
        torch.set_grad_enabled(previous)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"latents": torch.zeros(1)}, TypeError, "latents"),
        ({"width": 1001, "use_resolution_binning": False}, ValueError, "multiple of 32"),
        ({"height": 500, "use_resolution_binning": False}, ValueError, "multiple of 32"),
        ({"steps": 0}, ValueError, "steps"),
        ({"steps": 1001}, ValueError, "at most 1000"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"guidance": -1.0}, ValueError, "guidance"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"seed": -1}, ValueError, "seed"),
        ({"use_resolution_binning": 1}, TypeError, "use_resolution_binning"),
        ({"max_sequence_length": 0}, ValueError, "max_sequence_length"),
        ({"complex_human_instruction": "text"}, TypeError, "complex_human_instruction"),
        ({"complex_human_instruction": [1]}, TypeError, "complex_human_instruction"),
    ],
)
def test_invalid_options_fail_before_loading(tmp_path, options, error, match):
    backend = SanaBackend(tmp_path / "missing")
    with pytest.raises(error, match=match):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded


def test_catalog_defaults_and_instruction_option(tmp_path):
    for model_id in ("sana-1600m-1024", "sana-600m-1024"):
        spec = model_info(model_id)
        assert spec.family == "sana"
        options = SanaBackend(tmp_path, model_id=model_id).validate_options(T2IRequest("x"), {})
        assert options["steps"] == 20
        assert options["guidance"] == 4.5
        assert options["width"] == options["height"] == 1024
        assert options["negative_prompt"] == ""
        assert options["use_resolution_binning"] is True
        assert options["max_sequence_length"] == 300
        assert options["complex_human_instruction"] == tuple(COMPLEX_HUMAN_INSTRUCTION)
        assert options["latent_shape"] == (1, 32, 32, 32)
    backend = SanaBackend(tmp_path)
    disabled = backend.validate_options(T2IRequest("x"), {"complex_human_instruction": []})
    assert disabled["complex_human_instruction"] == ()
    none_negative = backend.validate_options(T2IRequest("x"), {"negative_prompt": None})
    assert none_negative["negative_prompt"] == ""


def test_request_parameters_are_options(tmp_path):
    backend = SanaBackend(tmp_path)
    options = backend.validate_options(T2IRequest("x", {"steps": 5}), {"guidance": 2.0})
    assert options["steps"] == 5 and options["guidance"] == 2.0
    with pytest.raises(TypeError, match="bogus"):
        backend.validate_options(T2IRequest("x", {"bogus": 1}), {})


def test_resolution_binning_snaps_the_request(tmp_path):
    assert classify_height_width_bin(1000, 800, 32) == (1088, 896)
    assert classify_height_width_bin(800, 1000, 32) == (896, 1088)
    assert classify_height_width_bin(1000, 800, 16) == (544, 448)
    with pytest.raises(ValueError, match="Invalid sample size"):
        classify_height_width_bin(1024, 1024, 64)
    backend = SanaBackend(tmp_path)
    options = backend.validate_options(T2IRequest("x"), {"width": 800, "height": 1000})
    assert (options["height"], options["width"]) == (1088, 896)
    assert (options["output_height"], options["output_width"]) == (1000, 800)
    assert options["latent_shape"] == (1, 32, 34, 28)


def test_incomplete_snapshot_is_reported(tmp_path):
    with pytest.raises(CheckpointError, match="not a Sana snapshot"):
        SanaBackend(tmp_path, model_id="sana-600m-1024").load()
