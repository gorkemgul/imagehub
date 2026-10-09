import pytest
import torch
from helpers import write_tiny_lumina2

from imagehub import pipeline
from imagehub.backends.lumina2 import DEFAULT_SYSTEM_PROMPT, Lumina2Backend
from imagehub.catalog import model_info
from imagehub.exceptions import CheckpointError
from imagehub.types import T2IRequest

SMALL = {"steps": 2, "width": 128, "height": 96, "max_sequence_length": 32}


@pytest.fixture(scope="module")
def tiny_lumina2(tmp_path_factory):
    return write_tiny_lumina2(tmp_path_factory.mktemp("tiny-lumina2") / "snapshot")


def test_generates_images_with_metadata(tiny_lumina2):
    result = pipeline("t2i", backend="lumina2", model=tiny_lumina2)("a cat", seed=3, **SMALL)
    assert len(result.images) == 1
    image = result.images[0]
    assert (image.width, image.height) == (128, 96)
    assert image.tensor.dtype == torch.uint8
    metadata = result.metadata
    assert metadata["seed"] == 3
    assert metadata["steps"] == 2
    assert metadata["guidance"] == 4.0
    assert metadata["negative_prompt"] == ""
    assert metadata["cfg_trunc_ratio"] == 1.0
    assert metadata["cfg_normalization"] is True
    assert metadata["scheduler"] == "flow_match_euler"
    assert metadata["device"] == "cpu" and metadata["dtype"] == "float32"


def test_options_change_the_result(tiny_lumina2):
    pipe = pipeline("t2i", backend="lumina2", model=tiny_lumina2)
    base = pipe("a cat", seed=1, steps=3, width=64, height=64, max_sequence_length=32)
    for override in (
        {"guidance": 1.0},
        {"cfg_normalization": False},
        {"cfg_trunc_ratio": 0.3},
        {"system_prompt": "Describe the image."},
        {"negative_prompt": "blurry"},
    ):
        other = pipe(
            "a cat", seed=1, steps=3, width=64, height=64, max_sequence_length=32, **override
        )
        assert not torch.equal(base.images[0].tensor, other.images[0].tensor), override


def test_seed_is_reproducible(tiny_lumina2):
    pipe = pipeline("t2i", backend="lumina2", model=tiny_lumina2)
    a = pipe("a cat", seed=11, **SMALL)
    b = pipe("a cat", seed=11, **SMALL)
    c = pipe("a cat", seed=12, **SMALL)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert not torch.equal(a.images[0].tensor, c.images[0].tensor)


def test_requests_are_isolated(tiny_lumina2):
    pipe = pipeline("t2i", backend="lumina2", model=tiny_lumina2)
    first = pipe("a cat", seed=1, **SMALL)
    pipe("a hi cat", seed=2, guidance=2.0, negative_prompt="blurry", **SMALL | {"steps": 3})
    again = pipe("a cat", seed=1, **SMALL)
    assert torch.equal(first.images[0].tensor, again.images[0].tensor)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_lumina2, grad_before):
    pipe = pipeline("t2i", backend="lumina2", model=tiny_lumina2)
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        result = pipe("a cat", seed=1, **SMALL | {"steps": 1})
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
        ({"width": 1001}, ValueError, "multiple of 16"),
        ({"height": 500}, ValueError, "multiple of 16"),
        ({"steps": 0}, ValueError, "steps"),
        ({"steps": 1001}, ValueError, "at most 1000"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"guidance": -1.0}, ValueError, "guidance"),
        ({"cfg_trunc_ratio": -0.1}, ValueError, "cfg_trunc_ratio"),
        ({"cfg_trunc_ratio": "1"}, ValueError, "cfg_trunc_ratio"),
        ({"cfg_normalization": 1}, TypeError, "cfg_normalization"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"system_prompt": 3}, TypeError, "system_prompt"),
        ({"seed": -1}, ValueError, "seed"),
        ({"max_sequence_length": 0}, ValueError, "max_sequence_length"),
        ({"max_sequence_length": 513}, ValueError, "at most 512"),
    ],
)
def test_invalid_options_fail_before_loading(tmp_path, options, error, match):
    backend = Lumina2Backend(tmp_path / "missing")
    with pytest.raises(error, match=match):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded


def test_snapshot_limits_are_checked_before_loading(tiny_lumina2):
    backend = Lumina2Backend(tiny_lumina2)
    with pytest.raises(ValueError, match="max_sequence_length must be below 300"):
        backend.infer(T2IRequest("a cat"), max_sequence_length=300)
    with pytest.raises(ValueError, match="width must be at most 1024"):
        backend.infer(T2IRequest("a cat"), width=1040)
    assert not backend.is_loaded


def test_catalog_defaults(tmp_path):
    spec = model_info("lumina-image-2.0")
    assert spec.family == "lumina2"
    options = Lumina2Backend(tmp_path, model_id="lumina-image-2.0").validate_options(
        T2IRequest("x"), {}
    )
    assert options["steps"] == 30
    assert options["guidance"] == 4.0
    assert options["width"] == options["height"] == 1024
    assert options["negative_prompt"] == ""
    assert options["system_prompt"] == DEFAULT_SYSTEM_PROMPT
    assert options["cfg_trunc_ratio"] == 1.0
    assert options["cfg_normalization"] is True
    assert options["max_sequence_length"] == 256
    assert options["latent_shape"] == (1, 16, 128, 128)


def test_none_options_select_the_defaults(tmp_path):
    options = Lumina2Backend(tmp_path).validate_options(
        T2IRequest("x"), {"negative_prompt": None, "system_prompt": None}
    )
    assert options["negative_prompt"] == ""
    assert options["system_prompt"] == DEFAULT_SYSTEM_PROMPT
    empty = Lumina2Backend(tmp_path).validate_options(T2IRequest("x"), {"system_prompt": ""})
    assert empty["system_prompt"] == ""


def test_request_parameters_are_options(tmp_path):
    backend = Lumina2Backend(tmp_path)
    options = backend.validate_options(T2IRequest("x", {"steps": 5}), {"guidance": 2.0})
    assert options["steps"] == 5 and options["guidance"] == 2.0
    with pytest.raises(TypeError, match="bogus"):
        backend.validate_options(T2IRequest("x", {"bogus": 1}), {})


def test_incomplete_snapshot_is_reported(tmp_path):
    with pytest.raises(CheckpointError, match="not a Lumina-Image 2.0 snapshot"):
        Lumina2Backend(tmp_path, model_id="lumina-image-2.0").load()
