import pytest
import torch
from helpers import write_tiny_infinity

from imagehub import pipeline
from imagehub.backends.infinity import InfinityBackend, scale_schedule
from imagehub.catalog import model_info
from imagehub.types import T2IRequest

PROMPT = "A tiny blue bird"


@pytest.fixture(scope="module")
def tiny_infinity(tmp_path_factory):
    return write_tiny_infinity(tmp_path_factory.mktemp("tiny-infinity") / "snapshot")


def test_generates_repeatable_256_image_and_metadata(tiny_infinity):
    pipe = pipeline("t2i", backend="infinity", model=tiny_infinity)
    first = pipe(PROMPT, seed=3, width=256, height=256)
    second = pipe(PROMPT, seed=3, width=256, height=256)
    assert torch.equal(first.images[0].tensor, second.images[0].tensor)
    assert (first.images[0].width, first.images[0].height) == (256, 256)
    assert first.metadata == {
        "model_id": None,
        "revision": None,
        "seed": 3,
        "cfg_scale": 3.0,
        "temperature": 0.5,
        "top_k": 900,
        "top_p": 0.97,
        "width": 256,
        "height": 256,
        "scales": 7,
        "device": "cpu",
        "dtype": "float32",
    }


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"unknown": 1}, TypeError, "Unknown Infinity options"),
        ({"cfg_scale": -1}, ValueError, "cfg_scale"),
        ({"temperature": 0}, ValueError, "temperature"),
        ({"top_k": -1}, ValueError, "top_k"),
        ({"top_p": 0}, ValueError, "top_p"),
        ({"top_p": 1.1}, ValueError, "top_p"),
        ({"width": 768, "height": 768}, ValueError, "256, 512 or 1024"),
        ({"width": 256, "height": 512}, ValueError, "must be equal"),
        ({"seed": -1}, ValueError, "seed"),
    ],
)
def test_options_are_validated_before_loading(tiny_infinity, options, error, match):
    backend = InfinityBackend(tiny_infinity)
    with pytest.raises(error, match=match):
        backend.validate_options(T2IRequest("x"), options)
    assert not backend.is_loaded


def test_request_parameters_are_isolated(tiny_infinity):
    backend = InfinityBackend(tiny_infinity)
    first = backend.validate_options(T2IRequest("x", {"width": 256, "height": 256}), {})
    second = backend.validate_options(T2IRequest("x"), {})
    assert first["width"] == 256 and second["width"] == 1024


def test_inference_restores_callers_autograd_mode(tiny_infinity):
    pipe = pipeline("t2i", backend="infinity", model=tiny_infinity)
    with torch.enable_grad():
        pipe(PROMPT, seed=1, width=256, height=256)
        assert torch.is_grad_enabled()


def test_schedules_and_catalog_defaults():
    assert [len(scale_schedule(size)) for size in (256, 512, 1024)] == [7, 10, 13]
    assert scale_schedule(256)[-1] == (1, 16, 16)
    assert model_info("infinity-2b").config["defaults"] == {
        "cfg_scale": 3.0,
        "temperature": 0.5,
        "top_k": 900,
        "top_p": 0.97,
        "width": 1024,
        "height": 1024,
    }
