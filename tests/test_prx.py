"""PRX backend option and catalog tests."""

import pytest
import torch
from helpers import write_tiny_prx

from imagehub import pipeline
from imagehub.backends.prx import PRXBackend
from imagehub.catalog import model_info
from imagehub.registry import available_backends
from imagehub.types import T2IRequest


@pytest.fixture(scope="module")
def tiny_prx(tmp_path_factory):
    return write_tiny_prx(tmp_path_factory.mktemp("tiny-prx") / "snapshot")


def test_generates_repeatably_and_restores_autograd(tiny_prx) -> None:
    pipe = pipeline("t2i", backend="prx", model=tiny_prx)
    options = {
        "steps": 2,
        "width": 64,
        "height": 48,
        "max_sequence_length": 24,
        "use_resolution_binning": False,
    }
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(True)
        first = pipe("a cat", seed=11, **options)
        pipe("a dog", seed=2, negative_prompt="blurry", **options)
        second = pipe("a cat", seed=11, **options)
        different = pipe("a cat", seed=12, **options)
        assert torch.is_grad_enabled()
        assert torch.equal(first.images[0].tensor, second.images[0].tensor)
        assert not torch.equal(first.images[0].tensor, different.images[0].tensor)
        assert (first.images[0].width, first.images[0].height) == (64, 48)
        assert first.metadata["seed"] == 11
    finally:
        torch.set_grad_enabled(previous)


def test_prx_catalog_entries() -> None:
    for model_id, size in (("prx-1024-t2i-beta", 1024), ("prx-512-t2i-sft", 512)):
        spec = model_info(model_id)
        assert spec.family == spec.catalog_id == "prx"
        assert spec.revision and len(spec.revision) == 40
        assert spec.config["defaults"] == {"width": size, "height": size}
        assert "tokenizer/tokenizer.json" in spec.files
        assert "text_encoder/model-00003-of-00003.safetensors" in spec.files
    assert {"prx", "prx-1024-t2i-beta", "prx-512-t2i-sft"} <= set(available_backends("t2i")["t2i"])


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        (
            {"width": 1000, "use_resolution_binning": False},
            ValueError,
            "multiples of 16",
        ),
        ({"height": 0}, ValueError, "height"),
        ({"steps": 1001}, ValueError, "at most 1000"),
        ({"guidance": 0.0}, ValueError, "guidance"),
        ({"negative_prompt": None}, TypeError, "negative_prompt"),
        ({"skip_text_cleaning": 1}, TypeError, "skip_text_cleaning"),
        ({"use_resolution_binning": 1}, TypeError, "use_resolution_binning"),
        ({"seed": -1}, ValueError, "seed"),
    ],
)
def test_invalid_options_fail_before_loading(tmp_path, options, error, match) -> None:
    backend = PRXBackend(tmp_path / "missing")
    with pytest.raises(error, match=match):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded


def test_published_defaults() -> None:
    backend = PRXBackend("missing", model_id="prx-1024-t2i-beta")
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert (options["steps"], options["guidance"]) == (28, 4.0)
    assert (options["width"], options["height"]) == (1024, 1024)
    assert options["latent_shape"] == (1, 16, 128, 128)
