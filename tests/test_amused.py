import pytest
import torch
from helpers import write_tiny_amused

from imagehub import pipeline
from imagehub.backends.amused import AmusedBackend, AmusedComponents, denoise
from imagehub.catalog import model_info
from imagehub.exceptions import CheckpointError
from imagehub.types import T2IRequest

SMALL = {"steps": 2}


@pytest.fixture(scope="module")
def tiny_amused(tmp_path_factory):
    return write_tiny_amused(tmp_path_factory.mktemp("tiny-amused") / "snapshot")


@pytest.fixture(scope="module")
def tiny_amused_512(tmp_path_factory):
    return write_tiny_amused(
        tmp_path_factory.mktemp("tiny-amused-512") / "snapshot", downsample=True
    )


def test_generates_images_with_metadata(tiny_amused):
    result = pipeline("t2i", backend="amused", model=tiny_amused)("a cat", seed=3, **SMALL)
    assert len(result.images) == 1
    image = result.images[0]
    assert (image.width, image.height) == (32, 32)
    assert image.tensor.dtype == torch.uint8
    metadata = result.metadata
    assert metadata["seed"] == 3
    assert metadata["steps"] == 2
    assert metadata["guidance"] == 10.0
    assert metadata["negative_prompt"] == ""
    assert metadata["aesthetic_score"] == 6
    assert metadata["crop_coord"] == [0, 0]
    assert metadata["temperature"] == [2, 0]
    assert metadata["scheduler"] == "amused"
    assert metadata["device"] == "cpu" and metadata["dtype"] == "float32"


def test_non_square_sizes(tiny_amused):
    result = pipeline("t2i", backend="amused", model=tiny_amused)(
        "a cat", seed=1, width=64, height=32, **SMALL
    )
    image = result.images[0]
    assert (image.width, image.height) == (64, 32)


def test_options_change_the_tokens(tiny_amused):
    components = AmusedComponents.load(tiny_amused, device=torch.device("cpu"), dtype=torch.float32)

    def tokens(**override):
        options = {
            "prompt": "a cat",
            "negative_prompt": "",
            "steps": 3,
            "guidance": 10.0,
            "height": 64,
            "width": 64,
            "generator": torch.Generator("cpu").manual_seed(1),
        }
        return denoise(components, **(options | override))

    base = tokens()
    assert torch.equal(base, tokens())
    assert base.shape == (1, 4, 4)
    for override in (
        {"guidance": 1.0},
        {"negative_prompt": "blurry"},
        {"prompt": "a dog"},
        {"aesthetic_score": 2},
        {"crop_coord": (3, 4)},
        {"temperature": 5},
        {"steps": 4},
    ):
        assert not torch.equal(base, tokens(**override)), override


def test_seed_is_reproducible(tiny_amused):
    pipe = pipeline("t2i", backend="amused", model=tiny_amused)
    a = pipe("a cat", seed=11, **SMALL)
    b = pipe("a cat", seed=11, **SMALL)
    c = pipe("a cat", seed=12, **SMALL)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert not torch.equal(a.images[0].tensor, c.images[0].tensor)


def test_resampling_transformer_generates(tiny_amused_512):
    result = pipeline("t2i", backend="amused", model=tiny_amused_512)("a cat", seed=1, **SMALL)
    assert (result.images[0].width, result.images[0].height) == (32, 32)


def test_requests_are_isolated(tiny_amused):
    pipe = pipeline("t2i", backend="amused", model=tiny_amused)
    first = pipe("a cat", seed=1, **SMALL)
    pipe("a hi cat", seed=2, guidance=2.0, negative_prompt="blurry", steps=3)
    again = pipe("a cat", seed=1, **SMALL)
    assert torch.equal(first.images[0].tensor, again.images[0].tensor)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_amused, grad_before):
    pipe = pipeline("t2i", backend="amused", model=tiny_amused)
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        result = pipe("a cat", seed=1, steps=1)
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
        ({"width": 0}, ValueError, "width"),
        ({"steps": 0}, ValueError, "steps"),
        ({"steps": 1001}, ValueError, "at most 1000"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"guidance": -1.0}, ValueError, "guidance"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"aesthetic_score": "6"}, ValueError, "aesthetic_score"),
        ({"crop_coord": (1,)}, ValueError, "crop_coord"),
        ({"crop_coord": (1.5, 0)}, ValueError, "crop_coord"),
        ({"temperature": (1, 2, 3)}, ValueError, "temperature"),
        ({"temperature": -1}, ValueError, "temperature"),
        ({"seed": -1}, ValueError, "seed"),
    ],
)
def test_invalid_options_fail_before_loading(tmp_path, options, error, match):
    backend = AmusedBackend(tmp_path / "missing")
    with pytest.raises(error, match=match):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded


def test_resampling_snapshot_needs_even_token_grids(tiny_amused_512):
    backend = AmusedBackend(tiny_amused_512)
    with pytest.raises(ValueError, match="multiple of 32"):
        backend.infer(T2IRequest("a cat"), width=48)
    assert not backend.is_loaded


@pytest.mark.parametrize(("model_id", "size"), [("amused-256", 256), ("amused-512", 512)])
def test_catalog_defaults(tmp_path, model_id, size):
    spec = model_info(model_id)
    assert spec.family == "amused"
    options = AmusedBackend(tmp_path, model_id=model_id).validate_options(T2IRequest("x"), {})
    assert options["steps"] == 12
    assert options["guidance"] == 10.0
    assert options["width"] == options["height"] == size
    assert options["negative_prompt"] == ""
    assert options["aesthetic_score"] == 6
    assert options["crop_coord"] == (0, 0)
    assert options["temperature"] == (2, 0)


def test_snapshot_config_gives_the_default_size(tiny_amused):
    options = AmusedBackend(tiny_amused).validate_options(T2IRequest("x"), {})
    assert options["width"] == options["height"] == 32


def test_request_parameters_are_options(tmp_path):
    backend = AmusedBackend(tmp_path)
    options = backend.validate_options(T2IRequest("x", {"steps": 5}), {"guidance": 2.0})
    assert options["steps"] == 5 and options["guidance"] == 2.0
    with pytest.raises(TypeError, match="bogus"):
        backend.validate_options(T2IRequest("x", {"bogus": 1}), {})


def test_incomplete_snapshot_is_reported(tmp_path):
    with pytest.raises(CheckpointError, match="not an aMUSEd snapshot"):
        AmusedBackend(tmp_path, model_id="amused-256").load()
