import pytest
import torch
from helpers import write_tiny_omnigen

from imagehub import pipeline
from imagehub.backends.omnigen import OmniGenBackend, omnigen_sigmas
from imagehub.catalog import model_info
from imagehub.exceptions import CheckpointError
from imagehub.types import T2IRequest

PIL = pytest.importorskip("PIL.Image")
SMALL = {"steps": 2, "width": 128, "height": 96}
EDIT = "make <img><|image_1|></img> look like a drawing"


@pytest.fixture(scope="module")
def tiny_omnigen(tmp_path_factory):
    return write_tiny_omnigen(tmp_path_factory.mktemp("tiny-omnigen") / "snapshot")


def _picture(width=70, height=50):
    return PIL.new("RGB", (width, height), (200, 40, 40))


def test_generates_images_with_metadata(tiny_omnigen):
    result = pipeline("t2i", backend="omnigen", model=tiny_omnigen)("a cat", seed=3, **SMALL)
    assert len(result.images) == 1
    image = result.images[0]
    assert (image.width, image.height) == (128, 96)
    assert image.tensor.dtype == torch.uint8
    metadata = result.metadata
    assert metadata["seed"] == 3
    assert metadata["steps"] == 2
    assert metadata["guidance"] == 2.5
    assert metadata["image_guidance"] is None
    assert metadata["image_input"] is False
    assert metadata["scheduler"] == "flow_match_euler"
    assert metadata["device"] == "cpu" and metadata["dtype"] == "float32"
    assert "input_height" not in metadata


def test_image_input_changes_the_result_and_records_metadata(tiny_omnigen):
    pipe = pipeline("t2i", backend="omnigen", model=tiny_omnigen)
    plain = pipe(EDIT.replace("<img><|image_1|></img>", "it"), seed=1, **SMALL)
    edited = pipe(EDIT, image=_picture(), seed=1, **SMALL)
    assert not torch.equal(plain.images[0].tensor, edited.images[0].tensor)
    assert edited.metadata["image_input"] is True
    assert edited.metadata["image_guidance"] == 1.6
    assert (edited.metadata["input_height"], edited.metadata["input_width"]) == (48, 64)
    assert (edited.images[0].width, edited.images[0].height) == (128, 96)


def test_input_image_size_as_output(tiny_omnigen):
    result = pipeline("t2i", backend="omnigen", model=tiny_omnigen)(
        EDIT,
        image=_picture(),
        use_input_image_size_as_output=True,
        seed=1,
        steps=1,
        width=64,
        height=64,
    )
    assert (result.images[0].width, result.images[0].height) == (64, 48)
    assert (result.metadata["width"], result.metadata["height"]) == (64, 48)


def test_options_change_the_result(tiny_omnigen):
    pipe = pipeline("t2i", backend="omnigen", model=tiny_omnigen)
    base = pipe("a cat", seed=1, steps=3, width=64, height=64)
    other = pipe("a cat", seed=1, steps=3, width=64, height=64, guidance=5.0)
    assert not torch.equal(base.images[0].tensor, other.images[0].tensor)
    edit = pipe(EDIT, image=_picture(), seed=1, steps=2, width=64, height=64)
    scaled = pipe(EDIT, image=_picture(), seed=1, steps=2, width=64, height=64, image_guidance=3.0)
    assert not torch.equal(edit.images[0].tensor, scaled.images[0].tensor)


def test_seed_is_reproducible(tiny_omnigen):
    pipe = pipeline("t2i", backend="omnigen", model=tiny_omnigen)
    a = pipe(EDIT, image=_picture(), seed=11, **SMALL)
    b = pipe(EDIT, image=_picture(), seed=11, **SMALL)
    c = pipe(EDIT, image=_picture(), seed=12, **SMALL)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert not torch.equal(a.images[0].tensor, c.images[0].tensor)


def test_requests_are_isolated(tiny_omnigen):
    pipe = pipeline("t2i", backend="omnigen", model=tiny_omnigen)
    first = pipe("a cat", seed=1, **SMALL)
    pipe(EDIT, image=_picture(), seed=2, guidance=1.0, **SMALL | {"steps": 3})
    again = pipe("a cat", seed=1, **SMALL)
    assert torch.equal(first.images[0].tensor, again.images[0].tensor)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_omnigen, grad_before):
    pipe = pipeline("t2i", backend="omnigen", model=tiny_omnigen)
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        result = pipe(EDIT, image=_picture(), seed=1, **SMALL | {"steps": 1})
        assert torch.is_grad_enabled() is grad_before
        assert not torch.is_inference_mode_enabled()
        assert not result.images[0].tensor.requires_grad
    finally:
        torch.set_grad_enabled(previous)


@pytest.mark.parametrize(
    ("prompt", "options", "error", "match"),
    [
        ("a cat", {"stepz": 3}, TypeError, "stepz"),
        ("a cat", {"latents": torch.zeros(1)}, TypeError, "latents"),
        ("a cat", {"width": 1001}, ValueError, "multiple of 16"),
        ("a cat", {"height": 500}, ValueError, "multiple of 16"),
        ("a cat", {"width": 3088}, ValueError, "at most 3072"),
        ("a cat", {"steps": 0}, ValueError, "steps"),
        ("a cat", {"steps": 1001}, ValueError, "at most 1000"),
        ("a cat", {"guidance": float("nan")}, ValueError, "guidance"),
        ("a cat", {"guidance": -1.0}, ValueError, "guidance"),
        ("a cat", {"image_guidance": "1"}, ValueError, "image_guidance"),
        ("a cat", {"seed": -1}, ValueError, "seed"),
        ("a cat", {"max_input_image_size": 0}, ValueError, "max_input_image_size"),
        ("a cat", {"use_input_image_size_as_output": 1}, TypeError, "use_input_image_size"),
        ("a cat", {"use_input_image_size_as_output": True}, ValueError, "requires an input image"),
        ("a cat", {"image": 3}, TypeError, "PIL"),
        ("a cat <img><|image_1|></img>", {}, ValueError, "no image"),
        ("a cat", {"image": "PIL"}, ValueError, "placeholder"),
        ("a cat <|image_1|>", {"image": "PIL"}, ValueError, "placeholder"),
        (EDIT + " <img><|image_2|></img>", {"image": "PIL"}, ValueError, "only one input image"),
    ],
)
def test_invalid_options_fail_before_loading(tmp_path, prompt, options, error, match):
    if options.get("image") == "PIL":
        options = options | {"image": _picture()}
    backend = OmniGenBackend(tmp_path / "missing")
    with pytest.raises(error, match=match):
        backend.infer(T2IRequest(prompt), **options)
    assert not backend.is_loaded


def test_catalog_defaults(tmp_path):
    spec = model_info("omnigen-v1")
    assert spec.family == "omnigen" and spec.image_input == "optional"
    options = OmniGenBackend(tmp_path, model_id="omnigen-v1").validate_options(T2IRequest("x"), {})
    assert options["steps"] == 50
    assert options["guidance"] == 2.5
    assert options["image_guidance"] == 1.6
    assert options["width"] == options["height"] == 1024
    assert options["max_input_image_size"] == 1024
    assert options["use_input_image_size_as_output"] is False
    assert options["latent_shape"] == (1, 4, 128, 128)
    assert options["pixels"] is None


def test_request_parameters_are_options(tmp_path):
    backend = OmniGenBackend(tmp_path)
    options = backend.validate_options(T2IRequest("x", {"steps": 5}), {"guidance": 2.0})
    assert options["steps"] == 5 and options["guidance"] == 2.0
    with pytest.raises(TypeError, match="bogus"):
        backend.validate_options(T2IRequest("x", {"bogus": 1}), {})


def test_snapshot_limits_are_checked_before_loading(tiny_omnigen):
    backend = OmniGenBackend(tiny_omnigen)
    with pytest.raises(ValueError, match="height must be at most 256"):
        backend.infer(T2IRequest("a cat"), height=272, width=64)
    with pytest.raises(ValueError, match="at most 256 pixels"):
        backend.infer(T2IRequest(EDIT), image=_picture(1000, 2), width=64, height=64)
    assert not backend.is_loaded


def test_incomplete_snapshot_is_reported(tmp_path):
    with pytest.raises(CheckpointError, match="not an OmniGen snapshot"):
        OmniGenBackend(tmp_path, model_id="omnigen-v1").load()


def test_sigma_schedule_matches_linspace():
    np = pytest.importorskip("numpy")
    for steps in (1, 2, 3, 7, 25, 50, 333, 1000):
        assert omnigen_sigmas(steps) == np.linspace(1, 0, steps + 1)[:steps].tolist()
