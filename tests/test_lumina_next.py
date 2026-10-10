import pytest
import torch
from helpers import write_tiny_lumina_next

from imagehub import pipeline
from imagehub.backends.lumina_next import (
    LuminaNextBackend,
    LuminaNextComponents,
    denoise,
    encode_prompt,
)
from imagehub.catalog import model_info
from imagehub.exceptions import CheckpointError
from imagehub.types import T2IRequest

SMALL = {"steps": 2, "width": 128, "height": 96}


@pytest.fixture(scope="module")
def tiny_lumina_next(tmp_path_factory):
    return write_tiny_lumina_next(tmp_path_factory.mktemp("tiny-lumina-next") / "snapshot")


def test_generates_images_with_metadata(tiny_lumina_next):
    result = pipeline("t2i", backend="lumina-next", model=tiny_lumina_next)(
        "a cat", seed=3, **SMALL
    )
    assert len(result.images) == 1
    image = result.images[0]
    assert (image.width, image.height) == (128, 96)
    assert image.tensor.dtype == torch.uint8
    metadata = result.metadata
    assert metadata["seed"] == 3
    assert metadata["steps"] == 2
    assert metadata["guidance"] == 4.0
    assert metadata["negative_prompt"] == ""
    assert metadata["scaling_watershed"] == 1.0
    assert metadata["proportional_attn"] is True
    assert metadata["scheduler"] == "flow_match_euler"
    assert metadata["device"] == "cpu" and metadata["dtype"] == "float32"


def test_options_change_the_result(tiny_lumina_next):
    pipe = pipeline("t2i", backend="lumina-next", model=tiny_lumina_next)
    base = pipe("a cat", seed=1, steps=3, width=64, height=64)
    for override in (
        {"guidance": 1.0},
        {"negative_prompt": "blurry"},
        {"proportional_attn": False},
    ):
        other = pipe("a cat", seed=1, steps=3, width=64, height=64, **override)
        assert not torch.equal(base.images[0].tensor, other.images[0].tensor), override


def test_watershed_switches_the_rotary_scaling(tiny_lumina_next):
    components = LuminaNextComponents.load(
        tiny_lumina_next, device=torch.device("cpu"), dtype=torch.float32
    )
    noise = torch.randn(1, 4, 12, 24, generator=torch.Generator().manual_seed(0))

    def run(watershed):
        return denoise(
            components,
            prompt="a cat",
            negative_prompt="",
            steps=4,
            guidance=4.0,
            scaling_watershed=watershed,
            proportional_attn=True,
            latents=noise,
        )

    # Every step is below 1.0, so the default always stretches the positions linearly.
    assert torch.equal(run(1.0), run(5.0))
    assert not torch.equal(run(1.0), run(0.0))
    assert not torch.equal(run(1.0), run(0.5))


def test_prompt_is_lower_cased(tiny_lumina_next):
    pipe = pipeline("t2i", backend="lumina-next", model=tiny_lumina_next)
    lower = pipe("a cat", seed=1, **SMALL)
    upper = pipe("  A CAT ", seed=1, **SMALL)
    assert torch.equal(lower.images[0].tensor, upper.images[0].tensor)


def test_seed_is_reproducible(tiny_lumina_next):
    pipe = pipeline("t2i", backend="lumina-next", model=tiny_lumina_next)
    a = pipe("a cat", seed=11, **SMALL)
    b = pipe("a cat", seed=11, **SMALL)
    c = pipe("a cat", seed=12, **SMALL)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert not torch.equal(a.images[0].tensor, c.images[0].tensor)


def test_requests_are_isolated(tiny_lumina_next):
    pipe = pipeline("t2i", backend="lumina-next", model=tiny_lumina_next)
    first = pipe("a cat", seed=1, **SMALL)
    pipe("a hi cat", seed=2, guidance=2.0, negative_prompt="blurry", **SMALL | {"steps": 3})
    again = pipe("a cat", seed=1, **SMALL)
    assert torch.equal(first.images[0].tensor, again.images[0].tensor)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_lumina_next, grad_before):
    pipe = pipeline("t2i", backend="lumina-next", model=tiny_lumina_next)
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        result = pipe("a cat", seed=1, **SMALL | {"steps": 1})
        assert torch.is_grad_enabled() is grad_before
        assert not torch.is_inference_mode_enabled()
        assert not result.images[0].tensor.requires_grad
    finally:
        torch.set_grad_enabled(previous)


def test_prompt_lengths_are_padded_to_a_multiple_of_eight(tiny_lumina_next):
    components = LuminaNextComponents.load(
        tiny_lumina_next, device=torch.device("cpu"), dtype=torch.float32
    )
    cpu = torch.device("cpu")
    embeds, mask, negative, negative_mask = encode_prompt(components, "a cat", "", cpu, True)
    assert embeds.shape[1] == 8 and mask.sum() == 4  # start, two words, end
    assert negative is not None and negative_mask is not None
    assert negative.shape == embeds.shape and negative_mask.sum() == 2
    long_prompt = " ".join(["cat"] * 20)
    embeds, mask, _, _ = encode_prompt(components, long_prompt, "x", cpu, False)
    assert embeds.shape[1] % 8 == 0 and embeds.shape[1] >= mask.sum()


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"max_sequence_length": 256}, TypeError, "max_sequence_length"),
        ({"clean_caption": True}, TypeError, "clean_caption"),
        ({"latents": torch.zeros(1)}, TypeError, "latents"),
        ({"width": 1001}, ValueError, "multiple of 16"),
        ({"height": 500}, ValueError, "multiple of 16"),
        ({"steps": 0}, ValueError, "steps"),
        ({"steps": 1001}, ValueError, "at most 1000"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"guidance": -1.0}, ValueError, "guidance"),
        ({"scaling_watershed": float("inf")}, ValueError, "scaling_watershed"),
        ({"scaling_watershed": "1"}, ValueError, "scaling_watershed"),
        ({"proportional_attn": 1}, TypeError, "proportional_attn"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"seed": -1}, ValueError, "seed"),
    ],
)
def test_invalid_options_fail_before_loading(tmp_path, options, error, match):
    backend = LuminaNextBackend(tmp_path / "missing")
    with pytest.raises(error, match=match):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded


def test_rotary_table_limit_is_checked_before_loading(tiny_lumina_next):
    backend = LuminaNextBackend(tiny_lumina_next)
    with pytest.raises(ValueError, match="width must be at most 6144"):
        backend.infer(T2IRequest("a cat"), width=6160)
    assert not backend.is_loaded


def test_catalog_defaults(tmp_path):
    spec = model_info("lumina-next")
    assert spec.family == "lumina-next"
    options = LuminaNextBackend(tmp_path, model_id="lumina-next").validate_options(
        T2IRequest("x"), {}
    )
    assert options["steps"] == 30
    assert options["guidance"] == 4.0
    assert options["width"] == options["height"] == 1024
    assert options["negative_prompt"] == ""
    assert options["scaling_watershed"] == 1.0
    assert options["proportional_attn"] is True
    assert options["latent_shape"] == (1, 4, 128, 128)


def test_none_negative_prompt_selects_the_default(tmp_path):
    options = LuminaNextBackend(tmp_path).validate_options(
        T2IRequest("x"), {"negative_prompt": None}
    )
    assert options["negative_prompt"] == ""


def test_request_parameters_are_options(tmp_path):
    backend = LuminaNextBackend(tmp_path)
    options = backend.validate_options(T2IRequest("x", {"steps": 5}), {"guidance": 2.0})
    assert options["steps"] == 5 and options["guidance"] == 2.0
    with pytest.raises(TypeError, match="bogus"):
        backend.validate_options(T2IRequest("x", {"bogus": 1}), {})


def test_incomplete_snapshot_is_reported(tmp_path):
    with pytest.raises(CheckpointError, match="not a Lumina-Next snapshot"):
        LuminaNextBackend(tmp_path, model_id="lumina-next").load()


def test_lumina_image_2_snapshot_is_refused(tmp_path, tiny_lumina_next):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(tiny_lumina_next, copy)
    (copy / "model_index.json").write_text('{"_class_name": "Lumina2Pipeline"}')
    with pytest.raises(CheckpointError, match="Unsupported Lumina-Next pipeline class"):
        LuminaNextBackend(copy).load()
