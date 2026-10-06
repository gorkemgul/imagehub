import pytest
import torch
from helpers import write_tiny_sd3

from imagehub import pipeline
from imagehub.backends.stable_diffusion_3 import StableDiffusion3Backend
from imagehub.catalog import model_info
from imagehub.exceptions import CheckpointError


@pytest.fixture(scope="module")
def tiny_sd3(tmp_path_factory):
    return write_tiny_sd3(tmp_path_factory.mktemp("tiny-sd3"))


def test_generates_images_with_metadata(tiny_sd3):
    result = pipeline("t2i", backend="stable-diffusion-3", model=tiny_sd3)(
        "a cat", seed=3, steps=2, width=32, height=32, num_images=2
    )
    assert len(result.images) == 2
    assert (result.images[0].width, result.images[0].height) == (32, 32)
    assert result.metadata["seed"] == 3 and result.metadata["steps"] == 2
    assert result.metadata["scheduler"] == "flow_match_euler"


def test_seed_and_latents_are_reproducible(tiny_sd3):
    pipe = pipeline("t2i", backend="sd3.5-medium", model=tiny_sd3)
    a = pipe("a cat", seed=11, steps=2, width=32, height=32)
    b = pipe("a cat", seed=11, steps=2, width=32, height=32)
    latents = torch.randn(1, 4, 4, 4, generator=torch.Generator("cpu").manual_seed(11))
    c = pipe("a cat", latents=latents, steps=2, width=32, height=32)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert torch.equal(a.images[0].tensor, c.images[0].tensor)


def test_requests_are_isolated(tiny_sd3):
    pipe = pipeline("t2i", backend="stable-diffusion-3", model=tiny_sd3)
    first = pipe("a cat", seed=1, steps=2, width=32, height=32)
    pipe(
        "a hi cat",
        seed=2,
        steps=3,
        guidance=4.0,
        negative_prompt="blurry",
        width=32,
        height=32,
        num_images=2,
        max_sequence_length=8,
    )
    again = pipe("a cat", seed=1, steps=2, width=32, height=32)
    assert torch.equal(first.images[0].tensor, again.images[0].tensor)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_sd3, grad_before):
    pipe = pipeline("t2i", backend="stable-diffusion-3", model=tiny_sd3)
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        result = pipe("a cat", seed=1, steps=2, width=32, height=32)
        assert torch.is_grad_enabled() is grad_before
        assert not torch.is_inference_mode_enabled()
        assert not result.images[0].tensor.requires_grad
    finally:
        torch.set_grad_enabled(previous)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 30}, ValueError, "multiple of 16"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1, "latents": torch.zeros(1, 4, 4, 4)}, ValueError, "either"),
        ({"latents": torch.zeros(1, 4, 8, 8)}, ValueError, "shape"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"max_sequence_length": 0}, ValueError, "max_sequence_length"),
    ],
)
def test_invalid_options_fail_before_loading(tiny_sd3, options, error, match):
    backend = StableDiffusion3Backend(tiny_sd3)
    with pytest.raises(error, match=match):
        pipeline("t2i", backend=backend)("a cat", **options)
    assert not backend.is_loaded


def test_catalog_entries_and_snapshot_hint(tmp_path):
    sd3_medium = model_info("sd3-medium")
    medium = model_info("sd3.5-medium")
    large = model_info("sd3.5-large")
    assert sd3_medium.family == medium.family == large.family == "stable-diffusion-3"
    assert sd3_medium.config == {"defaults": {"steps": 28, "guidance": 7.0}}
    assert medium.config == {"defaults": {"steps": 40, "guidance": 4.5}}
    assert large.config == {"defaults": {"steps": 28, "guidance": 3.5}}
    assert len(medium.files) == 26 and len(large.files) == 28
    pipe = pipeline("t2i", backend="sd3.5-medium", model=tmp_path / "nowhere")
    with pytest.raises(CheckpointError, match=r"imagehub\.pretrained sd3\.5-medium "):
        pipe("a cat")
