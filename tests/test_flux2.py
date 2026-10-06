import pytest
import torch
from helpers import write_tiny_flux2

from imagehub import pipeline
from imagehub.backends.flux2 import Flux2Backend, compute_empirical_mu
from imagehub.catalog import model_info
from imagehub.exceptions import CheckpointError


@pytest.fixture(scope="module")
def tiny_flux2(tmp_path_factory):
    return write_tiny_flux2(tmp_path_factory.mktemp("tiny-flux2"))


def test_generates_images_with_metadata(tiny_flux2):
    result = pipeline("t2i", backend="flux2", model=tiny_flux2)(
        "a cat", seed=3, steps=2, width=16, height=16, num_images=2
    )
    assert len(result.images) == 2
    assert (result.images[0].width, result.images[0].height) == (16, 16)
    assert result.metadata["seed"] == 3 and result.metadata["steps"] == 2
    assert result.metadata["scheduler"] == "flow_match_euler"


def test_seed_and_latents_are_reproducible(tiny_flux2):
    pipe = pipeline("t2i", backend="flux2-klein-4b", model=tiny_flux2)
    a = pipe("a cat", seed=11, steps=2, width=16, height=16)
    b = pipe("a cat", seed=11, steps=2, width=16, height=16)
    latents = torch.randn(1, 16, 4, 4, generator=torch.Generator().manual_seed(11))
    c = pipe("a cat", latents=latents, steps=2, width=16, height=16)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert torch.equal(a.images[0].tensor, c.images[0].tensor)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_flux2, grad_before):
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        pipe = pipeline("t2i", backend="flux2", model=tiny_flux2)
        pipe("a cat", seed=1, steps=1, width=16, height=16)
        assert torch.is_grad_enabled() is grad_before
    finally:
        torch.set_grad_enabled(previous)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 15}, ValueError, "multiple of 16"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1, "latents": torch.zeros(1, 16, 4, 4)}, ValueError, "either"),
        ({"latents": torch.zeros(1, 16, 8, 8)}, ValueError, "shape"),
        ({"negative_prompt": "blurry"}, TypeError, "negative_prompt"),
        ({"max_sequence_length": 0}, ValueError, "max_sequence_length"),
    ],
)
def test_invalid_options_fail_before_loading(tiny_flux2, options, error, match):
    backend = Flux2Backend(tiny_flux2)
    with pytest.raises(error, match=match):
        pipeline("t2i", backend=backend)("a cat", **options)
    assert not backend.is_loaded


def test_empirical_mu_matches_reference_formula():
    assert compute_empirical_mu(4096, 4) == pytest.approx(2.291179894115571)
    assert compute_empirical_mu(4301, 4) == pytest.approx(1.18469693)


def test_catalog_entry_and_snapshot_hint(tmp_path):
    spec = model_info("flux2-klein-4b")
    assert spec.family == "flux2" and spec.catalog_id == "flux-2"
    assert spec.revision == "e7b7dc27f91deacad38e78976d1f2b499d76a294"
    assert len(spec.files) == 18
    pipe = pipeline("t2i", backend="flux2-klein-4b", model=tmp_path / "nowhere")
    with pytest.raises(CheckpointError, match=r"imagehub\.pretrained flux2-klein-4b "):
        pipe("a cat")
