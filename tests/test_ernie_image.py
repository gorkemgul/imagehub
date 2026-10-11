import pytest
import torch
from helpers import write_tiny_ernie_image

from imagehub import pipeline
from imagehub.backends.ernie_image import ErnieImageBackend, ErnieImageComponents, denoise
from imagehub.catalog import model_info
from imagehub.exceptions import CheckpointError


@pytest.fixture(scope="module")
def tiny_ernie(tmp_path_factory):
    pytest.importorskip("transformers")
    return write_tiny_ernie_image(tmp_path_factory.mktemp("tiny-ernie"))


def test_generates_images_with_metadata(tiny_ernie):
    result = pipeline("t2i", backend="ernie-image", model=tiny_ernie)(
        "a cat", seed=3, steps=2, width=16, height=32, num_images=2
    )
    assert len(result.images) == 2
    assert (result.images[0].width, result.images[0].height) == (16, 32)
    assert result.metadata["seed"] == 3 and result.metadata["steps"] == 2
    assert result.metadata["guidance"] == 4.0
    assert result.metadata["scheduler"] == "flow_match_euler"
    assert result.metadata["prompt_enhancer"] is False


def test_seed_and_latents_are_reproducible(tiny_ernie):
    pipe = pipeline("t2i", backend="ernie-image", model=tiny_ernie)
    a = pipe("a cat", seed=11, steps=2, width=16, height=16)
    b = pipe("a cat", seed=11, steps=2, width=16, height=16)
    latents = torch.randn(1, 16, 4, 4, generator=torch.Generator().manual_seed(11))
    c = pipe("a cat", latents=latents, steps=2, width=16, height=16)
    d = pipe("a cat", seed=12, steps=2, width=16, height=16)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert torch.equal(a.images[0].tensor, c.images[0].tensor)
    assert not torch.equal(a.images[0].tensor, d.images[0].tensor)


def test_guidance_of_one_disables_the_negative_branch(tiny_ernie):
    components = ErnieImageComponents.load(
        tiny_ernie, device=torch.device("cpu"), dtype=torch.float32
    )
    noise = torch.randn(1, 16, 4, 4, generator=torch.Generator().manual_seed(1))

    def run(guidance, negative):
        return denoise(
            components,
            prompt="a cat",
            negative_prompt=negative,
            steps=2,
            guidance=guidance,
            latents=noise.clone(),
        )

    assert torch.equal(run(1.0, ""), run(1.0, "dog"))
    assert not torch.equal(run(4.0, ""), run(1.0, ""))
    assert not torch.equal(run(4.0, ""), run(4.0, "dog"))


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_ernie, grad_before):
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        pipe = pipeline("t2i", backend="ernie-image", model=tiny_ernie)
        pipe("a cat", seed=1, steps=1, width=16, height=16)
        assert torch.is_grad_enabled() is grad_before
    finally:
        torch.set_grad_enabled(previous)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"use_pe": True}, TypeError, "use_pe"),
        ({"width": 18}, ValueError, "multiple of 4"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"guidance": -1.0}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"seed": 1, "latents": torch.zeros(1, 16, 4, 4)}, ValueError, "either"),
        ({"latents": torch.zeros(1, 16, 8, 8)}, ValueError, "shape"),
    ],
)
def test_invalid_options_fail_before_loading(tiny_ernie, options, error, match):
    backend = ErnieImageBackend(tiny_ernie)
    with pytest.raises(error, match=match):
        pipeline("t2i", backend=backend)("a cat", **options)
    assert not backend.is_loaded


def test_catalog_entry_and_snapshot_hint(tmp_path):
    spec = model_info("ernie-image")
    pipe = pipeline("t2i", backend="ernie-image", model=tmp_path / "nowhere")
    with pytest.raises(CheckpointError, match=r"imagehub\.pretrained ernie-image "):
        pipe("a cat")
    assert spec.family == "ernie-image"
