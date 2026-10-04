import json

import pytest
import torch
from helpers import write_tiny_sdxl

from imagehub import pipeline
from imagehub.backends.stable_diffusion import initial_latents
from imagehub.backends.stable_diffusion_xl import StableDiffusionXLBackend
from imagehub.checkpoints import SNAPSHOT_FILE
from imagehub.exceptions import CheckpointError


@pytest.fixture(scope="module")
def tiny_sdxl(tmp_path_factory):
    return write_tiny_sdxl(tmp_path_factory.mktemp("tiny-sdxl"))


def test_generates_images_with_metadata(tiny_sdxl):
    result = pipeline("t2i", backend="stable-diffusion-xl", model=tiny_sdxl)(
        "a cat", seed=3, steps=4, width=64, height=48, num_images=2
    )
    assert len(result.images) == 2
    assert (result.images[0].width, result.images[0].height) == (64, 48)
    assert result.metadata["scheduler"] == "euler"
    assert result.metadata["seed"] == 3 and result.metadata["steps"] == 4
    assert result.metadata["device"] == "cpu" and result.metadata["dtype"] == "float32"


def test_seed_and_latents_are_reproducible(tiny_sdxl):
    pipe = pipeline("t2i", backend="sdxl-base-1.0", model=tiny_sdxl)
    a = pipe("a cat", seed=11, steps=3, width=32, height=32)
    b = pipe("a cat", seed=11, steps=3, width=32, height=32)
    latents = initial_latents(11, (1, 4, 4, 4))
    c = pipe("a cat", latents=latents, steps=3, width=32, height=32)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert torch.equal(a.images[0].tensor, c.images[0].tensor)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 30}, ValueError, "multiple of 8"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1, "latents": torch.zeros(1, 4, 4, 4)}, ValueError, "either"),
        ({"latents": torch.zeros(1, 4, 8, 8)}, ValueError, "shape"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"original_size": (1,)}, ValueError, "original_size"),
        ({"target_size": [32, 32]}, ValueError, "target_size"),
        ({"crops_coords_top_left": (0, -1)}, ValueError, "crops_coords_top_left"),
    ],
)
def test_invalid_options_fail_before_loading(tiny_sdxl, options, error, match):
    backend = StableDiffusionXLBackend(tiny_sdxl)
    with pytest.raises(error, match=match):
        pipeline("t2i", backend=backend)("a cat", **options)
    assert not backend.is_loaded


def test_missing_snapshot_hint_names_selected_model(tmp_path):
    pipe = pipeline("t2i", backend="sdxl-base-1.0", model=tmp_path / "nowhere")
    with pytest.raises(CheckpointError, match=r"imagehub\.pretrained sdxl-base-1.0 "):
        pipe("a cat")


def test_snapshot_metadata_and_mismatch_warning(tiny_sdxl, tmp_path):
    import shutil

    copy = shutil.copytree(tiny_sdxl, tmp_path / "copy")
    (copy / SNAPSHOT_FILE).write_text(json.dumps({"model_id": "other-sdxl", "revision": "f" * 40}))
    with pytest.warns(UserWarning, match="other-sdxl.*sdxl-base-1.0"):
        backend = StableDiffusionXLBackend(copy, model_id="sdxl-base-1.0").load()
    result = backend.infer(
        __import__("imagehub.types", fromlist=["T2IRequest"]).T2IRequest("cat"),
        seed=0,
        steps=1,
        width=32,
        height=32,
    )
    assert result.metadata["model_id"] == "other-sdxl"
    assert result.metadata["revision"] == "f" * 40


def _run(pipe, prompt="a cat", **options):
    return pipe(prompt, **({"steps": 2, "width": 32, "height": 32} | options))


def test_requests_are_isolated(tiny_sdxl):
    pipe = pipeline("t2i", backend="stable-diffusion-xl", model=tiny_sdxl)
    first = _run(pipe, seed=1)
    _run(
        pipe,
        "a hi cat",
        seed=2,
        negative_prompt="blurry",
        num_images=2,
        guidance=3.0,
        original_size=(64, 48),
        crops_coords_top_left=(1, 2),
        target_size=(32, 32),
    )
    again = _run(pipe, seed=1)
    assert torch.equal(first.images[0].tensor, again.images[0].tensor)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_sdxl, grad_before):
    pipe = pipeline("t2i", backend="stable-diffusion-xl", model=tiny_sdxl)
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        result = _run(pipe, seed=1)
        assert torch.is_grad_enabled() is grad_before
        assert not torch.is_inference_mode_enabled()
        assert not result.images[0].tensor.requires_grad
    finally:
        torch.set_grad_enabled(previous)
