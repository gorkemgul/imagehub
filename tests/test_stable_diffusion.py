import json

import pytest
import torch
from helpers import write_tiny_sd

from imagehub import pipeline
from imagehub.backends.stable_diffusion import (
    StableDiffusionBackend,
    initial_latents,
    resolve_device,
)
from imagehub.checkpoints import SNAPSHOT_FILE
from imagehub.exceptions import CheckpointError


@pytest.fixture(scope="module")
def tiny_sd(tmp_path_factory):
    return write_tiny_sd(tmp_path_factory.mktemp("tiny-sd"))


def test_generates_images_with_metadata(tiny_sd):
    with pipeline("t2i", backend="stable-diffusion", model=tiny_sd) as pipe:
        result = pipe("a cat", seed=3, steps=4, width=64, height=48, num_images=2)
    assert len(result.images) == 2
    assert (result.images[0].width, result.images[0].height) == (64, 48)
    meta = result.metadata
    assert meta["seed"] == 3 and meta["steps"] == 4 and meta["scheduler"] == "pndm"
    assert meta["model_id"] is None and meta["revision"] is None
    assert meta["device"] == "cpu" and meta["dtype"] == "float32"


def test_seed_is_reproducible_and_reported(tiny_sd):
    pipe = pipeline("t2i", backend="sd-v1-5", model=tiny_sd)
    a = pipe("a cat", seed=11, steps=3, width=32, height=32)
    b = pipe("a cat", seed=11, steps=3, width=32, height=32)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    random = pipe("a cat", steps=3, width=32, height=32)
    assert isinstance(random.metadata["seed"], int)


def test_latents_option_matches_seed(tiny_sd):
    pipe = pipeline("t2i", backend="stable-diffusion", model=tiny_sd)
    by_seed = pipe("a cat", seed=5, steps=3, width=32, height=32)
    latents = initial_latents(5, (1, 4, 4, 4))
    by_latents = pipe("a cat", latents=latents, steps=3, width=32, height=32)
    assert torch.equal(by_seed.images[0].tensor, by_latents.images[0].tensor)
    assert by_latents.metadata["seed"] is None


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 30}, ValueError, "multiple of 8"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1, "latents": torch.zeros(1, 4, 64, 64)}, ValueError, "either"),
        ({"latents": torch.zeros(1, 4, 8, 8)}, ValueError, "shape"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
    ],
)
def test_invalid_options_fail_before_loading(tiny_sd, options, error, match):
    backend = StableDiffusionBackend(tiny_sd)
    with pytest.raises(error, match=match):
        pipeline("t2i", backend=backend)("a cat", **options)
    assert not backend.is_loaded


def test_missing_snapshot_points_to_pretrained(tmp_path):
    pipe = pipeline("t2i", backend="sd-v1-5", model=tmp_path / "nowhere")
    with pytest.raises(CheckpointError, match="python -m imagehub.pretrained"):
        pipe("a cat")


def test_snapshot_metadata(tiny_sd, tmp_path):
    import shutil

    copy = shutil.copytree(tiny_sd, tmp_path / "copy")
    manifest = {"model_id": "sd-v1-5", "revision": "f" * 40, "files": {}}
    (copy / SNAPSHOT_FILE).write_text(json.dumps(manifest))
    result = pipeline("t2i", backend="sd-v1-5", model=copy)(
        "a cat", seed=0, steps=2, width=32, height=32
    )
    assert result.metadata["model_id"] == "sd-v1-5"
    assert result.metadata["revision"] == "f" * 40


def test_missing_snapshot_hint_names_selected_model(tmp_path):
    pipe = pipeline("t2i", backend="sd-v1-4", model=tmp_path / "nowhere")
    with pytest.raises(CheckpointError, match=r"imagehub\.pretrained sd-v1-4 "):
        pipe("a cat")


def test_missing_file_without_filename_still_explains(tiny_sd, tmp_path, monkeypatch):
    import shutil

    from imagehub.backends import stable_diffusion

    copy = shutil.copytree(tiny_sd, tmp_path / "copy")

    def broken(*args, **kwargs):
        raise FileNotFoundError("no such thing")

    monkeypatch.setattr(stable_diffusion.UNet2DConditionModel, "from_pretrained", broken)
    with pytest.raises(CheckpointError, match="no such thing is missing"):
        StableDiffusionBackend(copy, model_id="sd-v1-2").load()


def test_selected_model_differing_from_snapshot_warns(tiny_sd, tmp_path):
    import shutil

    copy = shutil.copytree(tiny_sd, tmp_path / "copy")
    (copy / SNAPSHOT_FILE).write_text(json.dumps({"model_id": "sd-v1-5", "revision": "f" * 40}))
    with pytest.warns(UserWarning, match="sd-v1-5.*sd-v1-4"):
        StableDiffusionBackend(copy, model_id="sd-v1-4").load()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        StableDiffusionBackend(copy, model_id="sd-v1-5").load()
        StableDiffusionBackend(copy).load()


def test_device_and_dtype_validation(tiny_sd):
    with pytest.raises(ValueError, match="Available: cpu"):
        resolve_device("quantum")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="not available"):
            StableDiffusionBackend(tiny_sd, device="cuda")
    with pytest.raises(ValueError, match="dtype"):
        StableDiffusionBackend(tiny_sd, dtype=torch.int8)


def run(pipe, prompt="a cat", **options):
    options = {"steps": 3, "width": 32, "height": 32} | options
    return pipe(prompt, **options)


def same(a, b):
    return len(a.images) == len(b.images) and all(
        torch.equal(x.tensor, y.tensor) for x, y in zip(a.images, b.images, strict=True)
    )


def test_requests_do_not_leak_into_each_other(tiny_sd):
    pipe = pipeline("t2i", backend="stable-diffusion", model=tiny_sd)
    first = run(pipe, seed=1)
    other = run(
        pipe, "a hi cat", seed=2, negative_prompt="blurry", num_images=2, steps=4, guidance=3.0
    )
    again = run(pipe, seed=1)
    assert len(other.images) == 2
    assert same(first, again)
    assert not same(first, other)


def test_independent_pipelines_interleaved_match_solo_runs(tiny_sd):
    a_solo = run(pipeline("t2i", backend="stable-diffusion", model=tiny_sd), seed=1)
    b_solo = run(pipeline("t2i", backend="stable-diffusion", model=tiny_sd), "hi", seed=2, steps=5)
    one = pipeline("t2i", backend="stable-diffusion", model=tiny_sd)
    two = pipeline("t2i", backend="stable-diffusion", model=tiny_sd)
    a = run(one, seed=1)
    b = run(two, "hi", seed=2, steps=5)
    a_after = run(one, seed=1)
    b_after = run(two, "hi", seed=2, steps=5)
    assert same(a, a_solo) and same(a_after, a_solo)
    assert same(b, b_solo) and same(b_after, b_solo)


def test_result_does_not_depend_on_earlier_steps_or_sizes(tiny_sd):
    fresh = run(pipeline("t2i", backend="stable-diffusion", model=tiny_sd), seed=7, steps=4)
    pipe = pipeline("t2i", backend="stable-diffusion", model=tiny_sd)
    run(pipe, seed=1, steps=7, width=64, height=32)
    run(pipe, seed=2, steps=2, width=32, height=48, num_images=2)
    run(pipe, seed=3, steps=1)
    assert same(run(pipe, seed=7, steps=4), fresh)
    run(pipe, seed=7, steps=9)
    assert same(run(pipe, seed=7, steps=4), fresh)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_sd, grad_before):
    pipe = pipeline("t2i", backend="stable-diffusion", model=tiny_sd)
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        result = run(pipe, num_images=2, seed=1)
        assert torch.is_grad_enabled() is grad_before
        assert not torch.is_inference_mode_enabled()
        assert all(not image.tensor.requires_grad for image in result.images)
        run(pipe, seed=1)
        assert torch.is_grad_enabled() is grad_before
    finally:
        torch.set_grad_enabled(previous)
