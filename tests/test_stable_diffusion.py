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


def test_device_and_dtype_validation(tiny_sd):
    with pytest.raises(ValueError, match="Available: cpu"):
        resolve_device("quantum")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="not available"):
            StableDiffusionBackend(tiny_sd, device="cuda")
    with pytest.raises(ValueError, match="dtype"):
        StableDiffusionBackend(tiny_sd, dtype=torch.int8)
