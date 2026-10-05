import pytest
import torch
from helpers import write_tiny_z_image

from imagehub import pipeline
from imagehub.backends.z_image import ZImageBackend
from imagehub.catalog import model_info
from imagehub.checkpoints import SNAPSHOT_FILE
from imagehub.exceptions import CheckpointError


@pytest.fixture(scope="module")
def tiny_z_image(tmp_path_factory):
    return write_tiny_z_image(tmp_path_factory.mktemp("tiny-z-image"))


def test_generates_images_with_metadata(tiny_z_image):
    result = pipeline("t2i", backend="z-image", model=tiny_z_image)(
        "a cat", seed=3, steps=2, width=32, height=32, num_images=2
    )
    assert len(result.images) == 2
    assert (result.images[0].width, result.images[0].height) == (32, 32)
    assert result.metadata["seed"] == 3 and result.metadata["steps"] == 2
    assert result.metadata["scheduler"] == "flow_match_euler"


def test_seed_and_latents_are_reproducible(tiny_z_image):
    pipe = pipeline("t2i", backend="z-image-turbo", model=tiny_z_image)
    a = pipe("a cat", seed=11, steps=2, width=32, height=32)
    b = pipe("a cat", seed=11, steps=2, width=32, height=32)
    latents = torch.randn(1, 16, 4, 4, generator=torch.Generator().manual_seed(11))
    c = pipe("a cat", latents=latents, steps=2, width=32, height=32)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert torch.equal(a.images[0].tensor, c.images[0].tensor)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 30}, ValueError, "multiple of 16"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1, "latents": torch.zeros(1, 16, 4, 4)}, ValueError, "either"),
        ({"latents": torch.zeros(1, 16, 8, 8)}, ValueError, "shape"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"cfg_normalization": 1}, TypeError, "cfg_normalization"),
        ({"cfg_truncation": 1.1}, ValueError, "cfg_truncation"),
        ({"max_sequence_length": 0}, ValueError, "max_sequence_length"),
    ],
)
def test_invalid_options_fail_before_loading(tiny_z_image, options, error, match):
    backend = ZImageBackend(tiny_z_image)
    with pytest.raises(error, match=match):
        pipeline("t2i", backend=backend)("a cat", **options)
    assert not backend.is_loaded


def test_catalog_entry_and_snapshot_hint(tmp_path):
    spec = model_info("z-image-turbo")
    assert spec.family == "z-image" and spec.catalog_id == "z-image"
    assert len(spec.files) == 19
    pipe = pipeline("t2i", backend="z-image-turbo", model=tmp_path / "nowhere")
    with pytest.raises(CheckpointError, match=r"imagehub\.pretrained z-image-turbo "):
        pipe("a cat")


def test_snapshot_metadata(tiny_z_image):
    (tiny_z_image / SNAPSHOT_FILE).write_text(
        '{"model_id": "z-image-turbo", "revision": "f332072aa78be7aecdf3ee76d5c247082da564a6"}'
    )
    result = pipeline("t2i", backend="z-image-turbo", model=tiny_z_image)(
        "a cat", seed=1, steps=1, width=32, height=32
    )
    assert result.metadata["model_id"] == "z-image-turbo"
