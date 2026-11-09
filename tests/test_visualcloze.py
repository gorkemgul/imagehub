from pathlib import Path

import pytest
import torch
from helpers import write_tiny_flux1

from imagehub import pipeline
from imagehub.backends.visualcloze import VisualClozeBackend, preprocess_image
from imagehub.catalog import model_info
from imagehub.types import T2IRequest

PILImage = pytest.importorskip("PIL.Image")


def make_image(width: int, height: int, seed: int = 0):
    values = torch.randint(
        0,
        256,
        (height * width * 3,),
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.uint8,
    )
    return PILImage.frombytes("RGB", (width, height), bytes(values.tolist()))


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    return write_tiny_flux1(
        tmp_path_factory.mktemp("visualcloze") / "snapshot",
        in_channels=384,
        out_channels=64,
        latent_channels=16,
        scheduler_config={"base_image_seq_len": 256, "max_image_seq_len": 4096},
    )


def request():
    return [[make_image(32, 32), make_image(32, 32, 1)], [make_image(32, 32, 2), None]]


def run(snapshot: Path, **options):
    return pipeline("t2i", backend="visualcloze-384", model=snapshot)(
        "Each row shows a process.",
        **({"steps": 1, "max_sequence_length": 8, "upsampling_strength": 0} | options),
    )


def test_catalog_entry_and_defaults():
    spec = model_info("visualcloze-384")
    assert spec.family == "visualcloze" and spec.image_input == "required"
    assert spec.revision == "59c469d2772d927ffe55f3543c4d3bd556fd46a4"
    assert spec.config["resolution"] == 384 and len(spec.files) == 23


def test_shapes_for_one_and_multiple_targets(snapshot):
    one = run(snapshot, image=request(), seed=0)
    assert len(one.images) == 1 and one.images[0].tensor.shape == (384, 384, 3)
    several = run(
        snapshot,
        image=[
            [make_image(32, 32), make_image(32, 32, 1), make_image(32, 32, 2)],
            [make_image(32, 32, 3), None, None],
        ],
        seed=0,
    )
    assert len(several.images) == 2 and all(
        image.tensor.shape == (384, 384, 3) for image in several.images
    )


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({}, ValueError),
        ({"image": [[make_image(32, 32), None], [make_image(32, 32)]]}, ValueError),
        ({"image": [[None, make_image(32, 32)], [make_image(32, 32), None]]}, ValueError),
        ({"image": request(), "height": 32}, TypeError),
        ({"image": request(), "upsampling_height": 32}, ValueError),
        ({"image": request(), "content_prompt": 1}, TypeError),
    ],
)
def test_options_validate_before_loading(options, error):
    backend = VisualClozeBackend(Path("missing"), model_id="visualcloze-384")
    with pytest.raises(error):
        backend.validate_options(T2IRequest("task"), options)
    assert not backend.is_loaded


def test_default_upsampling_size_rule(snapshot):
    image = make_image(2000, 1000)
    result = run(
        snapshot,
        image=[[make_image(32, 32), make_image(32, 32, 1)], [image, None]],
        seed=0,
        upsampling_strength=1,
    )
    assert (result.metadata["upsampling_height"], result.metadata["upsampling_width"]) == (
        720,
        1440,
    )


def test_upsampling_stage(snapshot):
    result = run(
        snapshot,
        image=request(),
        seed=0,
        upsampling_strength=1,
        upsampling_height=48,
        upsampling_width=32,
    )
    assert result.images[0].tensor.shape == (48, 32, 3)


def test_all_target_last_row_uses_1024_default_upsampling_size(snapshot):
    result = run(
        snapshot,
        image=[[make_image(32, 32), make_image(32, 32, 1)], [None, None]],
        seed=0,
        upsampling_strength=1,
    )
    assert (result.metadata["upsampling_height"], result.metadata["upsampling_width"]) == (
        1024,
        1024,
    )
    assert len(result.images) == 2 and all(
        image.tensor.shape == (1024, 1024, 3) for image in result.images
    )


def test_no_upsampling_leaves_size_metadata_empty(snapshot):
    result = run(snapshot, image=request(), seed=0, upsampling_strength=0)
    assert result.metadata["upsampling_height"] is None
    assert result.metadata["upsampling_width"] is None


def test_repeatability_request_isolation_and_autograd(snapshot):
    first = run(snapshot, image=request(), seed=1)
    with torch.enable_grad():
        second = run(snapshot, image=request(), seed=1)
        assert torch.is_grad_enabled()
    assert torch.equal(first.images[0].tensor, second.images[0].tensor)


def test_multiple_non_square_targets_keep_their_aspect():
    grid = [
        [make_image(80, 40), make_image(80, 40, 1), make_image(80, 40, 2)],
        [make_image(80, 40, 3), None, None],
    ]
    processed, sizes, targets = preprocess_image(grid, resolution=384, vae_scale=8)
    assert sizes[-1] == [[256, 528], [256, 528], [256, 528]] and targets == [0, 1, 1]
    assert processed[-1][0].shape[-2:] == (256, 528)
