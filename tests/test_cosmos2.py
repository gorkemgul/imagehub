import pytest
import torch
from helpers import write_tiny_cosmos2

from imagehub import pipeline
from imagehub.backends.cosmos2 import Cosmos2Backend
from imagehub.exceptions import CheckpointError
from imagehub.types import T2IRequest


@pytest.fixture(scope="module")
def tiny_cosmos2(tmp_path_factory):
    return write_tiny_cosmos2(tmp_path_factory.mktemp("tiny-cosmos2") / "snapshot")


def test_generates_image_with_metadata(tiny_cosmos2):
    result = pipeline("t2i", backend="cosmos2", model=tiny_cosmos2)(
        "a cat", seed=3, steps=2, guidance=1.0, width=48, height=32, max_sequence_length=16
    )
    assert len(result.images) == 1
    assert (result.images[0].width, result.images[0].height) == (48, 32)
    assert result.images[0].tensor.dtype == torch.uint8
    assert result.metadata == {
        "model_id": None,
        "revision": None,
        "seed": 3,
        "steps": 2,
        "guidance": 1.0,
        "width": 48,
        "height": 32,
        "negative_prompt": None,
        "scheduler": "flow_match_euler_karras",
        "device": "cpu",
        "dtype": "float32",
    }


def test_published_defaults_come_from_catalog(tmp_path):
    backend = Cosmos2Backend(tmp_path / "missing", model_id="cosmos-predict2-2b-t2i")
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert (options["steps"], options["guidance"]) == (35, 7.0)
    assert (options["width"], options["height"], options["max_sequence_length"]) == (
        1360,
        768,
        512,
    )
    assert options["latent_shape"] == (1, 16, 1, 96, 170)


def test_guidance_and_seed_are_repeatable_and_isolated(tiny_cosmos2):
    pipe = pipeline("t2i", backend="cosmos2", model=tiny_cosmos2)
    options = {"steps": 2, "width": 48, "height": 32, "max_sequence_length": 16}
    first = pipe("a cat", seed=11, guidance=1.0, **options)
    pipe("a dog", seed=2, guidance=2.0, negative_prompt="blurry", **options)
    repeated = pipe("a cat", seed=11, guidance=1.0, **options)
    different = pipe("a cat", seed=12, guidance=1.0, **options)
    assert torch.equal(first.images[0].tensor, repeated.images[0].tensor)
    assert not torch.equal(first.images[0].tensor, different.images[0].tensor)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_cosmos2, grad_before):
    pipe = pipeline("t2i", backend="cosmos2", model=tiny_cosmos2)
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        result = pipe(
            "a cat",
            seed=1,
            steps=1,
            guidance=1.0,
            width=32,
            height=32,
            max_sequence_length=16,
        )
        assert torch.is_grad_enabled() is grad_before
        assert not torch.is_inference_mode_enabled()
        assert not result.images[0].tensor.requires_grad
    finally:
        torch.set_grad_enabled(previous)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"latents": torch.zeros(1)}, ValueError, "latents"),
        ({"width": 1000}, ValueError, "multiples of 16"),
        ({"height": 520}, ValueError, "multiples of 16"),
        ({"width": 0}, ValueError, "width"),
        ({"steps": 0}, ValueError, "steps"),
        ({"steps": 1001}, ValueError, "at most 1000"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"guidance": -1.0}, ValueError, "guidance"),
        ({"guidance": True}, ValueError, "guidance"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"seed": -1}, ValueError, "seed"),
        ({"max_sequence_length": 0}, ValueError, "max_sequence_length"),
    ],
)
def test_invalid_options_fail_before_loading(tmp_path, options, error, match):
    backend = Cosmos2Backend(tmp_path / "missing")
    with pytest.raises(error, match=match):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded


def test_seed_and_latents_are_mutually_exclusive(tmp_path):
    backend = Cosmos2Backend(tmp_path / "missing")
    latents = torch.zeros(1, 16, 1, 4, 4)
    with pytest.raises(ValueError, match="either seed or latents"):
        backend.infer(T2IRequest("a cat"), seed=1, latents=latents, width=32, height=32)
    assert not backend.is_loaded


def test_incomplete_snapshot_names_download_command(tmp_path):
    (tmp_path / "tokenizer").mkdir()
    backend = Cosmos2Backend(tmp_path, model_id="cosmos-predict2-2b-t2i")
    with pytest.raises(CheckpointError, match="python -m imagehub.pretrained cosmos-predict2"):
        backend.load()


def test_wrong_pipeline_class_is_rejected(tiny_cosmos2, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(tiny_cosmos2, copy)
    (copy / "model_index.json").write_text('{"_class_name": "CosmosTextToWorldPipeline"}')
    with pytest.raises(CheckpointError, match="Unsupported Cosmos-Predict2 pipeline class"):
        Cosmos2Backend(copy).load()
