import pytest
import torch
from helpers import write_tiny_auraflow

from imagehub import pipeline
from imagehub.backends.auraflow import AuraFlowBackend
from imagehub.exceptions import CheckpointError
from imagehub.types import T2IRequest


@pytest.fixture(scope="module")
def tiny_auraflow(tmp_path_factory):
    return write_tiny_auraflow(tmp_path_factory.mktemp("tiny-auraflow") / "snapshot")


def test_generates_images_with_metadata(tiny_auraflow):
    result = pipeline("t2i", backend="auraflow", model=tiny_auraflow)(
        "a cat", seed=3, steps=2, width=64, height=48, max_sequence_length=16
    )
    assert len(result.images) == 1
    image = result.images[0]
    assert (image.width, image.height) == (64, 48)
    assert image.tensor.dtype == torch.uint8
    metadata = result.metadata
    assert metadata["seed"] == 3
    assert metadata["steps"] == 2
    assert metadata["guidance"] == 3.5
    assert metadata["negative_prompt"] is None
    assert metadata["scheduler"] == "flow_match_euler"
    assert metadata["device"] == "cpu" and metadata["dtype"] == "float32"


def test_published_defaults_come_from_the_catalog(tmp_path):
    backend = AuraFlowBackend(tmp_path / "missing", model_id="auraflow-v0.3")
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert (options["steps"], options["guidance"]) == (50, 3.5)
    assert (options["width"], options["height"], options["max_sequence_length"]) == (
        1024,
        1024,
        256,
    )
    assert options["latent_shape"] == (1, 4, 128, 128)


def test_guidance_one_disables_the_negative_pass(tiny_auraflow):
    pipe = pipeline("t2i", backend="auraflow", model=tiny_auraflow)
    options = {"seed": 1, "steps": 2, "width": 64, "height": 64, "max_sequence_length": 16}
    plain = pipe("a cat", guidance=1.0, **options)
    guided = pipe("a cat", **options)
    assert plain.metadata["guidance"] == 1.0
    assert not torch.equal(plain.images[0].tensor, guided.images[0].tensor)


def test_seed_is_reproducible_and_requests_are_isolated(tiny_auraflow):
    pipe = pipeline("t2i", backend="auraflow", model=tiny_auraflow)
    options = {"steps": 2, "width": 64, "height": 64, "max_sequence_length": 16}
    a = pipe("a cat", seed=11, **options)
    pipe("a hi cat", seed=2, guidance=2.0, negative_prompt="blurry", **options)
    b = pipe("a cat", seed=11, **options)
    c = pipe("a cat", seed=12, **options)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert not torch.equal(a.images[0].tensor, c.images[0].tensor)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_auraflow, grad_before):
    pipe = pipeline("t2i", backend="auraflow", model=tiny_auraflow)
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        result = pipe("a cat", seed=1, steps=1, width=64, height=64, max_sequence_length=16)
        assert torch.is_grad_enabled() is grad_before
        assert not torch.is_inference_mode_enabled()
        assert not result.images[0].tensor.requires_grad
    finally:
        torch.set_grad_enabled(previous)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"latents": torch.zeros(1)}, TypeError, "latents"),
        ({"width": 1000}, ValueError, "multiple of 16"),
        ({"height": 520}, ValueError, "multiple of 16"),
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
    backend = AuraFlowBackend(tmp_path / "missing")
    with pytest.raises(error, match=match):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded


def test_incomplete_snapshot_names_the_download_command(tmp_path):
    (tmp_path / "tokenizer").mkdir()
    backend = AuraFlowBackend(tmp_path, model_id="auraflow-v0.3")
    with pytest.raises(CheckpointError, match="python -m imagehub.pretrained auraflow-v0.3"):
        backend.load()


def test_wrong_pipeline_class_is_rejected(tiny_auraflow, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(tiny_auraflow, copy)
    (copy / "model_index.json").write_text('{"_class_name": "StableDiffusion3Pipeline"}')
    with pytest.raises(CheckpointError, match="Unsupported AuraFlow pipeline class"):
        AuraFlowBackend(copy).load()


def test_size_beyond_the_positional_grid_is_rejected(tiny_auraflow):
    # The tiny transformer has an 8x8 grid of 2x2 patches of 8-pixel latents.
    backend = AuraFlowBackend(tiny_auraflow)
    with pytest.raises(ValueError, match="at most 128"):
        backend.infer(T2IRequest("a cat"), width=144, height=64)
    assert not backend.is_loaded
