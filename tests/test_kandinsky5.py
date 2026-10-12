import json
from pathlib import Path

import pytest
import torch
from helpers import write_tiny_kandinsky5

from imagehub import pipeline
from imagehub.backends.kandinsky5 import Kandinsky5Backend
from imagehub.exceptions import CheckpointError
from imagehub.types import T2IRequest


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    return write_tiny_kandinsky5(tmp_path_factory.mktemp("kandinsky5"))


def test_generation_is_deterministic_and_metadata_complete(snapshot):
    with pipeline("t2i", backend="kandinsky5", model=snapshot) as pipe:
        first = pipe("a b", steps=2, guidance=3.5, width=64, height=64, seed=3)
        second = pipe("a b", steps=2, guidance=3.5, width=64, height=64, seed=3)
        other = pipe("a b", steps=2, guidance=3.5, width=64, height=64, seed=4)
    assert first.images[0].tensor.shape == (1024, 1024, 3)
    assert torch.equal(first.images[0].tensor, second.images[0].tensor)
    assert not torch.equal(first.images[0].tensor, other.images[0].tensor)
    assert first.metadata["seed"] == 3
    assert first.metadata["scheduler"] == "flow_match_euler"
    assert first.metadata["dtype"] == "float32"


def test_size_is_snapped_to_published_resolution(snapshot):
    with pipeline("t2i", backend="kandinsky5", model=snapshot) as pipe:
        result = pipe("a", steps=1, width=1000, height=700, seed=0)
    assert result.images[0].tensor.shape == (896, 1152, 3)
    assert (result.metadata["width"], result.metadata["height"]) == (1152, 896)


def test_num_images_and_given_latents(snapshot):
    latents = torch.randn(2, 1, 128, 128, 4)
    with pipeline("t2i", backend="kandinsky5", model=snapshot) as pipe:
        result = pipe("a", steps=2, width=64, height=64, num_images=2, latents=latents)
        again = pipe("a", steps=2, width=64, height=64, num_images=2, latents=latents)
    assert len(result.images) == 2
    assert torch.equal(result.images[1].tensor, again.images[1].tensor)


def test_catalog_defaults_apply_to_model_selector(snapshot):
    backend = Kandinsky5Backend(snapshot, model_id="kandinsky-5-t2i-lite")
    assert backend.option_defaults["width"] == 1024
    assert backend.option_defaults["guidance"] == 3.5
    assert backend.option_defaults["steps"] == 50
    assert backend.option_defaults["max_sequence_length"] == 512


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"steps": 0}, ValueError, "steps"),
        ({"steps": 1001}, ValueError, "at most 1000"),
        ({"num_images": 0}, ValueError, "num_images"),
        ({"guidance": -1}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"max_sequence_length": 1025}, ValueError, "max_sequence_length"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"timesteps": [1]}, TypeError, "Unknown Kandinsky 5 options"),
        ({"latents": torch.zeros(1, 1, 2, 2, 4)}, ValueError, "latents"),
    ],
)
def test_options_are_validated_before_loading(snapshot, options, error, message):
    backend = Kandinsky5Backend(snapshot)
    options = {"width": 64, "height": 64} | options
    with pytest.raises(error, match=message):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded


def test_missing_components_name_the_download_command(tmp_path):
    backend = Kandinsky5Backend(tmp_path, model_id="kandinsky-5-t2i-lite")
    with pytest.raises(CheckpointError, match="python -m imagehub.pretrained kandinsky-5-t2i-lite"):
        backend.load()


def test_snapshot_must_use_flow_match_euler(tmp_path):
    root = write_tiny_kandinsky5(tmp_path / "k5")
    path = root / "scheduler" / "scheduler_config.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | {"_class_name": "DDIMScheduler"}))
    with pytest.raises(ValueError, match="Unsupported Kandinsky 5 scheduler"):
        Kandinsky5Backend(root).load()


def test_autograd_mode_is_restored(snapshot):
    with pipeline("t2i", backend="kandinsky5", model=snapshot) as pipe:
        pipe("a", steps=1, width=64, height=64, seed=0)
    assert torch.is_grad_enabled()


def test_backend_directory_is_path(snapshot):
    backend = Kandinsky5Backend(Path(snapshot))
    assert backend.source == Path(snapshot)
