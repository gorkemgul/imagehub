import json

import pytest
import torch
from helpers import write_tiny_kandinsky3

from imagehub import pipeline
from imagehub.backends.kandinsky3 import Kandinsky3Backend, latent_size
from imagehub.exceptions import CheckpointError
from imagehub.types import T2IRequest


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    return write_tiny_kandinsky3(tmp_path_factory.mktemp("kandinsky3"))


def test_latent_size_rounds_up_to_64_pixels():
    assert [latent_size(n) for n in (8, 64, 72, 1000, 1024)] == [8, 8, 16, 128, 128]


def test_generation_is_deterministic_and_metadata_complete(snapshot):
    with pipeline("t2i", backend="kandinsky3", model=snapshot) as pipe:
        first = pipe("a b", steps=2, guidance=3.0, width=64, height=64, seed=3)
        second = pipe("a b", steps=2, guidance=3.0, width=64, height=64, seed=3)
        other = pipe("a b", steps=2, guidance=3.0, width=64, height=64, seed=4)
    assert first.images[0].tensor.shape == (64, 64, 3)
    assert torch.equal(first.images[0].tensor, second.images[0].tensor)
    assert not torch.equal(first.images[0].tensor, other.images[0].tensor)
    assert first.metadata["seed"] == 3
    assert first.metadata["scheduler"] == "ddpm"
    assert first.metadata["dtype"] == "float32"


def test_size_is_rounded_up_to_64_pixels_like_the_reference(snapshot):
    with pipeline("t2i", backend="kandinsky3", model=snapshot) as pipe:
        result = pipe("a", steps=1, width=72, height=64, seed=0)
    assert result.images[0].tensor.shape == (64, 128, 3)
    assert (result.metadata["width"], result.metadata["height"]) == (128, 64)


def test_num_images_and_given_latents(snapshot):
    latents = torch.randn(2, 4, 8, 8)
    with pipeline("t2i", backend="kandinsky3", model=snapshot) as pipe:
        result = pipe("a", steps=2, width=64, height=64, num_images=2, latents=latents, seed=1)
        again = pipe("a", steps=2, width=64, height=64, num_images=2, latents=latents, seed=1)
    assert len(result.images) == 2
    assert torch.equal(result.images[1].tensor, again.images[1].tensor)


def test_catalog_defaults_apply_to_model_selector(snapshot):
    backend = Kandinsky3Backend(snapshot, model_id="kandinsky-3")
    assert backend.option_defaults["width"] == 1024
    assert backend.option_defaults["guidance"] == 3.0
    assert backend.option_defaults["steps"] == 25


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"steps": 0}, ValueError, "steps"),
        ({"steps": 1001}, ValueError, "at most 1000"),
        ({"width": 60}, ValueError, "multiple of 8"),
        ({"num_images": 0}, ValueError, "num_images"),
        ({"guidance": -1}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"timesteps": [1]}, TypeError, "Unknown Kandinsky 3 options"),
        ({"latents": torch.zeros(1, 4, 2, 2)}, ValueError, "latents"),
    ],
)
def test_options_are_validated_before_loading(snapshot, options, error, message):
    backend = Kandinsky3Backend(snapshot)
    options = {"width": 64, "height": 64} | options
    with pytest.raises(error, match=message):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded


def test_missing_components_name_the_download_command(tmp_path):
    backend = Kandinsky3Backend(tmp_path, model_id="kandinsky-3")
    with pytest.raises(CheckpointError, match="python -m imagehub.pretrained kandinsky-3"):
        backend.load()


def test_snapshot_must_use_ddpm(tmp_path):
    root = write_tiny_kandinsky3(tmp_path / "k3")
    path = root / "scheduler" / "scheduler_config.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | {"_class_name": "DDIMScheduler"}))
    with pytest.raises(ValueError, match="Unsupported Kandinsky 3 scheduler"):
        Kandinsky3Backend(root).load()


def test_autograd_mode_is_restored(snapshot):
    with pipeline("t2i", backend="kandinsky3", model=snapshot) as pipe:
        pipe("a", steps=1, width=64, height=64, seed=0)
    assert torch.is_grad_enabled()
