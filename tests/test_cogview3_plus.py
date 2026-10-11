import json
import shutil
from pathlib import Path

import pytest
import torch
from helpers import write_tiny_cogview3_plus

from imagehub import pipeline
from imagehub.backends.cogview3_plus import CogView3PlusBackend
from imagehub.catalog import model_info
from imagehub.exceptions import CheckpointError
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.t5 import T5EncoderModel
from imagehub.types import T2IRequest

CONFIGS = Path(__file__).parent / "fixtures" / "cogview3_plus_configs"


@pytest.fixture(scope="module")
def tiny_cogview3_plus(tmp_path_factory):
    return write_tiny_cogview3_plus(tmp_path_factory.mktemp("tiny-cogview3-plus") / "snapshot")


def _shapes(model_class, name):
    config = json.loads((CONFIGS / f"{name}_config.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = model_class(config)
    return {key: list(value.shape) for key, value in model.state_dict().items()}


def test_text_encoder_and_vae_match_published_headers():
    for model_class, name in ((T5EncoderModel, "text_encoder"), (AutoencoderKL, "vae")):
        expected = json.loads((CONFIGS / f"{name}_tensors.json").read_text(encoding="utf-8"))
        assert _shapes(model_class, name) == expected


def test_published_indexes_list_exactly_the_catalog_weight_files():
    spec = model_info("cogview3-plus-3b")
    for component in ("text_encoder", "transformer"):
        index = next(CONFIGS.glob(f"{component}*.safetensors.index.json"))
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        listed = {f"{component}/{name}" for name in weight_map.values()}
        assert listed <= set(spec.files)
    assert len([name for name in spec.files if name.endswith(".safetensors")]) == 7


def test_generates_images_with_metadata(tiny_cogview3_plus):
    result = pipeline("t2i", backend="cogview3", model=tiny_cogview3_plus)(
        "a cat", seed=3, steps=2, width=64, height=48, max_sequence_length=16
    )
    assert len(result.images) == 1
    image = result.images[0]
    assert (image.width, image.height) == (64, 48)
    assert image.tensor.dtype == torch.uint8
    metadata = result.metadata
    assert metadata["seed"] == 3
    assert metadata["steps"] == 2
    assert metadata["guidance"] == 5.0
    assert metadata["negative_prompt"] is None
    assert metadata["scheduler"] == "ddim_cogvideox"
    assert metadata["device"] == "cpu" and metadata["dtype"] == "float32"


def test_published_defaults_come_from_the_catalog(tmp_path):
    backend = CogView3PlusBackend(tmp_path / "missing", model_id="cogview3-plus-3b")
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert (options["steps"], options["guidance"], options["max_sequence_length"]) == (50, 5.0, 224)
    assert (options["width"], options["height"]) == (1024, 1024)
    assert options["latent_shape"] == (1, 16, 128, 128)
    assert options["original_size"] is None and options["crops_coords_top_left"] == (0, 0)


def test_guidance_one_disables_the_negative_pass(tiny_cogview3_plus):
    pipe = pipeline("t2i", backend="cogview3", model=tiny_cogview3_plus)
    options = {"seed": 1, "steps": 2, "width": 64, "height": 64, "max_sequence_length": 16}
    plain = pipe("a cat", guidance=1.0, **options)
    guided = pipe("a cat", **options)
    assert plain.metadata["guidance"] == 1.0
    assert not torch.equal(plain.images[0].tensor, guided.images[0].tensor)


def test_negative_prompt_and_size_conditions_change_the_image(tiny_cogview3_plus):
    pipe = pipeline("t2i", backend="cogview3", model=tiny_cogview3_plus)
    options = {"seed": 1, "steps": 2, "width": 64, "height": 64, "max_sequence_length": 16}
    base = pipe("a cat", **options).images[0].tensor
    assert not torch.equal(base, pipe("a cat", negative_prompt="a hi", **options).images[0].tensor)
    assert not torch.equal(base, pipe("a cat", original_size=(128, 96), **options).images[0].tensor)
    assert not torch.equal(
        base, pipe("a cat", crops_coords_top_left=(16, 0), **options).images[0].tensor
    )


def test_seed_is_reproducible_and_requests_are_isolated(tiny_cogview3_plus):
    pipe = pipeline("t2i", backend="cogview3", model=tiny_cogview3_plus)
    options = {"steps": 2, "width": 64, "height": 64, "max_sequence_length": 16}
    a = pipe("a cat", seed=11, **options)
    pipe("a hi cat", seed=2, guidance=2.0, negative_prompt="a hi", **options)
    b = pipe("a cat", seed=11, **options)
    c = pipe("a cat", seed=12, **options)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert not torch.equal(a.images[0].tensor, c.images[0].tensor)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_cogview3_plus, grad_before):
    pipe = pipeline("t2i", backend="cogview3", model=tiny_cogview3_plus)
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
        ({"latents": torch.zeros(1)}, ValueError, "latents"),
        ({"latents": torch.zeros(1, 16, 64, 64)}, ValueError, "shape"),
        ({"seed": 1, "latents": torch.zeros(1, 16, 128, 128)}, ValueError, "either"),
        ({"width": 1000}, ValueError, "multiple of 16"),
        ({"height": 520}, ValueError, "multiple of 16"),
        ({"width": 2064}, ValueError, "at most 2048"),
        ({"width": 0}, ValueError, "width"),
        ({"steps": 0}, ValueError, "steps"),
        ({"steps": 1001}, ValueError, "at most 1000"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"guidance": -1.0}, ValueError, "guidance"),
        ({"guidance": True}, ValueError, "guidance"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"seed": -1}, ValueError, "seed"),
        ({"max_sequence_length": 0}, ValueError, "max_sequence_length"),
        ({"original_size": (0, 8)}, ValueError, "original_size"),
        ({"original_size": 8}, ValueError, "original_size"),
        ({"crops_coords_top_left": (-1, 0)}, ValueError, "crops_coords_top_left"),
        ({"crops_coords_top_left": (0,)}, ValueError, "crops_coords_top_left"),
    ],
)
def test_invalid_options_fail_before_loading(tmp_path, options, error, match):
    backend = CogView3PlusBackend(tmp_path / "missing")
    with pytest.raises(error, match=match):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded


def test_incomplete_snapshot_names_the_download_command(tmp_path):
    (tmp_path / "tokenizer").mkdir()
    backend = CogView3PlusBackend(tmp_path, model_id="cogview3-plus-3b")
    with pytest.raises(CheckpointError, match="python -m imagehub.pretrained cogview3-plus-3b"):
        backend.load()


def test_wrong_pipeline_class_is_rejected(tiny_cogview3_plus, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(tiny_cogview3_plus, copy)
    (copy / "model_index.json").write_text('{"_class_name": "CogView4Pipeline"}')
    with pytest.raises(CheckpointError, match="Unsupported CogView3-Plus pipeline class"):
        CogView3PlusBackend(copy).load()


def test_size_beyond_the_positional_grid_is_rejected(tiny_cogview3_plus):
    # The tiny transformer has an 8x8 grid of 2x2 patches of 8-pixel latents: 128 pixels.
    backend = CogView3PlusBackend(tiny_cogview3_plus)
    with pytest.raises(ValueError, match="at most 128"):
        backend.infer(T2IRequest("a cat"), width=144, height=64)
    assert not backend.is_loaded
