import warnings

import pytest
import torch
from helpers import write_tiny_llamagen

from imagehub import pipeline
from imagehub.backends.llamagen import (
    LlamaGenBackend,
    LlamaGenComponents,
    encode_prompts,
    find_gpt_checkpoint,
    generate,
    to_uint8,
)
from imagehub.catalog import model_info
from imagehub.exceptions import CheckpointError
from imagehub.types import T2IRequest

PROMPT = "A cat on a mat"


@pytest.fixture(scope="module")
def tiny_llamagen(tmp_path_factory):
    return write_tiny_llamagen(tmp_path_factory.mktemp("tiny-llamagen") / "snapshot")


def test_generates_images_with_metadata(tiny_llamagen):
    result = pipeline("t2i", backend="llamagen", model=tiny_llamagen)(PROMPT, seed=3)
    assert len(result.images) == 1
    image = result.images[0]
    assert (image.width, image.height) == (64, 64)
    assert image.tensor.dtype == torch.uint8
    metadata = result.metadata
    assert metadata["seed"] == 3
    assert metadata["cfg_scale"] == 7.5
    assert metadata["temperature"] == 1.0
    assert metadata["top_k"] == 1000
    assert metadata["top_p"] == 1.0
    assert metadata["cfg_interval"] == -1
    assert metadata["device"] == "cpu" and metadata["dtype"] == "float32"


def test_same_seed_same_image_and_other_seed_differs(tiny_llamagen):
    pipe = pipeline("t2i", backend="llamagen", model=tiny_llamagen)
    first = pipe(PROMPT, seed=1).images[0].tensor
    assert torch.equal(first, pipe(PROMPT, seed=1).images[0].tensor)
    assert not torch.equal(first, pipe(PROMPT, seed=2).images[0].tensor)


def test_model_id_selects_catalog_defaults_and_size(tiny_llamagen, tmp_path):
    backend = LlamaGenBackend(tiny_llamagen, model_id="llamagen-t2i-xl-256")
    assert backend.option_defaults["width"] == 256
    with pytest.raises(ValueError, match="width must be 256"):
        backend.validate_options(T2IRequest("x"), {"width": 64})


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"colour": 1}, TypeError, "Unknown LlamaGen options"),
        ({"cfg_scale": -1}, ValueError, "cfg_scale"),
        ({"cfg_scale": float("nan")}, ValueError, "cfg_scale"),
        ({"cfg_scale": True}, ValueError, "cfg_scale"),
        ({"temperature": 0}, ValueError, "temperature"),
        ({"top_k": -1}, ValueError, "top_k"),
        ({"top_k": 1.5}, ValueError, "top_k"),
        ({"top_p": 0}, ValueError, "top_p"),
        ({"top_p": 1.5}, ValueError, "top_p"),
        ({"cfg_interval": -2}, ValueError, "cfg_interval"),
        ({"seed": -1}, ValueError, "seed"),
        ({"width": 32}, ValueError, "width must be 64"),
        ({"height": 0}, ValueError, "height"),
    ],
)
def test_options_are_validated_before_loading(tiny_llamagen, options, error, match):
    backend = LlamaGenBackend(tiny_llamagen)
    with pytest.raises(error, match=match):
        backend.validate_options(T2IRequest("x"), options)
    assert not backend.is_loaded


def test_options_change_the_image(tiny_llamagen):
    pipe = pipeline("t2i", backend="llamagen", model=tiny_llamagen)
    base = pipe(PROMPT, seed=5).images[0].tensor
    for override in (
        {"cfg_scale": 1.0},
        {"cfg_scale": 3.0},
        {"temperature": 0.3},
        {"top_k": 3},
        {"top_p": 0.5},
    ):
        assert not torch.equal(base, pipe(PROMPT, seed=5, **override).images[0].tensor), override


def test_prompt_changes_the_image_and_is_lower_cased(tiny_llamagen):
    pipe = pipeline("t2i", backend="llamagen", model=tiny_llamagen)
    assert torch.equal(
        pipe("a cat on a mat", seed=4).images[0].tensor,
        pipe("  A CAT on a MAT ", seed=4).images[0].tensor,
    )
    assert not torch.equal(
        pipe("a cat", seed=4).images[0].tensor,
        pipe("a cat on a mat with a hat", seed=4).images[0].tensor,
    )


def test_cfg_interval_stops_guidance(tiny_llamagen):
    components = LlamaGenComponents.load(
        tiny_llamagen, device=torch.device("cpu"), dtype=torch.float32
    )
    cond, masks = encode_prompts(components, [PROMPT], torch.device("cpu"))

    def tokens(interval):
        return generate(
            components.gpt,
            cond,
            16,
            masks,
            cfg_scale=4.0,
            cfg_interval=interval,
            generator=torch.Generator().manual_seed(0),
        )

    assert torch.equal(tokens(-1), tokens(100))
    # Guidance covers the first token (prefill) and the decode steps up to the interval.
    assert torch.equal(tokens(2)[:, :4], tokens(-1)[:, :4])
    assert not torch.equal(tokens(2), tokens(-1))


def test_captions_are_left_padded_and_zeroed(tiny_llamagen):
    components = LlamaGenComponents.load(
        tiny_llamagen, device=torch.device("cpu"), dtype=torch.float32
    )
    embeddings, masks = encode_prompts(components, ["a b", "a"], torch.device("cpu"))
    assert embeddings.shape == (2, 120, 32) and masks.shape == (2, 120)
    lengths = [len(components.tokenizer.encode(text)) for text in ("a b", "a")]
    assert lengths[0] > lengths[1] > 1
    for row, length in enumerate(lengths):
        assert masks[row].tolist() == [0] * (120 - length) + [1] * length
        assert not embeddings[row, : 120 - length].any()
        assert embeddings[row, 120 - length :].abs().sum() > 0


def test_to_uint8_matches_the_save_image_normalisation():
    values = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]).reshape(1, 1, 1, 7)
    pixels = to_uint8(values.expand(1, 3, 1, 7))
    assert pixels.shape == (1, 1, 7, 3) and pixels.dtype == torch.uint8
    assert pixels[0, 0, :, 0].tolist() == [0, 0, 64, 128, 191, 255, 255]


def test_loading_reports_a_missing_or_ambiguous_snapshot(tiny_llamagen, tmp_path):
    with pytest.raises(CheckpointError, match="exactly one LlamaGen GPT checkpoint"):
        find_gpt_checkpoint(tmp_path)
    with pytest.raises(CheckpointError, match="not a LlamaGen snapshot"):
        LlamaGenComponents.load(
            tiny_llamagen,
            device=torch.device("cpu"),
            dtype=torch.float32,
            model_id="llamagen-t2i-xl-512",
        )
    assert find_gpt_checkpoint(tiny_llamagen)[1] == 64


def test_snapshot_of_another_model_warns(tiny_llamagen, tmp_path):
    import json
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(tiny_llamagen, copy)
    (copy / "imagehub_snapshot.json").write_text(json.dumps({"model_id": "other"}))
    backend = LlamaGenBackend(copy, model_id="llamagen-t2i-xl-256")
    with pytest.raises(CheckpointError):
        backend.load()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert LlamaGenBackend(copy).load().is_loaded


def test_catalog_defaults_are_the_reference_defaults():
    for model_id, size in (("llamagen-t2i-xl-256", 256), ("llamagen-t2i-xl-512", 512)):
        defaults = model_info(model_id).config["defaults"]
        assert defaults == {
            "cfg_scale": 7.5,
            "temperature": 1.0,
            "top_k": 1000,
            "top_p": 1.0,
            "cfg_interval": -1,
            "width": size,
            "height": size,
        }
