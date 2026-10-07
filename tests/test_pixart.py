from pathlib import Path

import pytest
import torch

from imagehub.backends.pixart import PixArtBackend, classify_height_width_bin
from imagehub.catalog import model_info
from imagehub.types import T2IRequest


def test_catalog_defaults_and_backend_defaults():
    sigma = model_info("pixart-sigma-xl-1024")
    alpha = model_info("pixart-alpha-xl-1024")
    assert sigma.family == alpha.family == "pixart"
    assert sigma.config == {"defaults": {"steps": 20, "guidance": 4.5, "max_sequence_length": 300}}
    assert alpha.config == {"defaults": {"steps": 20, "guidance": 4.5, "max_sequence_length": 120}}
    sigma_options = PixArtBackend(Path("missing"), model_id=sigma.id).validate_options(
        T2IRequest("a cat"), {}
    )
    alpha_options = PixArtBackend(Path("missing"), model_id=alpha.id).validate_options(
        T2IRequest("a cat"), {}
    )
    assert sigma_options["max_sequence_length"] == 300
    assert alpha_options["max_sequence_length"] == 120
    assert sigma_options["steps"] == alpha_options["steps"] == 20
    assert sigma_options["guidance"] == alpha_options["guidance"] == 4.5
    assert sigma_options["width"] == sigma_options["height"] == 1024
    assert sigma_options["use_resolution_binning"] is True


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 1001, "use_resolution_binning": False}, ValueError, "multiple of 8"),
        ({"steps": 0}, ValueError, "steps"),
        ({"steps": 1001}, ValueError, "at most 1000"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"seed": -1}, ValueError, "seed"),
        ({"use_resolution_binning": 1}, TypeError, "use_resolution_binning"),
        ({"max_sequence_length": 0}, ValueError, "max_sequence_length"),
        ({"latents": torch.zeros(1, 4, 64, 64)}, ValueError, "shape"),
    ],
)
def test_invalid_options_fail_before_loading(options, error, match):
    backend = PixArtBackend(Path("missing"))
    with pytest.raises(error, match=match):
        backend.validate_options(T2IRequest("a cat"), options)
    assert not backend.is_loaded


def test_resolution_binning_matches_reference_shapes():
    assert classify_height_width_bin(1000, 800, 128) == (1088, 896)
    assert classify_height_width_bin(800, 1000, 128) == (896, 1088)
    backend = PixArtBackend(Path("missing"))
    options = backend.validate_options(T2IRequest("a cat"), {"width": 800, "height": 1000})
    assert (options["height"], options["width"]) == (1088, 896)
    assert (options["output_height"], options["output_width"]) == (1000, 800)


def test_alpha_size_conditions():
    backend = PixArtBackend(Path("missing"))
    options = backend.validate_options(
        T2IRequest("a cat"),
        {"width": 1280, "height": 768, "use_resolution_binning": False},
    )
    resolution, aspect_ratio = backend.size_conditions(options, batch_size=2, do_cfg=True)
    assert resolution.tolist() == [[768.0, 1280.0]] * 4
    torch.testing.assert_close(aspect_ratio, torch.full((4, 1), 0.6), rtol=0, atol=0)
