from pathlib import Path

import pytest
import torch

from imagehub.backends.cogview4 import CogView4Backend, calculate_shift, cogview4_sigmas
from imagehub.catalog import model_info
from imagehub.types import T2IRequest


def test_catalog_entry_and_defaults():
    spec = model_info("cogview4-6b")
    assert spec.family == "cogview4"
    assert spec.catalog_id == "cogview4"
    assert spec.repo_id == "zai-org/CogView4-6B"
    assert spec.revision == "63a52b7f6dace7033380cd6da14d0915eab3e6b5"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert spec.config == {"defaults": {"steps": 50, "guidance": 5.0}}
    assert len(spec.files) == 18


def test_backend_uses_catalog_defaults():
    backend = CogView4Backend(Path("missing"), model_id="cogview4-6b")
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert (options["steps"], options["guidance"], options["max_sequence_length"]) == (
        50,
        5.0,
        1024,
    )


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 1000}, ValueError, "multiple of 16"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"seed": -1}, ValueError, "seed"),
        ({"max_sequence_length": 0}, ValueError, "max_sequence_length"),
        (
            {"seed": 1, "latents": torch.zeros(1, 16, 128, 128)},
            ValueError,
            "either",
        ),
        ({"latents": torch.zeros(1, 16, 64, 64)}, ValueError, "shape"),
    ],
)
def test_invalid_options_fail_before_loading(options, error, match):
    backend = CogView4Backend(Path("missing"))
    with pytest.raises(error, match=match):
        backend.validate_options(T2IRequest("a cat"), options)
    assert not backend.is_loaded


def test_sigmas_and_mu_match_reference_formula():
    assert cogview4_sigmas(4, 1000) == [
        1.0,
        0.6669999957084656,
        0.33399999141693115,
        0.0010000000474974513,
    ]
    assert calculate_shift(256) == pytest.approx(1.0)
    assert calculate_shift(4096) == pytest.approx(3.25)
