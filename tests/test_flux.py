from pathlib import Path

import pytest
import torch

from imagehub.backends.flux import (
    FluxBackend,
    _pack_latents,
    _prepare_latent_image_ids,
    _sigmas,
    _unpack_latents,
    calculate_shift,
)
from imagehub.catalog import model_info
from imagehub.types import T2IRequest


def test_catalog_entries_and_defaults():
    schnell = model_info("flux1-schnell")
    dev = model_info("flux1-dev")
    assert schnell.family == dev.family == "flux"
    assert schnell.catalog_id == dev.catalog_id == "flux-1"
    assert schnell.gated and dev.gated
    assert schnell.config == {"defaults": {"steps": 4, "guidance": 0.0, "max_sequence_length": 256}}
    assert dev.config == {"defaults": {"steps": 50, "guidance": 3.5, "max_sequence_length": 512}}
    assert len(schnell.files) == len(dev.files) == 23


def test_krea_dev_catalog_entry_and_defaults():
    dev = model_info("flux1-dev")
    krea = model_info("flux1-krea-dev")
    assert krea.family == "flux"
    assert krea.catalog_id == "flux-1"
    assert krea.repo_id == "black-forest-labs/FLUX.1-Krea-dev"
    assert krea.revision == "8162a9c7b05a641be098422bf2fcf335615c2f28"
    assert krea.provenance == "author"
    assert krea.gated
    assert krea.license == dev.license == "FLUX.1-dev Non-Commercial License"
    assert krea.image_input is None
    assert krea.files == dev.files
    assert krea.config == {"defaults": {"steps": 28, "guidance": 4.5, "max_sequence_length": 512}}
    options = FluxBackend(Path("missing"), model_id="flux1-krea-dev").validate_options(
        T2IRequest("a cat"), {}
    )
    assert (options["steps"], options["guidance"], options["max_sequence_length"]) == (28, 4.5, 512)


def test_backend_uses_catalog_defaults():
    schnell = FluxBackend(Path("missing"), model_id="flux1-schnell")
    dev = FluxBackend(Path("missing"), model_id="flux1-dev")
    request = T2IRequest("a cat")
    schnell_options = schnell.validate_options(request, {})
    dev_options = dev.validate_options(request, {})
    assert (
        schnell_options["steps"],
        schnell_options["guidance"],
        schnell_options["max_sequence_length"],
    ) == (4, 0.0, 256)
    assert (
        dev_options["steps"],
        dev_options["guidance"],
        dev_options["max_sequence_length"],
    ) == (50, 3.5, 512)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 1000}, ValueError, "multiple of 16"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"true_cfg_scale": float("inf")}, ValueError, "true_cfg_scale"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"seed": -1}, ValueError, "seed"),
        ({"max_sequence_length": 513}, ValueError, "max_sequence_length"),
        (
            {"seed": 1, "latents": torch.zeros(1, 16, 128, 128)},
            ValueError,
            "either",
        ),
        ({"latents": torch.zeros(1, 16, 64, 64)}, ValueError, "shape"),
    ],
)
def test_invalid_options_fail_before_loading(options, error, match):
    backend = FluxBackend(Path("missing"))
    with pytest.raises(error, match=match):
        backend.validate_options(T2IRequest("a cat"), options)
    assert not backend.is_loaded


def test_pack_and_unpack_latents():
    latents = torch.arange(1 * 2 * 4 * 6).reshape(1, 2, 4, 6)
    packed = _pack_latents(latents, 1, 2, 4, 6)
    assert packed.shape == (1, 6, 8)
    assert packed[0, 0].tolist() == [0, 1, 6, 7, 24, 25, 30, 31]
    assert torch.equal(_unpack_latents(packed, 32, 48, 8), latents)


def test_latent_image_ids():
    ids = _prepare_latent_image_ids(2, 2, 3, torch.device("cpu"), torch.float32)
    assert ids.shape == (6, 3)
    assert ids.tolist() == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 2.0],
    ]


def test_sigmas_and_mu_match_reference_formula():
    assert _sigmas(4) == [1.0, 0.75, 0.5, 0.25]
    assert calculate_shift(256) == pytest.approx(0.5)
    assert calculate_shift(4096) == pytest.approx(1.15)
