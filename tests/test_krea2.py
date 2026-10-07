from pathlib import Path

import pytest
import torch

from imagehub.backends.krea2 import (
    Krea2Backend,
    calculate_shift,
    pack_latents,
    prepare_position_ids,
    unpack_latents,
)
from imagehub.catalog import model_info
from imagehub.types import T2IRequest


def test_catalog_defaults_and_backend_defaults(tmp_path: Path) -> None:
    spec = model_info("krea2-turbo")
    assert spec.family == "krea2"
    assert spec.catalog_id == "krea-2"
    assert spec.repo_id == "krea/Krea-2-Turbo"
    assert spec.revision == "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"
    assert spec.provenance == "author"
    assert spec.license == "Krea 2 Community License"
    assert spec.gated is True
    assert spec.config == {"defaults": {"steps": 8, "guidance": 0.0}}
    assert len(spec.files) == 14
    backend = Krea2Backend(tmp_path, model_id="krea2-turbo")
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert options["steps"] == 8
    assert options["guidance"] == 0.0
    assert options["max_sequence_length"] == 512


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 15}, ValueError, "multiple of 16"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1, "latents": torch.zeros(1, 16, 2, 2)}, ValueError, "either"),
        ({"latents": torch.zeros(1, 4, 64)}, ValueError, "shape"),
        ({"max_sequence_length": 0}, ValueError, "max_sequence_length"),
    ],
)
def test_invalid_options_fail_before_loading(
    tmp_path: Path, options: dict[str, object], error: type[Exception], match: str
) -> None:
    backend = Krea2Backend(tmp_path)
    with pytest.raises(error, match=match):
        backend.validate_options(T2IRequest("a cat"), options)
    assert not backend.is_loaded


def test_pack_and_unpack_latents_match_reference_layout() -> None:
    raw = torch.arange(1 * 3 * 4 * 6).reshape(1, 3, 4, 6)
    packed = pack_latents(raw, patch_size=2)
    assert packed.shape == (1, 6, 12)
    assert torch.equal(unpack_latents(packed, 4, 6, patch_size=2), raw.unsqueeze(2))


def test_position_ids_match_text_and_image_coordinates() -> None:
    actual = prepare_position_ids(2, 2, 3, torch.device("cpu"))
    expected = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(actual, expected)


def test_calculate_shift_matches_reference_formula() -> None:
    assert calculate_shift(256) == pytest.approx(0.5)
    assert calculate_shift(4096) == pytest.approx(1.15)
    assert calculate_shift(6400, max_seq_len=6400) == pytest.approx(1.15)
    assert calculate_shift(1024, max_seq_len=4096) == pytest.approx(0.63)
