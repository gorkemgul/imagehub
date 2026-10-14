"""Tests for the native T5Gemma encoder."""

import json
from pathlib import Path

import torch

from imagehub.models.t5gemma import T5GemmaEncoder

FIXTURES = Path(__file__).parent / "fixtures" / "prx_configs"


def test_published_t5gemma_header_layout() -> None:
    config = json.loads((FIXTURES / "prx-512-t2i-sft/text_encoder/config.json").read_text())
    with torch.device("meta"):
        model = T5GemmaEncoder(config)
    expected: dict[str, list[int]] = {}
    for path in sorted((FIXTURES / "prx-512-t2i-sft_headers").glob("text_encoder_*.json")):
        expected.update(json.loads(path.read_text()))
    actual = model.state_dict()
    assert set(actual) == set(expected)
    assert {name: list(tensor.shape) for name, tensor in actual.items()} == {
        name: value for name, value in expected.items()
    }
