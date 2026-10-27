"""Checks catalog_id links against the sibling awesome-text-to-image-architectures checkout."""

import json
from pathlib import Path

import pytest

from imagehub.catalog import MODEL_CATALOG

CATALOG = (
    Path(__file__).resolve().parents[2] / "awesome-text-to-image-architectures/data/models.json"
)


def test_catalog_ids_exist_with_consistent_weight_licenses():
    if not CATALOG.is_file():
        pytest.skip(f"Catalog checkout not found at {CATALOG}")
    entries = {m["id"]: m for m in json.loads(CATALOG.read_text(encoding="utf-8"))["models"]}
    # Qwen-Image and Qwen-Image 2 are distinct architectures with distinct weight licenses.
    assert MODEL_CATALOG["qwen-image"].catalog_id == "qwen-image"
    assert MODEL_CATALOG["qwen-image-2.1"].catalog_id == "qwen-image-2"
    assert entries["qwen-image"]["license"]["weights"] == "Apache-2.0"
    assert "Qwen Research" in entries["qwen-image-2"]["license"]["weights"]
