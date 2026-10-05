"""Checks catalog_id links against the sibling awesome-text-to-image-architectures checkout."""

import json
from pathlib import Path

import pytest

from imagehub.catalog import MODEL_CATALOG

CATALOG = (
    Path(__file__).resolve().parents[2] / "awesome-text-to-image-architectures/data/models.json"
)


def _norm(value: str) -> str:
    if "(" in value and value.endswith(")"):
        value = value.rsplit("(", 1)[1][:-1]
    return value.lower().replace("_", "-").replace(" ", "-").replace("open-rail", "openrail")


def test_catalog_ids_exist_with_consistent_weight_licenses():
    if not CATALOG.is_file():
        pytest.skip(f"Catalog checkout not found at {CATALOG}")
    entries = {m["id"]: m for m in json.loads(CATALOG.read_text(encoding="utf-8"))["models"]}
    for spec in MODEL_CATALOG.values():
        assert spec.catalog_id in entries, spec.id
        weights = entries[spec.catalog_id]["license"]["weights"]
        # A catalog entry may cover several releases: "license-a (v2); license-b (v2.5)".
        parts = [part.strip() for part in weights.split(";")]
        assert any(
            _norm(part) == _norm(spec.license)
            or part.lower().split(" (")[0] == spec.license.lower()
            for part in parts
        ), (spec.id, weights)
