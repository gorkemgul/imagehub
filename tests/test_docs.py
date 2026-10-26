from pathlib import Path

import pytest

from imagehub.catalog import MODEL_CATALOG

ROOT = Path(__file__).resolve().parents[1]


def test_every_model_is_documented():
    catalog = (ROOT / "docs/catalog.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for spec in MODEL_CATALOG.values():
        assert f"`{spec.id}`" in catalog and f"`{spec.id}`" in readme
        assert spec.revision[:7] in catalog


def test_no_merge_conflict_markers():
    paths = [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            assert not line.startswith(("<<<<<<< ", ">>>>>>> ")), path


def test_site_describes_per_model_reference_and_uses_dynamic_counts():
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    home = (ROOT / "docs/index.md").read_text(encoding="utf-8")
    catalog = (ROOT / "docs/catalog.md").read_text(encoding="utf-8")
    assert "recorded reference implementation" in config
    assert "bit-identical to the recorded reference implementation" in home
    assert "{{ ih.families }}" in catalog
    assert "{{ ih.models }}" in catalog


def test_model_sections_cover_catalog_once():
    pytest.importorskip("markdown")
    from scripts.docs_hooks import _load, family_page, gallery_page, models_index, results_page

    data = _load(ROOT / "docs")
    documented = [model for family in data["families"].values() for model in family["models"]]
    assert sorted(documented) == sorted(MODEL_CATALOG)
    assert all(family_page(family, data) for family in data["families"].values())
    assert models_index(data)
    assert results_page(data)
    assert gallery_page(data)
