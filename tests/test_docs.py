from pathlib import Path

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
