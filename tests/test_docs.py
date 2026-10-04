from pathlib import Path

from imagehub.catalog import MODEL_CATALOG

ROOT = Path(__file__).resolve().parents[1]


def test_every_model_is_documented():
    catalog = (ROOT / "docs/catalog.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for spec in MODEL_CATALOG.values():
        if spec.family != "stable-diffusion":
            continue
        assert f"`{spec.id}`" in catalog and f"`{spec.id}`" in readme
        assert spec.revision[:7] in catalog
