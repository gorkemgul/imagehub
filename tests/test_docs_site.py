import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mkdocs")
pytest.importorskip("material")
pytest.importorskip("mkdocs_static_i18n")

ROOT = Path(__file__).resolve().parents[1]


def test_docs_site_builds_strictly(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--quiet",
            "-d",
            str(tmp_path / "site"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    from imagehub.catalog import MODEL_CATALOG

    index = (tmp_path / "site/index.html").read_text(encoding="utf-8")
    assert f"<strong>{len(MODEL_CATALOG)}</strong>" in index
    assert (tmp_path / "site/tr/quickstart/index.html").is_file()
    assert (tmp_path / "site/parity/gallery/index.html").is_file()
