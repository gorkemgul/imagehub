import ast
import json
import re
import warnings
from pathlib import Path

import pytest

import imagehub
from imagehub import ImageHubLicenseWarning, pipeline
from imagehub.catalog import MODEL_CATALOG
from imagehub.registry import available_backends

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


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


def test_documented_pipeline_examples_name_registered_backends(tmp_path):
    paths = [
        ROOT / "README.md",
        *(path for path in (ROOT / "docs").rglob("*.md") if "design" not in path.parts),
    ]
    registered = set(available_backends("t2i")["t2i"])
    documented: set[str] = set()
    for path in paths:
        for block in PYTHON_BLOCK.findall(path.read_text(encoding="utf-8")):
            if "pipeline(" not in block:
                continue
            if path.name == "api.md" and block.lstrip().startswith("pipeline(task, *, backend"):
                continue
            tree = ast.parse(block, filename=str(path))
            for call in (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "pipeline"
            ):
                # The API page displays the function signature rather than a runnable example.
                if call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == "task":
                    continue
                backend = next((kw.value for kw in call.keywords if kw.arg == "backend"), None)
                assert isinstance(backend, ast.Constant) and isinstance(backend.value, str), path
                assert backend.value in registered, (path, backend.value)
                documented.add(backend.value)

    assert documented
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImageHubLicenseWarning)
        for backend in documented:
            created = pipeline("t2i", backend=backend, model=tmp_path / "missing")
            assert created.backend is not None


def test_caller_facing_exceptions_are_exported_from_top_level():
    names = {
        "ImageHubError",
        "MissingDependencyError",
        "BackendBusyError",
        "BackendContractError",
        "CheckpointError",
        "DownloadError",
        "ImageHubLicenseWarning",
        "ImagehubLicenseWarning",
    }
    assert names <= set(imagehub.__all__)
    assert all(hasattr(imagehub, name) for name in names)


def test_site_describes_per_model_reference_and_uses_dynamic_counts():
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    home = (ROOT / "docs/index.md").read_text(encoding="utf-8")
    catalog = (ROOT / "docs/catalog.md").read_text(encoding="utf-8")
    assert "recorded reference implementation" in config
    assert "bit-identical to the recorded reference implementation" in home
    assert "{{ ih.families }}" in catalog
    assert "{{ ih.models }}" in catalog
    assert "| Code license | Code commercial use |" in catalog


def test_every_kept_fixture_has_provenance():
    fixture_root = ROOT / "tests" / "fixtures"
    manifest = json.loads((fixture_root / "PROVENANCE.json").read_text(encoding="utf-8"))
    recorded = {ROOT / entry["path"] for entry in manifest}
    actual = {
        path
        for path in fixture_root.rglob("*")
        if path.is_file()
        and path.name not in {"PROVENANCE.json", "RESTRICTED.json"}
        and "_restricted" not in path.parts
    }
    assert recorded == actual
    assert all(entry["source"] and entry["revision"] and entry["license"] for entry in manifest)


def test_restricted_fixture_manifest_is_complete_and_pinned():
    fixture_root = ROOT / "tests" / "fixtures"
    manifest = json.loads((fixture_root / "RESTRICTED.json").read_text(encoding="utf-8"))
    removed = {entry["target"] for entry in manifest}
    assert len(removed) == len(manifest) == 298
    assert all(len(entry["revision"]) == 40 for entry in manifest)
    assert all(len(entry["sha256"]) == 64 and entry["size"] > 0 for entry in manifest)
    assert all(entry["repo"] and entry["upstream_path"] and entry["license"] for entry in manifest)


def test_official_port_license_records_are_complete():
    families = {
        "bagel",
        "emu3",
        "hart",
        "hidream-o1",
        "ideogram4",
        "infinity",
        "llamagen",
        "lumina-dimoo",
        "meissonic",
        "nextstep",
        "pixeldit",
        "show-o",
    }
    for family in families:
        directory = ROOT / "licenses" / family
        source = json.loads((directory / "SOURCE.json").read_text(encoding="utf-8"))
        assert (directory / "LICENSE").is_file()
        assert set(source) == {
            "upstream",
            "revision",
            "license",
            "commercial_use",
            "policy",
        }
        assert len(source["revision"]) == 40


def test_pixeldit_derivative_files_have_license_header():
    for relative in (
        "imagehub/models/pixeldit.py",
        "imagehub/backends/pixeldit.py",
        "tests/test_pixeldit.py",
        "tests/reference/test_ref_pixeldit.py",
    ):
        beginning = (ROOT / relative).read_text(encoding="utf-8")[:300]
        assert "Derivative of NVlabs PixelDiT under NSCLv1; non-commercial" in beginning


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
