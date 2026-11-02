---
name: prepare-release-evidence
description: Prepare an imagehub release: version bump, validation records, package contents, minimum-version and installed-wheel tests, CUDA smoke test and PyPI publishing. Use for packaging, versioning, CI, validation records or any release step.
---

# Prepare Release Evidence

Publishing and paid cloud resources need the maintainer's explicit approval at the moment they
happen (see `.ai/AGENTS.md`).

## 1. Validation is current

1. Run the full reference suite on every snapshot listed in the catalog and rebuild
   `docs/validation.json` with `scripts/build_validation.py`.
2. Every catalog model is `full` (or its status is stated honestly in `docs/catalog.md`).
3. `docs/validation.md` matches `docs/validation.json`: environment, measurements, device smoke
   results, limits and test counts.

## 2. Package contents

```bash
rm -rf dist && .venv/bin/python -m build
tar tzf dist/imagehub-*.tar.gz
unzip -l dist/imagehub-*-py3-none-any.whl
```

- The wheel contains only `imagehub/` and its dist-info.
- The sdist contains only the paths in `[tool.hatch.build.targets.sdist]`; never `.ai/`
  working notes, `docs/research/`, `checkpoints/`, `artifacts/` or `scripts/`.
- License metadata lists `LICENSE`, `THIRD_PARTY_NOTICES.md` and `licenses/*.txt`.

## 3. Installed-wheel tests

Run the unit tests against the installed wheel (not the source tree), without optional extras:

```bash
uv venv -p 3.13 /tmp/imagehub-wheel && uv pip install --python /tmp/imagehub-wheel dist/*.whl pytest
uv venv -p 3.11 /tmp/imagehub-min && uv pip install --python /tmp/imagehub-min "torch==2.2.2" "numpy<2" dist/*.whl pytest
```

For each environment run `python -m pytest <repo>/tests -q -p no:cacheprovider --rootdir <repo> --import-mode=importlib`
from outside the repository and confirm `imagehub.__file__` points into site-packages.

## 4. CUDA smoke test (maintainer approval)

On an approved GPU machine:

```bash
pip install -e ".[dev,reference]"
python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5
IMAGEHUB_REFERENCE_TESTS=1 IMAGEHUB_SD_V1_5=$PWD/checkpoints/sd-v1-5 \
  IMAGEHUB_RECORD=cuda-measurements.json python -m pytest tests/reference/test_ref_devices.py -m cuda
```

Merge the `smoke_cuda` record into `docs/validation-measurements.json`, rebuild
`docs/validation.json`, and fill the CUDA row in `docs/validation.md`.

## 5. Version and publish (maintainer approval)

1. Bump `version` in `pyproject.toml` and `__version__` in `imagehub/__init__.py` together.
2. Show the maintainer the wheel and sdist listings and the verification results, then publish
   with `uv publish dist/*` using the maintainer's token.
3. Tag the release (`vX.Y.Z`) on the published commit.
