---
title: Contributing
---

# Contributing

The most useful contribution is a new model. The bar is the one every existing model met: it must
reproduce the reference implementation, and the evidence has to be in the repository.

## Setup

```bash
uv venv -p 3.13 .venv && uv pip install --python .venv -e ".[dev,image,docs]"
.venv/bin/python scripts/fetch_fixtures.py
.venv/bin/python -m pytest
.venv/bin/mkdocs serve        # preview this site
```

Set `HF_TOKEN` when your accepted repository licenses grant access to gated fixtures. Restricted
files are verified against `tests/fixtures/RESTRICTED.json` and remain in the ignored
`tests/fixtures/_restricted` cache.

## Adding a model

1. **Open an issue** naming the model, its repository and its weight license. One issue per model
   keeps review focused.
2. **Download the real configs first** and keep them as test fixtures
   (`tests/fixtures/<model>_configs/`). Write components against the configs that were actually
   published, not against assumptions.
3. **Implement the components** in `imagehub/models/` (and schedulers in `imagehub/schedulers/`,
   tokenizers in `imagehub/tokenization_*.py`) as plain PyTorch. Share a component with an
   existing family only when it is genuinely the same code.
4. **Add the backend** in `imagehub/backends/`: a `TextToImageBackend` that validates options
   before loading, loads lazily, and returns an `ImageResult` whose metadata records the run.
   Register it in `imagehub/registry.py`.
5. **Add the catalog entry** in `imagehub/catalog.py`: model id, pinned 40-character revision, file
   list, license, provenance, and the `catalog_id` that links to
   [awesome-text-to-image-architectures](https://github.com/gorkemgul/awesome-text-to-image-architectures).
6. **Prove parity** with reference tests under `tests/reference/`: strict load, each component, then
   an end-to-end generation from the same noise on CPU in float32. They are skipped unless
   `IMAGEHUB_REFERENCE_TESTS=1` and never download anything. Tests that import the authors' code
   find its checkout with `reference_source("<name>")` from `tests/reference_sources.py`, which reads
   `IMAGEHUB_REFERENCE_SOURCES` (default `checkpoints/ref-src`).
7. **Record the result** in `docs/validation.json` (with `scripts/build_validation.py`) and
   `docs/validation.md`, add the model to `docs/models.md` and `docs/catalog.md`, and to the table in
   the README.
8. **Open a pull request** against `main`, one model per PR, closing its issue.

??? tip "What a models.md section needs"
    A level-1 heading with the architecture name, then the line
    `Catalog: [...] · Models: \`id\`, ... · Task: \`t2i\``, then **Architecture**, **Usage**,
    **Settings**, **Seed compatibility**, **Out of scope** and **License** paragraphs. The site builds
    the family page, the model grid and the navigation from this file and the catalog, so there is
    nothing else to edit.

## What the docs site builds from

Nothing on this site repeats a fact by hand.

| Page | Source |
| --- | --- |
| Landing counts and model grid | `imagehub/catalog.py`, `docs/validation.json` |
| Family pages | `docs/models.md` split by catalog family |
| Results | `docs/validation.json` |
| Gallery | `docs/assets/gallery/*.json` and the image pairs beside them |
| Catalog, Validation record | `docs/catalog.md`, `docs/validation.md` |

The generator is `scripts/docs_hooks.py`; it runs inside `mkdocs serve` and `mkdocs build`.
`mkdocs build --strict` must pass.

### Adding a gallery pair

Put `<model-id>.reference.webp`, `<model-id>.imagehub.webp` and `<model-id>.json` (name, family,
settings, `max_pixel_diff`) into `docs/assets/gallery/`. The pair appears on the gallery page and on the
model's own page.

### Translating

English pages are the source. A translation is a sibling file with the locale before the extension,
for example `quickstart.tr.md` next to `quickstart.md`. Pages without a translation fall back to
English, so a language can grow page by page.

## Style

Code is formatted with ruff (line length 100) and type-checked with mypy. Documentation is plain
and specific: state what was measured, on what, and what is not claimed.
