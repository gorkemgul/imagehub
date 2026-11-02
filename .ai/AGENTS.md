# Repository Guidelines

## Guidance ownership

Each concern has exactly one owner:

| File | Owns |
| --- | --- |
| `.ai/GOAL.md` | The product contract, invariants, completion criteria and roadmap |
| `.ai/LOOP.md` | How one bounded iteration is selected, implemented, verified and delivered |
| `.ai/AGENTS.md` | Repository operations, safety rules, language and skill routing |
| `.ai/review-rules.md` | First-pass pull-request review |
| `.ai/skills/` | Task-specific workflows |

Read `.ai/GOAL.md` and `.ai/LOOP.md` completely before planning or editing. Do not duplicate
guidance elsewhere.

## Language

Everything committed to the repository is in English: code, identifiers, comments,
docstrings, configuration, documentation, commit messages and pull requests. Conversation with
the maintainer may use the maintainer's language.

## Hard rules

- Runtime dependencies stay `torch>=2.2` only (see `.ai/GOAL.md`, Invariants).
- Commit messages and pull requests carry no AI attribution lines (no `Co-Authored-By`, no
  "Generated with" footers).
- Do not describe other people's projects as inspiration or as the source of the design in
  repository content. Attribution for code or conventions that were followed belongs only in
  `THIRD_PARTY_NOTICES.md` and `licenses/`.
- Never commit model weights, downloaded snapshots, `dist/`, or working notes. Snapshots live in
  `checkpoints/`; working notes live in the git-ignored `docs/research/`.
- Never loosen a parity tolerance to make a test pass. Measure, record, and apply the tolerance
  rule in `.ai/skills/add-or-validate-image-model/SKILL.md`.

## Actions that need the maintainer's explicit approval

Pushing to `main`, force-pushing, rewriting published history, deleting branches or
repositories, publishing to PyPI, creating paid cloud resources (for example RunPod), and
changing repository settings. Work on a feature branch and open a pull request instead.

## Environment

```bash
uv venv -p 3.13 .venv
uv pip install --python .venv -e ".[dev,image,reference]"
.venv/bin/python -m pytest
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
```

- ruff excludes Markdown (`extend-exclude = ["*.md"]`), because it would reformat code
  blocks inside documentation.
- Reference tests run only with `IMAGEHUB_REFERENCE_TESTS=1` and the snapshot variables, for
  example `IMAGEHUB_SD_V1_5=$PWD/checkpoints/sd-v1-5`. They never download anything.
- On python.org builds of Python for macOS, downloads may fail TLS verification. Set
  `SSL_CERT_FILE="$(.venv/bin/python -m certifi)"` or run the installer's
  `Install Certificates.command`.

## Skill routing

Read the whole matching skill before acting:

| Task | Skill |
| --- | --- |
| Adding, completing or auditing a model, checkpoint, tokenizer, scheduler or backend | `.ai/skills/add-or-validate-image-model/SKILL.md` |
| Packaging, versioning, validation records, CUDA smoke tests, PyPI releases | `.ai/skills/prepare-release-evidence/SKILL.md` |
| Reviewing a pull request | `.ai/review-rules.md` |

## Layout

```
imagehub/
  catalog.py        published weight sets (ModelSpec), linked to the awesome catalog
  registry.py       lazy backend selection; catalog ids are aliases of their family
  factory.py        pipeline()
  pretrained.py     python -m imagehub.pretrained (explicit, verified downloads)
  download.py       HTTPS, pinned commits, sha256 / git blob sha1 verification
  checkpoints.py    strict safetensors / weights-only pickle loading, NativeModel, snapshot manifest
  tokenization.py   native tokenizers
  image.py, types.py, exceptions.py
  models/           network definitions that keep the published tensor names
  schedulers/       samplers shared across families
  backends/         one adapter per family: loading, option validation, inference
  pipelines/        task orchestration
tests/              unit tests (no network) and tests/reference/ (opt-in parity)
docs/               catalog.md, models.md, validation.md, validation.json
```
