# Translations

The documentation is written in English, which stays at the site root. The Turkish version is
served under `/tr/`. A page without a Turkish translation is shown in English on the Turkish site,
so every page is reachable in both languages.

## Coverage

| Pages | English | Turkish |
| --- | --- | --- |
| Home, Quickstart, Guides, Parity overview, Gallery, Results, API, Contributing, Translations | source | translated |
| Model pages (generated from `docs/models.md`) | source | English fallback |
| Model catalog (`docs/catalog.md`) | source | English fallback |
| Full validation record (`docs/validation.md`) | source | English fallback |
| Navigation labels | source | translated |

The model pages, the catalog and the validation record are reference material that changes with
every model, so they are kept in English only.

## File naming

The site uses the suffix structure of `mkdocs-static-i18n`. Keep the English file name and add the
locale before `.md`:

```text
docs/
├── index.md
├── index.tr.md
└── guides/
    ├── training.md
    └── training.tr.md
```

Do not create a separate directory tree for a language.

## Rules

- A change to a translated page updates both files in the same pull request.
- Keep heading levels, code blocks, anchors, HTML attributes and relative link targets unchanged.
- Translate prose, headings, table text and image descriptions.
- Do not translate model ids, Python names, options, commands, file paths or license names.
- Check numbers and support claims against the English page and `imagehub/catalog.py`.
- A new page in the navigation also needs its label in `nav_translations` in `mkdocs.yml`.

## Build

```bash
mkdocs build --strict
```

The build writes the English site at the root and the Turkish site under `tr/`. Check both, and
the language switcher, before opening the pull request.
