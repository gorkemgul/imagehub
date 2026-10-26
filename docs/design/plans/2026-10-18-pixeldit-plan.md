# PixelDiT implementation plan

1. Capture the pinned PixelDiT and Gemma-2 configs and tensor layouts in
   `tests/fixtures/pixeldit_configs`.
2. Add a native `PixelDiT` model preserving the published `core.*` tensor names, including
   rotary joint attention, patch and pixel pathways, and strict checkpoint preparation.
3. Add the backend with prompt encoding, classifier-free FLOW DPM-Solver++ sampling, direct
   pixel conversion, option validation, snapshot loading, and result metadata.
4. Add the catalog entry, subrepository files, and lazy registry entry.
5. Add unit tests, exact random-weight and tiny end-to-end reference tests, plus opt-in
   real-checkpoint component and public-pipeline tests.
6. Document the model, catalog, validation scope, licenses, and validation grouping; run all
   required formatting, type, fast-suite, clean-environment, and reference checks.
