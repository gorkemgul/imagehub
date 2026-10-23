# DreamShaper 8 implementation plan

1. Store the pinned published configs and full-precision safetensors layouts.
2. Add the pinned `dreamshaper-8` catalog entry and catalog/layout tests.
3. Reuse the Stable Diffusion backend and existing SD 1.x components unchanged.
4. Add dedicated real-checkpoint component tests and separate seed-0/seed-42 end-to-end tests.
5. Update the README, catalog, model, validation, validation grouping and third-party documentation.
6. Build the local verified snapshot and run targeted tests, formatting, lint, typing and both fast
   suites.
