# BAGEL implementation plan

1. Capture the pinned model configuration, tokenizer configuration, weight index and tensor
   headers as offline fixtures.
2. Add native BAGEL MoT, vision-layout and autoencoder modules, with tests for exact published
   names and tiny tensor behavior.
3. Add prompt caching, rectified-flow generation, option validation and snapshot loading in a
   BAGEL backend; test deterministic tiny generation and validation-before-load behavior.
4. Register the pinned checkpoint and backend and cover catalog/download invariants.
5. Add opt-in reference parity tests, documentation, validation grouping and licensing notices.
6. Download all pinned checkpoint files, build a verified local snapshot, then run formatting,
   typing, unit, clean-environment and reference skip checks.
