# Emu3-Gen implementation plan

1. Pin the official code, generator and vision-tokenizer revisions; capture configs, indexes and
   safetensors headers under `tests/fixtures/emu3_gen_configs`.
2. Add native Emu3 byte-BPE tokenization, causal language-model/cache modules, constrained sampling,
   and the complete published vision-tokenizer tensor layout and decoder.
3. Add the lazy backend, catalog entry and registry mapping with strict snapshot loading and option
   validation before model loading.
4. Add offline unit tests, opt-in random/reference parity, and real-snapshot component plus separate
   seed-0/seed-42 public-pipeline comparisons.
5. Document the family in the README, catalog, model and validation pages, validation grouping and
   third-party notices; then run formatting, typing, focused reference tests and both fast suites.

