# KOALA-Lightning implementation plan

1. Store both pinned published config sets and safetensors tensor layouts; add catalog and strict
   layout assertions for `koala-lightning-700m` and `koala-lightning-1b`.
2. Reuse the SDXL backend and native components unchanged; configure 10 steps, guidance 3.5 and
   trailing Euler spacing, then add exact random-weight U-Net and tiny-pipeline parity tests.
3. Add dedicated real-checkpoint header, tokenizer, text encoder, U-Net, VAE, scheduler and
   end-to-end/seed reference tests in `tests/reference/test_ref_koala_lightning.py`.
4. Add both checkpoints to the README SDXL row, catalog and model documentation, validation
   narrative, validation grouping and third-party notices.
5. Build both local verified snapshots, run targeted checks, formatting, lint, typing, both
   required fast suites and the new reference tests without running long real-checkpoint tests.
