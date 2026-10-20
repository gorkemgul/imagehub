# KOALA implementation plan

1. Store both pinned published config sets and safetensors tensor layouts; add catalog and strict
   layout assertions for `koala-700m` and `koala-1b`.
2. Reuse the SDXL backend and native components unchanged; add exact random-weight U-Net and tiny
   pipeline parity tests for both published layouts.
3. Add dedicated real-checkpoint header, tokenizer, text encoder, U-Net, VAE, scheduler and short
   end-to-end/seed reference tests in `tests/reference/test_ref_koala.py`.
4. Add both checkpoints to the README SDXL row, catalog and model documentation, validation
   narrative, validation grouping and third-party notices.
5. Build both local verified snapshots, run targeted checks, formatting, lint, typing, both required
   fast suites and the new reference tests without running long real-checkpoint generation.
