# Segmind-Vega implementation plan

1. Store the pinned published configs and safetensors tensor layouts; add catalog and strict-layout
   assertions for the new checkpoint.
2. Reuse the SDXL backend and native components unchanged; add tiny Segmind-Vega snapshot support
   and exact random-weight U-Net plus tiny-pipeline parity tests.
3. Add dedicated real-checkpoint header, tokenizer, text encoder, U-Net, VAE, scheduler and short
   end-to-end/seed reference tests in `tests/reference/test_ref_segmind_vega.py`.
4. Add Segmind-Vega to the README SDXL row, catalog and model documentation, validation narrative,
   validation grouping and third-party notices.
5. Build the local verified snapshot, run targeted checks, formatting, lint, typing, both required
   fast suites and the new reference tests without real weights.
