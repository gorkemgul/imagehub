# Segmind SD implementation plan

1. Store pinned published configs and tensor layouts for both checkpoints; add catalog and strict
   layout assertions.
2. Add the published no-mid-block U-Net option and the two published scheduler configurations,
   with exact random-weight/reference trajectory tests.
3. Reuse the Stable Diffusion backend and add tiny snapshot end-to-end parity coverage.
4. Add dedicated real-checkpoint header, tokenizer, text encoder, U-Net, VAE, scheduler, and
   separate seed-0/seed-42 public-pipeline tests for both ids.
5. Update the README, catalog/model/validation documentation, validation grouping, and notices;
   build both local verified snapshots and run all shared-module and final checks.
