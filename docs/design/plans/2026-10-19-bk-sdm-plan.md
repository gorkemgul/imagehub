# BK-SDM implementation plan

1. Store the pinned published configs and safetensors tensor layouts for Base, Small and Tiny; add
   catalog and strict-layout assertions for all three checkpoints.
2. Extend the shared U-Net to represent diffusers' `mid_block_type=null` as an identity, and verify
   exact random-weight parity for every BK-SDM structure.
3. Reuse the Stable Diffusion backend and native CLIP, VAE, tokenizer and PNDM scheduler unchanged;
   add tiny BK-SDM snapshots and exact guided/unguided non-square pipeline parity.
4. Add dedicated real-checkpoint header, tokenizer, text encoder, U-Net, VAE, scheduler and separate
   seed-0/seed-42 public-pipeline parity tests in `tests/reference/test_ref_bk_sdm.py`.
5. Add BK-SDM to the README Stable Diffusion row, catalog and model documentation, validation
   narrative, validation grouping and third-party notices.
6. Build all three local verified snapshots, run targeted and shared-module reference checks,
   formatting, lint, typing and both required fast suites.
