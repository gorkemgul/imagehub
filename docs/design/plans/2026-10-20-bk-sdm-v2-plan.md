# BK-SDM v2 implementation plan

1. Store pinned published configs and safetensors layouts for Base, Small and Tiny.
2. Add failing catalog, strict-layout, tiny generation and exact reference parity tests for all three IDs.
3. Reuse the Stable Diffusion backend and existing SD 2.1/BK-SDM native components unchanged.
4. Add dedicated real-checkpoint component and separate seed-0/seed-42 end-to-end reference tests.
5. Update the README, catalog, model, validation, grouping and third-party attribution documentation.
6. Build the three local verified snapshots and run targeted tests, formatting, lint, typing and both fast suites.
