# Show-o implementation plan

1. Store the three pinned published config sets and both safetensors layouts; add the `show-o-512`
   catalog entry and strict meta-device layout tests.
2. Implement native Phi-1.5, MAGVIT-v2, CodeGen tokenization, omni-attention prompting, discrete
   denoising, and a local-snapshot backend.
3. Add exact random-weight component and tiny-pipeline reference parity plus unit coverage for
   validation, deterministic seeds, request isolation, and autograd restoration.
4. Add opt-in real-checkpoint component and separate seed-0/seed-42 short end-to-end tests.
5. Update the README, catalog, model and validation documentation, validation grouping, and notices;
   build the verified local snapshot and run all required checks.
