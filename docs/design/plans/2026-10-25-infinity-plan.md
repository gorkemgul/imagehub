# Infinity-2B implementation plan

1. Pin and fixture the official weight repository, Flan-T5-XL assets, PyTorch archive tensor
   layouts, and the exact original-code revision.
2. Add failing published-layout and random-weight parity tests for the Infinity transformer,
   binary spherical quantizer, and VAE decoder.
3. Implement the native modules with published tensor names and plain PyTorch inference paths.
4. Add bitwise next-scale sampling, the backend, tiny published-layout snapshots, option and
   lifecycle tests, registry entry, and pinned catalog entry.
5. Add subprocess-isolated real-checkpoint component tests and separate seed-0/seed-42 short
   end-to-end public-pipeline tests.
6. Update the README, catalog, models, validation, validation grouping, and third-party notices.
7. Format, lint, type-check, run affected reference tests and both fast suites, then create the
   verified local snapshot manifest from the completed foreground downloads.
