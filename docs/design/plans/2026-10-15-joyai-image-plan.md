# JoyAI-Image implementation plan

1. Record the pinned Edit and Edit Plus configs, tokenizer/processor files, weight indexes, and
   safetensors headers; add both checkpoints to the `joyai-image` catalog family.
2. Implement the published JoyAI dual-stream transformer, including the Edit Plus padded patch
   sequence and batched rotary-position path, with exact random-weight reference tests.
3. Extend the native Qwen3-VL text model for visual embeddings and deep-stack features, then add
   the vision patch encoder and merger with strict header and exact reference coverage.
4. Implement one-prompt, one-image inference for both reference pipelines, including preprocessing,
   conditioning, VAE latents, guidance, flow-match sampling, decoding, and result metadata.
5. Add tiny exact end-to-end tests and snapshot-gated header, tokenizer/vision, component,
   end-to-end, and independent seed records for each checkpoint.
6. Update the registry and model, catalog, validation, and README documentation; run shared-user
   reference coverage, formatting, type checks, and both regular and clean-environment fast suites.
