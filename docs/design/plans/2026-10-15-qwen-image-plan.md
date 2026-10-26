# Qwen-Image implementation plan

1. Add a native original-release dual-stream transformer with published tensor names,
   non-persistent RoPE buffers, strict meta-device loading, and exact random-weight parity
   tests for tiny and published-width configurations.
2. Split the Qwen-Image backend by snapshot configuration while retaining the 2.1 path:
   load Qwen2.5-VL, the original transformer and VAE; reproduce the original prompt,
   packing, dynamic shift, true-CFG normalization, and decoding behavior.
3. Add a tiny original-layout snapshot helper and exact end-to-end diffusers comparisons for
   two seeds, guidance on/off, and a non-square image, plus unit coverage for dispatch and
   option validation.
4. Add opt-in real-checkpoint tests for headers, tokenizer, text encoder, transformer, VAE,
   scheduler, shortened end-to-end generation, and separate public-pipeline seed parity.
5. Run formatting, static checks, focused reference tests, the full fast suite in both Python
   environments, and document the uncommitted result in the requested report.
