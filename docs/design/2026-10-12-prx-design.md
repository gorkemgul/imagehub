# PRX design

- Date: 2026-10-12
- Issue: #123

## Scope

PRX is a text-to-image family with `prx-1024-t2i-beta` and `prx-512-t2i-sft`. Both author-published Apache-2.0
repositories use a 16-block, 1792-wide PRX denoiser, the FLUX AutoencoderKL and a flow-match Euler scheduler. The
1024 repository is pinned at `0370504b9a433fbdff814b8fcba9da03bf72a7a3`; the 512 repository is pinned at
`2996423bc26e8eaca48774fac1797484214dfea0`. Their transformer and VAE are published as single safetensors files;
the shared text encoder is published in three float32 safetensors shards.

## Reference and components

The reference is diffusers 0.40.0 `PRXPipeline` and `PRXTransformer2DModel`. The native components are the PRX
transformer and the encoder-only T5Gemma model. The transformer implements patch sequence conversion, two-axis RoPE,
masked text/image attention, adaptive modulation and the final patch projection. T5Gemma reuses the matching Gemma2
attention, MLP and normalization pieces while preserving the published flat tensor layout. The existing Gemma
tokenizer, FLUX AutoencoderKL and flow-match Euler scheduler are reused.

## Validation

Fixtures retain all published component configurations and safetensors headers. Meta-device layout checks cover the
denoiser and T5Gemma encoder. Exact random-weight tests cover both modules, and a tiny pipeline test covers guidance
on/off, non-square output and two seeds. Snapshot-gated tests cover tokenizer fuzz, every component, end-to-end output
and seed parity for both published checkpoints; their measurements are pending a real-checkpoint run.
