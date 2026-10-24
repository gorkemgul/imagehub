# BAGEL design

## Scope

Add text-to-image inference for `ByteDance-Seed/BAGEL-7B-MoT` under the `bagel-7b-mot`
catalog id. Image understanding, editing, reasoning-token generation, quantized loading and
TaylorSeer acceleration are outside this change.

## Published artifacts

The author checkpoint is pinned to revision
`5019f57d168e5816e8f3f701b17cc816bb7cf24b` and is Apache-2.0 licensed. It contains one
bfloat16 EMA model file, one float32 autoencoder file, Qwen2 tokenizer files and the published
configuration files. The reference is the author repository at
`a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f` and its `InterleaveInferencer` /
`Bagel.generate_image` path.

## Components

- A native Qwen2 Mixture-of-Transformers decoder keeps separate understanding and generation
  projections, normalizations and MLPs while sharing the token sequence and attention.
- The published SigLIP vision branch is represented and strictly loaded even though text-only
  generation does not execute it.
- BAGEL's timestep, latent patch projections and fixed two-dimensional position embeddings are
  native PyTorch modules with published tensor names.
- The autoencoder reuses the native KL autoencoder implementation with a key conversion for the
  original FLUX tensor layout.
- The existing native Qwen2 tokenizer reads the published tokenizer JSON files.

## Pipeline behavior

The backend first caches the prompt through the understanding expert. Each rectified-flow step
runs a non-causal image query containing image boundary tokens and all latent patches. Text CFG
uses an empty prompt cache, matching the official pure text-to-image path. Euler integration runs
from noise at 1 to data at 0. Defaults are 1024x1024, 50 steps, text guidance 4, CFG interval
0.4-1.0 and timestep shift 3.

## Memory and validation

The release has 7B active and 14B total parameters. Loading requires roughly 30 GB for published
bfloat16 weights, plus attention and decoder working memory. Unit tests use tiny random models;
reference tests cover published headers, tokenizer, random-weight decoder/autoencoder parity,
components and two shortened 64x64 one-step end-to-end runs. The long default generation is left
to validation hardware.

