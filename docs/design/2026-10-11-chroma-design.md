# Chroma design

- Date: 2026-10-11
- Issue: #111
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

One weight set of a new family `chroma`, text-to-image, at `full` validation status, catalog id `chroma1-hd`, by the authors,
not gated, Apache-2.0 (model card):

| Model id | Repository | Revision | Transformer | Text encoder | VAE |
| --- | --- | --- | --- | --- | --- |
| `chroma1-hd` | `lodestones/Chroma1-HD` | `0e0c60ece1e82b17cb7f77342d765ba5024c40c0` | 8.9B, two bfloat16 shards | T5-XXL 4.7B, two bfloat16 shards | FLUX autoencoder, bfloat16 |

Only bfloat16 weights are published; the catalog lists those files. The repository also contains a single-file
`Chroma1-HD.safetensors` checkpoint; the diffusers-layout shards are used instead.

Out of scope: img2img/inpainting pipelines, IP-Adapter, LoRA, single-file loading, reduced-precision parity.

## Reference

diffusers 0.40.0 `ChromaPipeline` (`pipelines/chroma/pipeline_chroma.py`), `ChromaTransformer2DModel`
(`models/transformers/transformer_chroma.py`); the FLUX autoencoder, T5-XXL encoder/tokenizer and flow-match Euler
scheduler are reused from existing imagehub modules.

## Components

| Part | Status |
| --- | --- |
| T5 tokenizer (SentencePiece Unigram, `tokenizer.json`, `added_tokens.json`) | **exists** |
| T5-XXL text encoder (`d_model=4096`, 24 layers, gated GELU, attention mask) | **exists** |
| `ChromaTransformer2DModel` (distilled-guidance approximator, RoPE, joint attention with text/image attention mask) | **new** |
| FLUX `AutoencoderKL` (`scaling_factor=0.3611`, `shift_factor=0.1159`) | **exists** |
| Flow-match Euler scheduler with `shift=3.0` and exponential shift | **exists** |

## Pipeline

As `ChromaPipeline.__call__` for one prompt: T5 tokenization to `max_sequence_length` with padding; T5 encoding with the
padding attention mask; a Chroma-specific attention mask that keeps the real text tokens plus one padding token and masks
the rest; classifier-free guidance with the supplied negative prompt (default `""`) when `guidance_scale > 1`; 35 steps at
1024×1024 by default; latents `randn(B, 16, H/8, W/8)` then packed into 2×2 patches; custom flow-match sigmas with mu
shift; the transformer predicts noise; unpack and VAE decode with `latents / scaling_factor + shift_factor`; uint8 as
diffusers.

The large model (~8.9B transformer + ~4.7B text encoder) requires the reference end-to-end result to be computed, saved
and freed before loading the native pipeline.

## Memory

fp32: transformer ~35 GB, text encoder ~19 GB, VAE ~1 GB. Reference and native cannot both reside in memory, so
real-checkpoint parity is done component-by-component and the end-to-end reference output is saved to disk before the
native run.

## Validation

Tokenizer parity on fixed prompts; random-weight parity for the transformer (tiny and published config at small spatial
sizes); scheduler parity; tiny end to end against `ChromaPipeline` (guidance on/off, non-square, two seeds); header check
for the published bfloat16 shards; real-checkpoint tests that skip cleanly without the snapshot.

## Task breakdown

1. `ChromaTransformer2DModel` native module with header checks and random-weight parity.
2. `chroma` backend, catalog entry and registry registration.
3. Unit tests and the tiny end-to-end reference parity test.
4. Real-checkpoint test file and docs; the pull request closes #111.
