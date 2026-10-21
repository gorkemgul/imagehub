# FLUX.1-Lite design

- Date: 2026-10-20
- Issue: #162
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Add `flux-1-lite-8b`, Freepik's ungated 8B distillation of FLUX.1-dev, to the existing
`flux` family. The checkpoint is `Freepik/flux.1-lite-8B` at revision
`16b9373dd3d05bc5f125558f5e2b8470fbb873e8`. The catalog uses its component-based diffusers
layout and excludes the duplicate single-file checkpoint and sample assets.

## Reference

Diffusers 0.40.0 `FluxPipeline` and `FluxTransformer2DModel`, with transformers 5.18.0. The
published model index identifies the same pipeline and component classes.

## Components

| Part | Status |
| --- | --- |
| CLIP ViT-L/14 text encoder and tokenizer | existing FLUX.1 implementation; published layout matches |
| T5-v1.1-XXL text encoder and tokenizer | existing FLUX.1 implementation; published layout matches |
| FLUX transformer | existing configurable implementation; 8 double-stream and 38 single-stream blocks |
| 16-channel KL autoencoder | existing FLUX.1 implementation; published layout matches |
| Flow-match Euler scheduler | existing dynamic-shifting implementation; published config matches |

The repository publishes bfloat16 component weights only. CPU parity upcasts the same stored
weights to float32 on both sides. No shared native module change is required.

## Pipeline

Generation follows `FluxPipeline.__call__`: CLIP pooled conditioning, T5 sequence conditioning,
2x2 latent packing, dynamically shifted flow-match Euler denoising, latent unpacking and VAE
decoding. The model-card defaults are 28 steps, guidance 3.5, 512 text tokens and 1024x1024.

## Memory

The distilled transformer has about 8B parameters and its two published shards total about
16.3 GB in bfloat16. Real end-to-end reference output is computed in a subprocess and freed
before the native pipeline loads, avoiding simultaneous full pipelines.

## Validation

Published config and tensor-header fixtures cover the reduced transformer layout. Existing exact
FLUX.1 random-weight and tiny-pipeline parity tests cover the configurable architecture and full
generation path. A dedicated real-checkpoint suite covers strict headers, both tokenizers and text
encoders, one transformer step, VAE encode/decode, scheduler behavior, and separate short public-
pipeline end-to-end runs for seed 0 and seed 42. The real runs are left to the controller.

## Out of scope

Training, LoRA, the duplicate single-file checkpoint, safety checking, watermarking and fp16 or
bfloat16 device parity are out of scope.
