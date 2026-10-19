# BK-SDM design

- Date: 2026-10-19
- Issue: #158
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Add `bk-sdm-base`, `bk-sdm-small` and `bk-sdm-tiny`, three ungated CreativeML OpenRAIL-M
text-to-image checkpoints in the existing `stable-diffusion` family. The author repositories and
pinned revisions are:

| Model | Repository | Revision |
| --- | --- | --- |
| BK-SDM Base | `nota-ai/bk-sdm-base` | `0c03b7a0369b49f97d1acf7256d4ee55ced9b2e0` |
| BK-SDM Small | `nota-ai/bk-sdm-small` | `572238db7ed3a10858900803f3fc8cca53e893e0` |
| BK-SDM Tiny | `nota-ai/bk-sdm-tiny` | `0364108e53b7f7f4d2585e817a0b7a83dc261cfa` |

The catalog downloads the full-precision component safetensors. It excludes duplicate pickle and
single-file checkpoints.

## Reference

Diffusers 0.40.0 `StableDiffusionPipeline` and transformers 5.18.0. Every published model index
names `StableDiffusionPipeline`.

## Components

| Part | Status |
| --- | --- |
| CLIP ViT-L/14 text encoder and tokenizer | existing Stable Diffusion implementation; identical published blobs |
| Block-pruned conditional U-Net | existing configurable U-Net; add support for an omitted mid block |
| f=8 KL autoencoder | existing Stable Diffusion implementation; identical published blobs |
| PNDM scheduler | existing Stable Diffusion implementation; identical published config |

Base keeps four U-Net stages and its cross-attention mid block. Small removes the mid block. Tiny
removes the mid block and the innermost stage, leaving three down and three up stages. All variants
use one residual/attention layer per block rather than Stable Diffusion 1.4's two.

## Pipeline

Generation follows `StableDiffusionPipeline.__call__`: the final CLIP hidden state conditions the
U-Net, classifier-free guidance is optional, PNDM denoises standard normal latents, and the VAE
decodes them before uint8 conversion. Defaults remain 50 steps, guidance 7.5 and 512×512.

## Memory

The published U-Nets contain approximately 0.76B, 0.66B and 0.50B parameters. A complete fp32
pipeline needs roughly 4–5 GB, and component-by-component validation avoids holding two complete
pipelines longer than necessary.

## Validation

Published tensor headers cover all three weighted components of every checkpoint. Dedicated tests
cover strict layouts, exact random-weight U-Net parity for all three structures, exact tiny-pipeline
parity with and without guidance at non-square sizes, tokenizer/text-encoder/U-Net/VAE/scheduler
component parity, and separate short public-pipeline end-to-end runs at seeds 0 and 42 for every
checkpoint. The real runs use four steps and 64×64 output to keep controller runtime short.

## Out of scope

Training, DreamBooth, image-to-image, safety checking, fp16 parity, SD 2.1-derived BK-SDM variants,
2M-data variants, Core ML exports, duplicate pickle weights and single-file checkpoints are out of
scope.
