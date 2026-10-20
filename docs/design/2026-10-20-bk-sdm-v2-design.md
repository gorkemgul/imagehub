# BK-SDM v2 design

- Date: 2026-10-20
- Issue: #164
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Add `bk-sdm-v2-base`, `bk-sdm-v2-small` and `bk-sdm-v2-tiny`, three ungated CreativeML OpenRAIL-M text-to-image checkpoints in the existing `stable-diffusion` family. The author repositories are pinned to these revisions:

| Model | Repository | Revision |
| --- | --- | --- |
| BK-SDM v2 Base | `nota-ai/bk-sdm-v2-base` | `21915be4c3a83f3fa45f93559750df44943509b1` |
| BK-SDM v2 Small | `nota-ai/bk-sdm-v2-small` | `ad82c51a3b568e855a7f79c5b893ea2f9ebb197f` |
| BK-SDM v2 Tiny | `nota-ai/bk-sdm-v2-tiny` | `68277af553777858cd47e133f92e4db47321bc74` |

The catalog downloads the full-precision component safetensors and excludes duplicate pickle and fp16 weights.

## Reference

Diffusers 0.40.0 `StableDiffusionPipeline` and transformers 5.18.0. Each published model index names `StableDiffusionPipeline`.

## Components

The checkpoints combine the existing SD 2.1 OpenCLIP text encoder, tokenizer, VAE and DDIM scheduler with the three existing BK-SDM pruning layouts. The U-Nets use 1024-wide conditioning, per-stage attention head dimensions and linear attention projections. All required configuration is already supported, so no shared runtime module changes are planned.

## Pipeline

Generation follows the existing Stable Diffusion backend: the final text encoder hidden state conditions the U-Net, classifier-free guidance is optional, DDIM denoises standard normal latents and the VAE decodes them before uint8 conversion. Published defaults are 50 steps, guidance 7.5 and 512×512.

## Memory

The published U-Nets contain approximately 0.98B, 0.88B and 0.72B parameters. Component-by-component validation limits peak memory; short 64×64 end-to-end comparisons use four denoising steps.

## Validation

Fixtures and meta-device checks cover every published tensor. Tests cover exact random-weight U-Net parity at tiny and published layouts, exact tiny-pipeline parity with guidance on and off at non-square sizes, real-checkpoint tokenizer/text-encoder/U-Net/VAE/scheduler parity, and separate public-pipeline runs at seeds 0 and 42 for each checkpoint.

## Out of scope

Training, DreamBooth, image-to-image, safety checking, fp16 parity, 2M-data variants, Core ML exports, duplicate pickle weights and single-file checkpoints are out of scope.
