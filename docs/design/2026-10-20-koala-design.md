# KOALA design

- Date: 2026-10-20
- Issue: #161
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Add `koala-700m` and `koala-1b`, two text-to-image checkpoints in the existing
`stable-diffusion-xl` family. The author-published weights come from `etri-vilab/koala-700m` at
revision `5e73a318a1ee1b97cc31fd50f09ac13e018c0b63` and `etri-vilab/koala-1b` at revision
`8ef26f38fe4d141c35fda24966bd48777fc3c8ad`.

The catalog downloads the full-precision component safetensors. It excludes fp16 copies, framework
conversions, duplicate VAE files and deployment exports. The Hub repositories do not declare a
weight license, so the catalog records `Unspecified`; the linked author repository separately
describes dual licensing for its software.

## Reference

Diffusers 0.40.0 `StableDiffusionXLPipeline`, pinned at
`d035dcd7cc7c88e0a154609b62887d50bba9fdc2`, and transformers 5.18.0. Both published model indexes
name `StableDiffusionXLPipeline`.

## Components

| Part | Status |
| --- | --- |
| CLIP ViT-L/14 text encoder and tokenizer | existing SDXL implementation; published layout matches |
| OpenCLIP bigG text encoder and tokenizer | existing SDXL implementation; published layout matches |
| KOALA-700M U-Net, transformer depths `[1, 2, 5]`, no mid-block | existing configurable U-Net |
| KOALA-1B U-Net, transformer depths `[1, 2, 6]`, cross-attention mid-block | existing configurable U-Net |
| f=8 KL autoencoder | existing SDXL implementation; published scaling factor `0.13025` |
| Euler scheduler with leading spacing | existing SDXL implementation; published config matches |

No shared native module changes are expected.

## Pipeline

Generation follows `StableDiffusionXLPipeline.__call__`: two penultimate CLIP hidden states,
OpenCLIP pooled conditioning, six SDXL size/crop time ids, optional classifier-free guidance,
Euler denoising, VAE decoding and uint8 conversion. Defaults remain 50 steps, guidance 5.0 and
1024×1024. The model card examples use 25 steps and negative prompting, which are usage
recommendations rather than pipeline defaults.

## Memory

The published U-Nets contain about 0.78B and 1.16B parameters. Their full-precision files are about
3.1 GB and 4.6 GB; shared text encoders and VAE add about 3.6 GB. Component validation limits how
long two complete implementations coexist in memory.

## Validation

Published tensor headers cover all four weighted components for both repositories. Dedicated tests
cover exact random-weight U-Net parity for both layouts, exact tiny-pipeline parity with and without
guidance at non-square sizes, tokenizer/text-encoder/U-Net/VAE/scheduler component parity, and two
short real-checkpoint end-to-end runs per checkpoint through the public pipeline API (seed 0 and
seed 42). Real end-to-end runs use four steps and 64×64 output for controller runtime.

## Out of scope

Training, LoRA, image-to-image, safety checking, watermarking, fp16 parity, framework conversions,
deployment exports and duplicate weight files are out of scope.
