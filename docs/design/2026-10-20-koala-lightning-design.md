# KOALA-Lightning design

- Date: 2026-10-20
- Issue: #165
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Add `koala-lightning-700m` and `koala-lightning-1b`, two few-step text-to-image checkpoints in
the existing `stable-diffusion-xl` family and `koala` catalog architecture. The author-published
weights come from `etri-vilab/koala-lightning-700m` at revision
`358e0e3576adf2ebc563b64cfcf8a088a4a6407a` and `etri-vilab/koala-lightning-1b` at revision
`b4114f7577bb08e08a6b9792eb3a3a8b34c95adb`.

The catalog downloads the full-precision component safetensors. It excludes fp16 copies,
framework conversions, duplicate VAE files and deployment exports. Neither Hub repository
declares a weight license, so the catalog records `Unspecified`; the linked author repository
separately describes dual licensing for its software.

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
| Euler scheduler with trailing spacing | existing scheduler; catalog override applies the model-card setting |

All four published tensor layouts exactly match the corresponding base KOALA release. No shared
native module changes are required.

## Pipeline

Generation follows `StableDiffusionXLPipeline.__call__`: two penultimate CLIP hidden states,
OpenCLIP pooled conditioning, six SDXL size/crop time ids, classifier-free guidance, Euler
denoising, VAE decoding and uint8 conversion. The model cards prescribe Euler trailing timestep
spacing, 10 denoising steps and guidance 3.5; these become the catalog defaults for both models.
The published scheduler files retain leading spacing, so the backend's existing per-model
scheduler override is required.

## Memory

The published U-Nets contain about 0.78B and 1.16B parameters. Their full-precision files are about
3.1 GB and 4.6 GB; shared text encoders and VAE add about 3.6 GB. Component validation limits how
long two complete implementations coexist in memory.

## Validation

Published tensor headers cover all four weighted components for both repositories. Dedicated
tests cover strict published layouts, exact random-weight U-Net parity, exact tiny-pipeline parity
with the model defaults, trailing Euler scheduler parity, tokenizer/text-encoder/U-Net/VAE
component parity, and two real-checkpoint end-to-end runs per checkpoint through the public
pipeline API (seed 0 and seed 42). The controller runs the real-checkpoint tests.

## Out of scope

Training, LoRA, image-to-image, safety checking, watermarking, fp16 parity, framework conversions,
deployment exports and duplicate weight files are out of scope.
