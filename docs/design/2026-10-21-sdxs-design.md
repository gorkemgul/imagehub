# SDXS-512 design

- Date: 2026-10-21
- Issue: #171
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Add `sdxs-512-0.9`, the author's ungated SDXS-512 text-to-image checkpoint, to the existing
`stable-diffusion` family. The weights come from `IDKiro/sdxs-512-0.9` at revision
`c332f05f60eb4b453de513be52c2a18c48d8cfe6` under CreativeML Open RAIL++-M. The catalog
downloads the full-precision component safetensors and excludes the duplicate text-encoder
pickle, optional large VAE, demo, and generated example.

## Reference

Diffusers 0.40.0 `StableDiffusionPipeline`, pinned at
`d035dcd7cc7c88e0a154609b62887d50bba9fdc2`, and transformers 5.18.0. The published model index
also names `StableDiffusionPipeline`.

## Components

| Part | Status |
| --- | --- |
| OpenCLIP ViT-H/14 text encoder and tokenizer | existing Stable Diffusion implementation |
| Three-stage, one-layer U-Net without a mid block | existing configurable U-Net |
| Tiny distilled autoencoder | existing `AutoencoderTiny`; select it from the published config |
| Euler scheduler with trailing spacing | existing Euler implementation |

The Stable Diffusion component container needs to accept either the KL or tiny autoencoder. No
native model or scheduler changes are required.

## Pipeline

Generation follows `StableDiffusionPipeline.__call__`: CLIP prompt encoding, optional
classifier-free guidance, Euler input scaling and one-step denoising, TinyVAE decoding, and uint8
conversion. The model card requires one inference step and guidance 0, which become the catalog
defaults. The default output is 512×512 from the U-Net sample size.

## Memory

The full-precision text encoder and U-Net are about 1.36 GB and 1.31 GB; the TinyVAE is about
9.8 MB. Component-by-component validation avoids retaining two complete pipelines longer than
needed.

## Validation

Published tensor headers cover the text encoder, U-Net, and TinyVAE. Dedicated tests cover exact
random-weight U-Net and TinyVAE parity, exact tiny-pipeline parity for defaults, guidance,
non-square output, and two seeds, plus real-checkpoint header, tokenizer, component, scheduler,
and public-pipeline end-to-end/seed comparisons. The long real-checkpoint tests are left for the
controller.

## Out of scope

Training, image conditioning, safety checking, fp16 parity, the optional large VAE, the duplicate
pickle text encoder, and the DreamShaper SDXS releases are out of scope.
