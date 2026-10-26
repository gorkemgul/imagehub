# Segmind-Vega design

- Date: 2026-10-19
- Issue: #155
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Add `segmind-vega`, an ungated Apache-2.0 text-to-image checkpoint in the existing
`stable-diffusion-xl` family. The weights come from `segmind/Segmind-Vega` at revision
`7714c4363e5856ff974a4f4b068e8691f26d0b40`; the author code repository is
`segmind/SSD-1B` at `2bfc35f41f21531c61e3d3470bca907fd839d035`.

The catalog downloads the full-precision component safetensors. It excludes the fp16 copies and
the duplicate single-file checkpoint.

## Reference

Diffusers 0.40.0 `StableDiffusionXLPipeline`, pinned at
`d035dcd7cc7c88e0a154609b62887d50bba9fdc2`, and transformers 5.18.0. The published model index
also names `StableDiffusionXLPipeline`.

## Components

| Part | Status |
| --- | --- |
| CLIP ViT-L/14 text encoder and tokenizer | existing SDXL implementation; published layout matches |
| OpenCLIP bigG text encoder and tokenizer | existing SDXL implementation; published layout matches |
| SDXL U-Net with transformer depths `[1, 1, 2]` | existing configurable U-Net; new release config and parity coverage |
| f=8 KL autoencoder | existing SDXL implementation; published layout matches |
| Euler scheduler with leading spacing | existing SDXL implementation; published config matches |

No shared native module changes are required.

## Pipeline

Generation follows `StableDiffusionXLPipeline.__call__`: two penultimate CLIP hidden states,
OpenCLIP pooled conditioning, six SDXL size/crop time ids, optional classifier-free guidance,
Euler denoising, VAE decoding and uint8 conversion. Defaults remain the SDXL defaults: 50 steps,
guidance 5.0 and 1024×1024. The model card recommends negative prompting and guidance near 9.0,
but those are usage recommendations rather than pipeline defaults.

## Memory

The U-Net has about 0.74B parameters (about 3.0 GB in float32). The two text encoders and VAE add
about 3.6 GB. Component-by-component reference validation avoids retaining two complete pipelines
longer than necessary.

## Validation

Published tensor headers cover all four weighted components. Dedicated tests cover exact
random-weight U-Net parity at tiny and published layouts, exact tiny-pipeline parity with and
without guidance at a non-square size, tokenizer/text-encoder/U-Net/VAE/scheduler component parity,
and two short real-checkpoint end-to-end runs through the public pipeline API (seed 0 and seed 42).
The real runs use four steps and 64×64 output to keep controller runtime short.

## Out of scope

Training, LoRA, image-to-image, safety checking, watermarking, fp16 parity, the fp16 duplicate
files and the single-file checkpoint are out of scope.
