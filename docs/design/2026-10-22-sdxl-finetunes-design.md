# Community SDXL fine-tunes design

- Date: 2026-10-22
- Issue: #173
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Add three text-to-image checkpoints to the existing `stable-diffusion-xl` family:

- `juggernaut-xl-v9`: `RunDiffusion/Juggernaut-XL-v9` at
  `cf419233522daa0b9ea36c3aff98fa2cab1fb0fb`;
- `animagine-xl-4.0`: `cagliostrolab/animagine-xl-4.0` at
  `2b7c1b397761bf5bd3cc42e5b39ec99314a75a96`;
- `dreamshaper-xl-1.0`: `Lykon/dreamshaper-xl-1-0` at
  `41e6644752a8c9aa63930e6043c4fd83c7708420`.

The catalog downloads component checkpoints only. Duplicate single-file checkpoints are excluded.
Juggernaut publishes only float16 component tensors, Animagine publishes float16 component tensors
under unqualified filenames, and DreamShaper publishes float32 component tensors.

## Reference

Diffusers 0.40.0 `StableDiffusionXLPipeline` and transformers 5.18.0. Every repository's
`model_index.json` names `StableDiffusionXLPipeline`.

## Components

| Part | Status |
| --- | --- |
| CLIP ViT-L/14 text encoder and tokenizer | existing SDXL implementation; published layout matches |
| OpenCLIP bigG text encoder and tokenizer | existing SDXL implementation; published layout matches |
| SDXL U-Net | existing implementation; all three publish the base SDXL layout |
| f=8 KL autoencoder | existing SDXL implementation; published layouts match |
| Euler scheduler | existing implementation; used by Animagine |
| DDPM scheduler with trailing spacing | existing scheduler extended for the published Juggernaut spacing |
| second-order DEIS with leading spacing | existing scheduler extended for the published DreamShaper spacing |

No native model changes are required.

## Pipeline

Generation keeps each repository's published scheduler. The existing SDXL path supplies two CLIP
conditioning streams, pooled text conditioning, six size/crop time ids, classifier-free guidance,
VAE decoding and uint8 conversion. The public defaults remain 50 steps, guidance 5.0 and the U-Net
native 1024×1024 size.

## Memory

Each checkpoint is the full SDXL architecture. A float32 pipeline needs roughly 13 GB, while strict
reference comparison temporarily needs two pipelines. Real validation therefore compares components
and uses short 64×64 end-to-end runs, releasing reference objects between model parameters where
possible.

## Validation

Pinned fixtures record every published component tensor name and shape. Tests cover strict meta-device
layout checks, exact trailing-DDPM and leading-DEIS scheduler parity, the existing exact SDXL random
weight and tiny-pipeline coverage, real tokenizer/text-encoder/U-Net/VAE/scheduler comparisons, and
separate public-pipeline end-to-end comparisons at seed 0 and seed 42 for every checkpoint.

## Out of scope

Training, LoRA, long-prompt weighting, image-to-image, safety checking, watermarking, alternate
schedulers, fp16 arithmetic parity, duplicate single-file checkpoints and Animagine's additional
optimized single-file checkpoint are out of scope.
