# Segmind small-sd and tiny-sd design

- Date: 2026-10-19
- Issue: #157
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Add `segmind-small-sd` and `segmind-tiny-sd`, two ungated CreativeML OpenRAIL-M
checkpoints in the existing `stable-diffusion` family. The author repositories are pinned to
`f8b087e68071a29bbe85a4b07e6e1e8998d59aa7` and
`cad0bd7495fa6c4bcca01b19a723dc91627fe84f`, respectively. Both map to the `bk-sdm`
architecture catalog entry.

The repositories publish component pickles rather than safetensors. They are loaded with
PyTorch's weights-only path. Small publishes float32 tensors; Tiny publishes float16 tensors and
has no full-precision alternative.

## Reference

Diffusers 0.40.0 `StableDiffusionPipeline` and transformers 5.18.0. The published model indexes
also name `StableDiffusionPipeline`.

## Components

| Part | Status |
| --- | --- |
| CLIP text encoder and tokenizer | existing SD 1.x implementation |
| Block-pruned conditional U-Net | existing configurable U-Net; Tiny adds the published no-mid-block option |
| f=8 KL autoencoder | existing implementation |
| DPM-Solver++ scheduler | existing scheduler extended to the published scaled-linear schedule |
| DEIS scheduler | new second-order scheduler for Small's published configuration |

## Pipeline

Generation follows `StableDiffusionPipeline.__call__`: CLIP prompt conditioning, optional
classifier-free guidance, the checkpoint's scheduler, VAE decoding and uint8 conversion. Defaults
remain 50 steps, guidance 7.5 and 512×512.

## Memory

Small's U-Net is about 0.58B parameters and Tiny's about 0.32B. Component-by-component reference
validation avoids holding duplicate complete pipelines longer than needed.

## Validation

Published tensor layouts cover all three weighted components for both snapshots. Tests cover the
published U-Net layouts on the meta device, exact random-weight U-Net and scheduler parity, exact
tiny-pipeline parity, real-checkpoint component parity, and separate short public-API end-to-end
runs at seeds 0 and 42 for each checkpoint.

## Out of scope

Training, LoRA, image-to-image, safety checking, watermarking, and mixed-precision parity are out
of scope.
