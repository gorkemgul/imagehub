# Hyper-SD design

- Date: 2026-10-07
- Issue: #35
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`hyper-sdxl-1step` (`ByteDance/Hyper-SD`, file `Hyper-SDXL-1step-Unet.safetensors`, revision
`bc08d97`), the only full checkpoint in the repository, text-to-image only, at `full` validation
status, in the existing `stable-diffusion-xl` family. The repository's other files are LoRA adapters
(SD 1.5, SDXL, SD3, FLUX.1-dev) and are out of scope, as are the ComfyUI fp16 copy and fp16/bf16
parity. The model card has no license metadata and the repository's license file covers only the
FLUX-related files, so the catalog records the license as `Unspecified`.

## Reference

The model card's usage: `stabilityai/stable-diffusion-xl-base-1.0` with this U-Net loaded into it,
the scheduler replaced by `LCMScheduler.from_config(<SDXL scheduler config>)`, and
`pipe(prompt, num_inference_steps=1, guidance_scale=0, timesteps=[800])`, with diffusers 0.40.0
`StableDiffusionXLPipeline` and `LCMScheduler`.

## Components

| Part | Status |
| --- | --- |
| SDXL text encoders, tokenizers, VAE | exist (from `sdxl-base-1.0`) |
| U-Net: same layout as the SDXL base U-Net (1680 fp32 tensors) | exists; a composite `base` + `overlay` entry, as SDXL-Lightning |
| LCM scheduler built from the SDXL scheduler config, with explicit `timesteps` | scheduler exists (SD 1.x LCM); the SDXL backend and explicit timesteps are new |

## Pipeline

As `StableDiffusionXLPipeline.__call__` with the LCM scheduler and `timesteps=[800]`: no
classifier-free guidance at guidance 0, the LCM step (with generator noise between steps when there
is more than one), VAE decode. Defaults from the model card: 1 step, guidance 0.0, timesteps
`[800]`, 1024×1024. The catalog entry selects the scheduler class and default timesteps; the
`timesteps` option overrides them.

## Memory

As SDXL: about 14 GB in fp32 per pipeline; reference and native fit together.

## Validation

LCM scheduler parity with explicit timesteps (unit and reference), the header check of the overlay
U-Net, a tiny-snapshot end to end with the LCM scheduler and explicit timesteps against diffusers,
then real-checkpoint U-Net single step at 1024×1024, end to end at the defaults (latents and pixels)
and the seed test.

## Task breakdown

1. LCM explicit timesteps, LCM in the SDXL backend, the catalog entry and tests.
2. Real-checkpoint parity, docs; PR closes #35.
