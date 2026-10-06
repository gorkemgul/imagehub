# Composite checkpoints design (SDXL-Lightning)

- Date: 2026-10-06
- Issue: #29 (SDXL-Lightning); reused by #35 (Hyper-SD), #36 (DMD2) and similar releases
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Problem

Some releases publish only the part that changed and rely on another, already published model for
the rest. `ByteDance/SDXL-Lightning` publishes distilled SDXL U-Net weights
(`sdxl_lightning_{2,4,8}step_unet.safetensors`, fp16) and its model card loads the text encoders,
tokenizers, VAE and scheduler from `stabilityai/stable-diffusion-xl-base-1.0`, switching the Euler
scheduler to `timestep_spacing="trailing"`. A catalog entry today has one repository and one commit.

## Design

A catalog entry may name a **base model** (another catalog entry) and **overlay files** from its own
repository, as VoiceHub's OpenVoice names its base MeloTTS model and revision.

- `ModelSpec.base`: the catalog id of the base model (pinned through that entry's revision).
- `ModelSpec.files` keeps listing files of the entry's own repository; `ModelSpec.overlay` maps a
  snapshot path to one of those files, e.g. `{"unet/diffusion_pytorch_model.safetensors":
  "sdxl_lightning_4step_unet.safetensors"}`.
- `ModelSpec.config["scheduler"]`: published-config overrides applied when the backend builds the
  scheduler (here `{"timestep_spacing": "trailing"}`), so every downloaded file stays byte-identical
  to its source and verifiable.
- `download_pretrained` downloads the base entry's files (minus those replaced by the overlay) at
  the base revision and the overlay files at the entry's revision, verifies every file as today
  (sha256 for LFS, git blob sha1 otherwise) and writes one snapshot directory. The manifest records,
  per file, the repository, the commit and the hash; `revision` remains the entry's own commit and
  a `base` field records the base model id and commit.
- The backend loads the snapshot like any SDXL snapshot. Overlay weights in fp16 are cast to the
  requested dtype on load, as diffusers does when the model card loads them into an fp32 U-Net.

## Catalog entries

`sdxl-lightning-2step`, `sdxl-lightning-4step`, `sdxl-lightning-8step` (family
`stable-diffusion-xl`, catalog id `sdxl-lightning`, base `sdxl-base-1.0`, license `openrail++`):
defaults steps 2/4/8, guidance 0.0. The 1-step model (`*_1step_unet_x0`) predicts the clean sample
(`prediction_type="sample"`) and is a follow-up. The full single-file checkpoints and LoRAs in the
repository are not used.

## Reference

diffusers 0.40.0 exactly as the model card: `UNet2DConditionModel.from_config(base, subfolder="unet")`
in float32, `load_state_dict` of the overlay file, `StableDiffusionXLPipeline.from_pretrained(base,
unet=unet)`, `EulerDiscreteScheduler.from_config(..., timestep_spacing="trailing")`, guidance 0.

## Validation

Manifest and download tests (composite snapshot, per-file provenance, overlay verification, failure
cases); real-checkpoint U-Net single step and end to end at 1024×1024 with the default steps for each
of the three models, plus the seed test. The text encoders and VAE are the SDXL base components,
already validated; their parity is not re-measured.

## Task breakdown

1. Catalog fields, composite download and manifest, scheduler overrides in the SDXL backend; unit
   tests.
2. Catalog entries and real-checkpoint tests; parity run; docs; PR closes #29.
