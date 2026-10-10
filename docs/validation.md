# Validation — imagehub 0.1

Validated on CPU on October 4, 2026. Environment: Python 3.13.11 / PyTorch 2.14.1.
Reference: diffusers 0.40.0, transformers 5.18.0. Machine: Apple M2 Max, 96 GB unified memory,
12 CPU threads. Sources, commits and file fingerprints are in [validation.json](validation.json).
References are recorded per model in `validation.json` under `reference`.

## Scope

- **5 checkpoints, `full`:** `sd-v1-1`, `sd-v1-2`, `sd-v1-3`, `sd-v1-4`, `sd-v1-5`.
  The published tensors were loaded strictly. A 50-step generation from the same initial noise was
  compared against diffusers.
- The observed difference in every comparison is **exactly 0**. On CPU in float32, the local
  implementation produces a bit-identical image to the one diffusers produces.
- This statement is not a quality benchmark.

## Comparisons (CPU, float32)

| Comparison | Checkpoint | Largest observed difference | Tolerance |
| --- | --- | --- | --- |
| Tokenizer token ids | v1.4 (14 prompts) and v1.1–v1.5 (1,000 random prompts each) | identical | identical |
| CLIP text encoder | v1.4 | 0 | 0 |
| U-Net single step (t = 981, 501, 1) | v1.4 | 0 | 0 |
| VAE decode / encode | v1.4 | 0 / 0 | 0 |
| Initial noise from seed | — | identical | identical |
| End to end: final latent / pixels | v1.1 | 0 / 0 | 0 / 0 |
| End to end: final latent / pixels | v1.2 | 0 / 0 | 0 / 0 |
| End to end: final latent / pixels | v1.3 | 0 / 0 | 0 / 0 |
| End to end: final latent / pixels | v1.4 | 0 / 0 | 0 / 0 |
| End to end: final latent / pixels | v1.5 | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | v1.5 | 0 | 0 |

End-to-end test settings: prompt "a photograph of an astronaut riding a horse", 50 steps,
guidance 7.5, 512×512, initial noise `seed=1234`. On the diffusers side, `safety_checker=None`
and `feature_extractor=None` were used.

Tolerances are set by this rule: if the measured difference is 0, the tolerance is 0; otherwise it
is the next order of magnitude up. Since every measurement came out 0, no tolerance needed to be
loosened.

## Tensor header check

For each of the five checkpoints, the tensor names and shapes of three components (`text_encoder`,
`unet`, `vae`) were read without loading the weights into memory (header only for safetensors,
`map_location="meta"` for `.bin`) and compared with the model's `state_dict` after
`prepare_state_dict`. Across all 5 × 3 components, missing, extra and shape-mismatched tensors are
**0**. Tensor counts: 196 / 686 / 248 (after dropping `text_model.embeddings.position_ids`).
Record: `tensor_header_check`, [validation.json](validation.json).

## LCM

Validated on CPU in the same environment as above. `lcm-dreamshaper-v7` was loaded strictly from
`SimianLuo/LCM_Dreamshaper_v7`. The observed difference in every comparison is **exactly 0**: on
CPU in float32, the local implementation produces a bit-identical result to the reference. The
full run took 3 minutes.

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| CLIP text encoder | 0 | 0 |
| U-Net single step with guidance embedding (t = 999) | 0 | 0 |
| VAE decode / encode | 0 / 0 | 0 |
| LCM scheduler (4 steps, seeded noise between steps) | 0 | 0 |
| End to end: final latent / pixels (768×768, 4 steps, guidance 8.0) | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | 0 | 0 |

The end-to-end prompt was "a photograph of an astronaut riding a horse". The default run uses
768×768, 4 steps and guidance 8.0. The LCM scheduler config's `steps_offset` is accepted and
ignored because the reference never uses it.

The header-only tensor check covered `text_encoder`, `unet` and `vae`: tensor counts
196 / 687 / 248, with missing, extra and shape-mismatched tensors all **0**. Record:
`tensor_header_check`, [validation.json](validation.json).

## Stable Diffusion 2.1

`sd-v2-1` was loaded strictly from `sd2-community/stable-diffusion-2-1` (revision `bb21548`, a copy of the withdrawn
`stabilityai/stable-diffusion-2-1` release) and compared with diffusers 0.40.0 `StableDiffusionPipeline` on CPU in
float32. Every comparison is exact. The real-checkpoint run takes about 14 minutes on an M2 Max (CPU).

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| Tokenizer token ids (5 prompts) | identical | exact |
| OpenCLIP-H text encoder | 0 | 0 |
| U-Net single step (t = 981, 481, 1) | 0 | 0 |
| VAE decode / encode (768×768) | 0 | 0 |
| DDIM timesteps with leading spacing (25 and 50 steps) | 0 | 0 |
| End to end: final latent / pixels (768×768, 50 steps, guidance 7.5) | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | 0 | 0 |

The tokenizer exposed a loading rule that now applies to every CLIP tokenizer: when `tokenizer_config.json` has no
`added_tokens_decoder`, transformers merges `special_tokens_map.json` over it (this repository pads with `!`, which
only the map names), and it registers the named special tokens in the order bos, eos, unk, pad.

Without weights, a tiny random v-prediction model with the DDIM scheduler and `upcast_attention`
matches `StableDiffusionPipeline` exactly, and the DDIM scheduler matches diffusers exactly for
epsilon, sample and v-prediction trajectories at 25 and 50 steps.

## SD-Turbo

Validated on CPU in the same environment as above. `sd-turbo` was loaded strictly from
`stabilityai/sd-turbo`. The observed difference in every comparison is **exactly 0**: on CPU in
float32, the local implementation produces a bit-identical result to the reference. The full run
took 1 minute.

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| Tokenizer token ids (5 prompts) | identical | exact |
| OpenCLIP-H text encoder | 0 | 0 |
| U-Net single step (t = 999, 499) | 0 | 0 |
| VAE decode / encode | 0 / 0 | 0 |
| Euler timesteps / sigmas with trailing spacing (1, 2 and 4 steps) | 0 / 0 | 0 |
| End to end: final latent / pixels (512×512, 1 step, guidance 0.0) | 0 / 0 | 0 / 0 |
| Guided end to end: final latent / pixels (512×512, 4 steps, guidance 2.0, negative prompt) | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | 0 | 0 |

The end-to-end prompt was "a photograph of an astronaut riding a horse" with initial noise
`seed=1234`. The default run uses 512×512, 1 step and guidance 0.0; the guided run uses 4 steps,
guidance 2.0 and a negative prompt.

The header-only tensor check covered `text_encoder`, `unet` and `vae`: tensor counts
372 / 686 / 248, with missing, extra and shape-mismatched tensors all **0**. Record:
`tensor_header_check`, [validation.json](validation.json).

## SDXL

Validated on CPU on October 5, 2026, in the same environment as above. Seven checkpoints, `full`:
`sdxl-base-1.0`, `sdxl-lightning-2step`, `sdxl-lightning-4step`, `sdxl-lightning-8step`,
`sdxl-turbo`, `ssd-1b` and `playground-v2.5`. The published tensors were loaded strictly. A
generation at each model's native size was compared against diffusers from the same initial noise.

### Comparisons (CPU, float32)

| Comparison | Checkpoint | Largest observed difference | Tolerance |
| --- | --- | --- | --- |
| Tokenizer token ids (2 tokenizers, 18 prompts) | sdxl-base-1.0 | identical | identical |
| CLIP ViT-L/14 text encoder (penultimate layer) | sdxl-base-1.0 | 0 | 0 |
| OpenCLIP bigG text encoder (penultimate layer and pooled output) | sdxl-base-1.0 | 0 / 0 | 0 / 0 |
| U-Net single step (t = 999, 500, 1) | sdxl-base-1.0 | 0 | 0 |
| VAE decode / encode | sdxl-base-1.0 | 0 / 0 | 0 |
| Euler timesteps / sigmas (50 and 30 steps) | sdxl-base-1.0 | 0 / 0 | 0 |
| End to end: final latent / pixels | sdxl-base-1.0 | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | sdxl-base-1.0 | 0 | 0 |
| U-Net single step (1024×1024, t = 999) | sdxl-lightning-2step | 0 | 0 |
| End to end: final latent / pixels (1024×1024, 2 steps, guidance 0.0) | sdxl-lightning-2step | 0 / 0 | 0 / 0 |
| End to end, seed 42 with the defaults (pixels) | sdxl-lightning-2step | 0 | 0 |
| U-Net single step (1024×1024, t = 999) | sdxl-lightning-4step | 0 | 0 |
| End to end: final latent / pixels (1024×1024, 4 steps, guidance 0.0) | sdxl-lightning-4step | 0 / 0 | 0 / 0 |
| End to end, seed 42 with the defaults (pixels) | sdxl-lightning-4step | 0 | 0 |
| U-Net single step (1024×1024, t = 999) | sdxl-lightning-8step | 0 | 0 |
| End to end: final latent / pixels (1024×1024, 8 steps, guidance 0.0) | sdxl-lightning-8step | 0 / 0 | 0 / 0 |
| End to end, seed 42 with the defaults (pixels) | sdxl-lightning-8step | 0 | 0 |
| Tokenizer token ids (2 tokenizers, 18 prompts) | sdxl-turbo | identical | identical |
| CLIP ViT-L/14 text encoder (penultimate layer) | sdxl-turbo | 0 | 0 |
| OpenCLIP bigG text encoder (penultimate layer and pooled output) | sdxl-turbo | 0 / 0 | 0 / 0 |
| U-Net single step (t = 999, 500, 1) | sdxl-turbo | 0 | 0 |
| VAE decode / encode | sdxl-turbo | 0 / 0 | 0 |
| Euler ancestral timesteps / sigmas (50 and 30 steps) | sdxl-turbo | 0 / 0 | 0 |
| End to end: final latent / pixels (1 step, guidance 0.0) | sdxl-turbo | 0 / 0 | 0 / 0 |
| End to end: final latent / pixels (4 steps, guidance 2.0) | sdxl-turbo | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | sdxl-turbo | 0 | 0 |
| Tokenizer token ids (2 tokenizers, 18 prompts) | ssd-1b | identical | identical |
| CLIP ViT-L/14 text encoder (penultimate layer) | ssd-1b | 0 | 0 |
| OpenCLIP bigG text encoder (penultimate layer and pooled output) | ssd-1b | 0 / 0 | 0 / 0 |
| U-Net single step (t = 999, 500, 1) | ssd-1b | 0 | 0 |
| VAE decode / encode | ssd-1b | 0 / 0 | 0 |
| Euler timesteps / sigmas (50 and 30 steps) | ssd-1b | 0 / 0 | 0 |
| End to end: final latent / pixels | ssd-1b | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | ssd-1b | 0 | 0 |
| Tokenizer token ids (2 tokenizers, 18 prompts) | playground-v2.5 | identical | identical |
| CLIP ViT-L/14 text encoder (penultimate layer) | playground-v2.5 | 0 | 0 |
| OpenCLIP bigG text encoder (penultimate layer and pooled output) | playground-v2.5 | 0 / 0 | 0 / 0 |
| U-Net single step (t = 999, 500, 1) | playground-v2.5 | 0 | 0 |
| VAE decode / encode | playground-v2.5 | 0 / 0 | 0 |
| EDM DPM-Solver++ timesteps / sigmas (50 and 30 steps) | playground-v2.5 | 0 / 0 | 0 |
| End to end: final latent / pixels | playground-v2.5 | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | playground-v2.5 | 0 | 0 |

The observed difference in every comparison is **exactly 0**: on CPU in float32, the local
implementation produces a bit-identical image to the one diffusers produces. The same tolerance
rule applies: since every measurement came out 0, no tolerance needed to be loosened.

End-to-end test settings for `sdxl-base-1.0` and `ssd-1b`: prompt "a photograph of an astronaut
riding a horse", 1024×1024, guidance 5.0, initial noise `seed=1234` (the seed 42 run uses the same
settings). 30 steps are used instead of the default 50 to keep CPU runtime reasonable. SDXL Turbo's
native size is 512×512. Its default run uses 1 step and guidance 0.0; the guided run uses 4 steps
and guidance 2.0. Its Euler ancestral scheduler draws the per-step ancestral noise from the same
seeded CPU generator as the initial noise. `playground-v2.5` uses 1024×1024, 30 steps and guidance
3.0; 30 steps are used instead of the default 50 to keep CPU runtime reasonable.

The SDXL-Lightning reference is built exactly as described by the model card: a float32 U-Net from
the SDXL base configuration loaded with the published U-Net file, placed in the base pipeline, and
an Euler scheduler with trailing timestep spacing. The text encoders and VAE are the validated SDXL
base components. The three defaults are respectively 2, 4 and 8 steps at 1024×1024 with guidance
0.0. Each validation run took 2–4 minutes.

### Hyper-SD

`hyper-sdxl-1step` was validated on CPU in float32 from the composite snapshot: SDXL base at
revision `4621659` plus `Hyper-SDXL-1step-Unet.safetensors` from `ByteDance/Hyper-SD` at revision
`bc08d97`. Every comparison was exact.

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| LCM explicit-timestep schedules and trajectories (1 and 4 steps; unit and reference) | 0 | 0 |
| Tiny-snapshot end to end with LCM and explicit timesteps (1 and 4 steps) | 0 | 0 |
| Overlay U-Net header (1680 tensors) | 0 missing / extra / mismatched | exact |
| U-Net single step (1024×1024, t = 800) | 0 | 0 |
| End to end: final latent / pixels (1024×1024, 1 step, guidance 0.0, timesteps `[800]`) | 0 / 0 | 0 / 0 |
| End to end, seed 42 with the defaults (pixels) | 0 | 0 |

The reference pipeline loads the composite snapshot with
`StableDiffusionXLPipeline.from_pretrained`. Loading the U-Net in the model card's other documented
way—an empty model followed by `load_state_dict`—places the weights in freshly allocated memory
instead of preserving the file's memory-mapped layout. CPU summation order then differs, and one
U-Net step differs by about `1e-4`; the same effect occurs with the SDXL base U-Net. The parity
claim is therefore against `from_pretrained`. The real-checkpoint run took 2 minutes.

### DMD2

`dmd2-sdxl-4step` and `dmd2-sdxl-1step` were validated on CPU in float32 from composite snapshots:
SDXL base at revision `4621659` plus `dmd2_sdxl_4step_unet.bin` or
`dmd2_sdxl_1step_unet.bin` from `tianweiy/DMD2` at revision `be22767`. Every comparison was exact.

| Comparison | Checkpoint | Largest observed difference | Tolerance |
| --- | --- | --- | --- |
| Overlay U-Net header (1680 tensors) | dmd2-sdxl-4step | 0 missing / extra / mismatched | exact |
| U-Net single step (1024×1024, t = 999) | dmd2-sdxl-4step | 0 | 0 |
| End to end: final latent / pixels (1024×1024, 4 steps, guidance 0.0, timesteps `[999, 749, 499, 249]`) | dmd2-sdxl-4step | 0 / 0 | 0 / 0 |
| End to end, seed 42 with the defaults (pixels) | dmd2-sdxl-4step | 0 | 0 |
| Overlay U-Net header (1680 tensors) | dmd2-sdxl-1step | 0 missing / extra / mismatched | exact |
| U-Net single step (1024×1024, t = 399) | dmd2-sdxl-1step | 0 | 0 |
| End to end: final latent / pixels (1024×1024, 1 step, guidance 0.0, timesteps `[399]`) | dmd2-sdxl-1step | 0 / 0 | 0 / 0 |
| End to end, seed 42 with the defaults (pixels) | dmd2-sdxl-1step | 0 | 0 |

The end-to-end comparisons used fixed initial noise with seed 1234 and a scheduler generator seeded
with 42 for the LCM noise drawn between steps. Zip-format PyTorch pickles are memory-mapped with
`torch.load(..., weights_only=True, mmap=True)`, as in diffusers. Loading the weights into fresh
memory changed CPU summation order and made one U-Net step differ by about `6e-5`. The
real-checkpoint run for both models took 4 minutes.

For `sdxl-base-1.0`, the same header-only tensor check covered the four components
(`text_encoder`, `text_encoder_2`, `unet`, `vae`): tensor counts 196 / 517 / 1680 / 248, and
missing, extra and shape-mismatched tensors are **0**. Record: `tensor_header_check`,
[validation.json](validation.json).

For each SDXL-Lightning checkpoint, the header-only tensor check covered its distilled U-Net: 1680
tensors, with missing, extra and shape-mismatched tensors all **0**. Record:
`tensor_header_check`, [validation.json](validation.json).

For `sdxl-turbo`, the same header-only tensor check covered the four components (`text_encoder`,
`text_encoder_2`, `unet`, `vae`): tensor counts 196 / 517 / 1680 / 248, and missing, extra and
shape-mismatched tensors are **0**. Record: `tensor_header_check`,
[validation.json](validation.json).

For `ssd-1b`, the same header-only tensor check covered the four components (`text_encoder`,
`text_encoder_2`, `unet`, `vae`): tensor counts 196 / 517 / 944 / 248, and missing, extra and
shape-mismatched tensors are **0**. Record: `tensor_header_check`,
[validation.json](validation.json).

For `playground-v2.5`, the same header-only tensor check covered the four components
(`text_encoder`, `text_encoder_2`, `unet`, `vae`): tensor counts 196 / 517 / 1680 / 248, and
missing, extra and shape-mismatched tensors are **0**. Record: `tensor_header_check`,
[validation.json](validation.json).

## Z-Image

Validated on CPU on October 5, 2026, in the same environment as above. `z-image-turbo` and `z-image`
were loaded strictly from `Tongyi-MAI/Z-Image-Turbo` and `Tongyi-MAI/Z-Image`, respectively. The
observed difference in every comparison is **exactly 0**: on CPU in float32, the local implementation
produces a bit-identical result to the reference.

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| Tokenizer token ids and attention masks (3 fixed prompts, seeded fuzz, exhaustive per-code-point check) | identical | exact |
| `z-image` tokenizer token ids and attention masks (3 fixed prompts) | identical | exact |
| Qwen3 text encoder (`hidden_states[-2]`) | 0 | 0 |
| `z-image` Qwen3 text encoder (`hidden_states[-2]`) | 0 | 0 |
| Transformer single step (1024×1024, t = 0.5 and 0.25) | 0 | 0 |
| `z-image` transformer single step (1024×1024, t = 0.5 and 0.25) | 0 | 0 |
| VAE decode / encode | 0 / 0 | 0 |
| `z-image` VAE decode / encode | 0 / 0 | 0 |
| Flow-match scheduler timesteps / sigmas | 0 / 0 | 0 |
| `z-image` flow-match scheduler timesteps / sigmas (shift 6.0) | 0 / 0 | 0 |
| End to end: final latent / pixels (1024×1024, 9 steps, guidance 0.0) | 0 / 0 | 0 / 0 |
| `z-image` end to end: final latent / pixels (1024×1024, 8 steps, guidance 4.0) | 0 / 0 | 0 / 0 |
| Guided end to end: final latent / pixels (1024×1024, 4 steps, guidance 4.0, negative prompt) | 0 / 0 | 0 / 0 |
| End to end, seed 42 (1024×1024, 9 steps, guidance 0.0; pixels) | 0 | 0 |
| `z-image` end to end, seed 42 (1024×1024, 8 steps, guidance 4.0; pixels) | 0 | 0 |

The end-to-end prompt was "a photograph of an astronaut riding a horse" with initial noise
`seed=1234`; the guided run uses a negative prompt. The tokenizer check covers fixed prompts, seeded
fuzz and every Unicode code point. The reference tokenizer's Unicode tables are older than Python's
for 108 combining marks, which it treats as unassigned.

The `z-image` end-to-end checks use 8 steps instead of the default 50 only to shorten the CPU run.

To fit the 96 GB machine, each component pair (ours and the reference) is loaded and freed one at a
time, and in the end-to-end checks the reference pipeline runs first and is freed before ours runs.

For each checkpoint, the header-only tensor check covered `text_encoder`, `transformer` and `vae`:
tensor counts 398 / 521 / 244, with missing, extra and shape-mismatched tensors all **0**. Record:
`tensor_header_check`, [validation.json](validation.json).

The full validation run took 47 minutes.

## FLUX.1

Validated on CPU in the same environment as above. `flux1-schnell`, `flux1-dev` and `flux1-krea-dev`
were loaded strictly from `black-forest-labs/FLUX.1-schnell` at revision `741f7c3`,
`black-forest-labs/FLUX.1-dev` at revision `3de623f` and `black-forest-labs/FLUX.1-Krea-dev` at
revision `8162a9c`, respectively. `flux1-krea-dev` shares the `flux1-dev` architecture and configs and is
checked with the same comparisons below (guidance 4.5, 512 tokens). The observed difference in
every comparison is **exactly 0**: on CPU in float32, the local implementation produces a
bit-identical result to the reference.

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| `flux1-schnell` CLIP tokenizer / pooled CLIP-L | identical / 0 | exact / 0 |
| `flux1-schnell` T5 tokenizer / T5-XXL encoder (256 tokens) | identical / 0 | exact / 0 |
| `flux1-schnell` transformer single step (1024×1024) | 0 | 0 |
| `flux1-schnell` VAE decode / encode | 0 / 0 | 0 / 0 |
| `flux1-schnell` flow-match Euler scheduler (static shift 1.0) | exact | exact |
| `flux1-schnell` end to end: final latent / pixels (1024×1024, 4 steps, guidance 0.0) | 0 / 0 | 0 / 0 |
| `flux1-schnell` end to end, seed 42 (pixels) | 0 | 0 |
| `flux1-dev` CLIP tokenizer / pooled CLIP-L | identical / 0 | exact / 0 |
| `flux1-dev` T5 tokenizer / T5-XXL encoder (512 tokens) | identical / 0 | exact / 0 |
| `flux1-dev` transformer single step (1024×1024, guidance-distilled) | 0 | 0 |
| `flux1-dev` VAE decode / encode | 0 / 0 | 0 / 0 |
| `flux1-dev` flow-match Euler scheduler (dynamic shifting) | exact | exact |
| `flux1-dev` end to end: final latent / pixels (1024×1024, 4 steps, guidance 3.5) | 0 / 0 | 0 / 0 |
| `flux1-dev` end to end, seed 42 (pixels) | 0 | 0 |
| `flux1-krea-dev` CLIP tokenizer / pooled CLIP-L | identical / 0 | exact / 0 |
| `flux1-krea-dev` T5 tokenizer / T5-XXL encoder (512 tokens) | identical / 0 | exact / 0 |
| `flux1-krea-dev` transformer single step (1024×1024, guidance-distilled) | 0 | 0 |
| `flux1-krea-dev` VAE decode / encode | 0 / 0 | 0 / 0 |
| `flux1-krea-dev` flow-match Euler scheduler (dynamic shifting) | exact | exact |
| `flux1-krea-dev` end to end: final latent / pixels (1024×1024, 4 steps, guidance 4.5) | 0 / 0 | 0 / 0 |
| `flux1-krea-dev` end to end, seed 42 (pixels) | 0 | 0 |

The end-to-end prompt was "a photograph of an astronaut riding a horse" with initial noise
`seed=42`. Schnell was tested at its 4-step, guidance 0.0 defaults. Dev used 4 steps and guidance
3.5 instead of its 50-step default only to shorten the CPU run, and Krea [dev] likewise used 4 steps
and guidance 4.5.

The header-only tensor check covered `text_encoder`, `text_encoder_2`, `transformer` and `vae`:
`flux1-schnell` had 196 / 219 / 1156 / 244 tensors, and `flux1-dev` and `flux1-krea-dev` each had
196 / 219 / 1160 / 244 tensors, with missing, extra and shape-mismatched tensors all **0**. Record:
`tensor_header_check`, [validation.json](validation.json).

The reference and native transformers, about 48 GB each in float32, were never loaded together.
Each test ran the reference first, saved its outputs, freed it, then ran the native model. Schnell's
component and seed tests took 11 minutes and its fixed-latent end-to-end test took 13 minutes. The
full dev run took 25 minutes. The Krea [dev] real-checkpoint run took about 30 minutes on an M2 Max (CPU).

## FLUX.1 Kontext

Validated on CPU in the same environment as above, against diffusers 0.40.0 `FluxKontextPipeline`.
`flux1-kontext-dev` was loaded strictly from `black-forest-labs/FLUX.1-Kontext-dev` at revision
`24e9ded`. The observed difference in every comparison is **exactly 0**: on CPU in float32, the
local implementation produces a bit-identical result to the reference, with identical bytes for
images.

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| Input preprocessing over several sizes (300×200, 512×768, 1024×1024, 1365×768, 2000×1100) and modes (RGB, L, RGBA), with and without the automatic resolution choice: resized image bytes and normalized tensor | 0 | 0 |
| VAE encode of the preprocessed input (same sizes and modes) | 0 | 0 |
| Tiny end to end, with an input image (square and non-square, several seeds, a rescaled request): final latent / pixels | 0 / 0 | 0 / 0 |
| Tiny end to end, without an input image: final latent / pixels | 0 / 0 | 0 / 0 |
| Tiny end to end, with true CFG and a negative prompt: final latent / pixels | 0 / 0 | 0 / 0 |
| Input preprocessing and VAE encode, real checkpoint (1024×768 and 640×960 inputs) | 0 | 0 |
| Transformer single step with the image tokens, real checkpoint (two timesteps) | 0 | 0 |
| End to end with an input image at the defaults: final latent / pixels (1024×768 input, 28 steps, guidance 3.5, seed 0) | 0 / 0 | 0 / 0 |
| End to end without an input image: final latent / pixels (4 steps) | 0 / 0 | 0 / 0 |
| End to end, seed 1234, non-square input image (640×960) and non-square output (1360×768, 4 steps) (pixels) | 0 | 0 |

The tiny end-to-end tests use random weights (2 steps) and compare the sequence layout (image tokens, their
position ids and the slicing of the output) against `FluxKontextPipeline`. The real-checkpoint
runs use fixed synthetic input images; the run with an image uses the 28-step, guidance 3.5
defaults, while the run without an image and the seed test use 4 steps to shorten the CPU run. Existing FLUX.1 tests are unchanged and still exact.

The header-only tensor check covered `text_encoder`, `text_encoder_2`, `transformer` and `vae`:
tensor counts 196 / 219 / 1160 / 244, with missing, extra and shape-mismatched tensors all **0**.
Record: `tensor_header_check`, [validation.json](validation.json).

The reference and native transformers were never held in one process (the transformer comparison
saves the reference output first, then runs the native model). The real-checkpoint run took about
2 hours 7 minutes on an Apple M2 Max CPU.

## FLUX.2 klein

Validated on CPU in the same environment as above. `flux2-klein-4b` and `flux2-klein-9b` were loaded
strictly from `black-forest-labs/FLUX.2-klein-4B` and `black-forest-labs/FLUX.2-klein-9B`,
respectively. The observed difference in every comparison is **exactly 0**: on CPU in float32, the
local implementation produces a bit-identical result to the reference.

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| Prompt string / token ids / attention masks | identical | exact |
| Qwen3-4B text encoder (layers 9, 18 and 27 stacked) | 0 | 0 |
| FLUX.2 transformer single step | 0 | 0 |
| VAE decode / encode | 0 / 0 | 0 |
| Flow-match Euler scheduler with empirical shift | exact | exact |
| End to end: final latent / pixels (1024×1024, 4 steps, guidance 1.0) | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | 0 | 0 |
| `flux2-klein-9b` prompt string / token ids / attention masks | identical | exact |
| `flux2-klein-9b` Qwen3-8B text encoder (layers 9, 18 and 27 stacked) | 0 | 0 |
| `flux2-klein-9b` FLUX.2 transformer single step | 0 | 0 |
| `flux2-klein-9b` VAE decode / encode | 0 / 0 | 0 / 0 |
| `flux2-klein-9b` flow-match Euler scheduler with empirical shift | exact | exact |
| `flux2-klein-9b` end to end: final latent / pixels (1024×1024, 4 steps, guidance 1.0) | 0 / 0 | 0 / 0 |
| `flux2-klein-9b` end to end, seed 42 (pixels) | 0 | 0 |

The end-to-end prompt was "a photograph of an astronaut riding a horse" with initial noise
`seed=1234`.

The header-only tensor check covered `text_encoder`, `transformer` and `vae`: `flux2-klein-4b` had
398 / 169 / 251 tensors and `flux2-klein-9b` had 399 / 233 / 251 tensors, with missing, extra and
shape-mismatched tensors all **0**. Record: `tensor_header_check`,
[validation.json](validation.json).

For the 9B run, one reference/native component pair was loaded at a time; peak memory use was about
65 GB. The 4B validation run took 12 minutes and the 9B validation run took 29 minutes.

## HunyuanDiT

Validated on CPU in the same environment as above. `hunyuan-dit-v1.1-distilled` was loaded strictly
from `Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled` at revision `527cf2e`. The observed
difference in every comparison is **exactly 0**: on CPU in float32, the local implementation
produces a bit-identical result to the reference.

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| BERT and mT5 tokenizers: fixed prompts and 2500 seeded fuzz strings, ids / masks at 77 / 256 tokens | identical | exact |
| BERT-large encoder (3 prompts, including Chinese and the empty negative prompt) | 0 | 0 |
| mT5-XL encoder (3 prompts) | 0 | 0 |
| HunyuanDiT transformer single step (1024×1024, batch 2, timestep 500) | 0 | 0 |
| VAE decode / encode | 0 / 0 | 0 / 0 |
| DDPM timesteps (25 / 50 steps) | exact / exact | exact / exact |
| DDPM 25-step trajectory with generator noise (v-prediction / epsilon) | 0 / 0 | 0 / 0 |
| End to end: final latent / pixels (1024×1024, 25 steps, guidance 5.0) | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | 0 | 0 |

The end-to-end prompt was "a photograph of an astronaut riding a horse" with fixed initial noise
`seed=7` and scheduler generator `seed=42`. It used the 25-step, guidance 5.0 defaults. The seed
end-to-end check used `seed=42`.

The header-only tensor check covered `text_encoder`, `text_encoder_2`, `transformer` and `vae`:
tensor counts 391 / 219 / 1545 / 248, with missing, extra and shape-mismatched tensors all **0**.
Record: `tensor_header_check`, [validation.json](validation.json).

The reference tokenizers require `sentencepiece` and `protobuf`, which are included in the
`reference` extra; imagehub reads `spiece.model` natively at runtime. The reference and native
components were loaded together: in float32 the transformer used about 6.1 GB, mT5-XL about 6.7 GB
and BERT about 1.4 GB. The end-to-end reference test file took 52 minutes.

## Kandinsky 2.2

Validated on CPU in the same environment as above. `kandinsky-2.2` was loaded strictly from the
decoder repository `kandinsky-community/kandinsky-2-2-decoder` at revision `9ae140d` and the prior
repository `kandinsky-community/kandinsky-2-2-prior` at revision `9fc51ad`. The prior is stored in
the snapshot's `prior/` subfolder. The observed difference in every comparison is **exactly 0**: on
CPU in float32, the local implementation produces a bit-identical result to the reference.

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| Token ids | identical | exact |
| CLIP text embeddings | 0 | 0 |
| CLIP vision zero image embedding | 0 | 0 |
| PriorTransformer single step | 0 | 0 |
| Prior output after 25 steps | 0 | 0 |
| Decoder U-Net single step (512×512) | 0 | 0 |
| MoVQ decode | 0 | 0 |
| End to end at the defaults (pixels) | 0 | 0 |
| End to end, seed 42 (pixels) | 0 | 0 |

The defaults are 25 prior steps with guidance 4.0, followed by 100 decoder steps with guidance 4.0
at 512×512. Random-weight tests also established exact parity for the CLIP vision model,
PriorTransformer, image-conditioned U-Net, MoVQ, UnCLIP scheduler and the Kandinsky DDPM options.
A tiny end-to-end test matched `KandinskyV22CombinedPipeline` exactly.

The header-only tensor check covered `movq`, `prior/image_encoder`, `prior/prior`,
`prior/text_encoder` and `unet`: tensor counts 431 / 777 / 338 / 518 / 724, with missing, extra and
shape-mismatched tensors all **0**. Record: `tensor_header_check`,
[validation.json](validation.json).

The reference and native models were loaded together and used about 16 GB in float32. The
real-checkpoint run took about 13 minutes.

## Stable Cascade

Validated on CPU in the same environment as above. `stable-cascade` and `stable-cascade-lite` were
loaded strictly from the decoder repository `stabilityai/stable-cascade` at revision `a89f66d` and
the prior repository `stabilityai/stable-cascade-prior` at revision `7ca32c2`. The prior is stored
in the snapshot's `prior/` subfolder. The reference is diffusers 0.40.0
`StableCascadeCombinedPipeline`. The observed difference in every comparison is **exactly 0**: on
CPU in float32, the local implementation produces a bit-identical result to the reference.

### Comparisons (CPU, float32, both weight sets)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| Masked CLIP text embeddings, pooled embeddings and negative branch | 0 | 0 |
| Stage C U-Net single step (ratio 1 and 0.5) | 0 | 0 |
| Stage C latents after 60 steps | 0 | 0 |
| Stage B U-Net single step (ratio 1 and 0.5) | 0 | 0 |
| Stage B latents after 12 steps | 0 | 0 |
| Stage A decode | 0 | 0 |
| End to end at the defaults (pixels) | 0 | 0 |
| End to end, second seed (pixels) | 0 | 0 |

The defaults are 60 prior steps with guidance 4.0, followed by 12 decoder steps with guidance 0.0 at
512×512. Random-weight tests also established exact parity for both U-Net configurations (tiny and
with the published layer layout, including odd latent sizes), the DDPM Wuerstchen scheduler with
generator noise and several `scaler` values, Paella decoding and encoding, and the CLIP text model
with an attention mask (GELU and QuickGELU, all-ones and padded masks). A tiny end-to-end test
matched `StableCascadeCombinedPipeline` exactly, including a non-square size and decoder guidance.

The header-only tensor check covered the stage C U-Net, the stage B U-Net, the text encoder and stage
A. For `stable-cascade-lite`: `prior/prior_lite` 782, `decoder_lite` 970, `prior/text_encoder` 517
and `vqgan` 122 tensors. For `stable-cascade`: `prior/prior` 1550 and `decoder` 1726, with the same
text encoder and stage A. Missing, extra and shape-mismatched tensors were all **0**. Record:
`tensor_header_check`, [validation.json](validation.json).

The stages were compared one at a time so that the reference and native models were never all
resident together.

## Würstchen v2

`wuerstchen-v2` is loaded strictly from the decoder repository `warp-ai/wuerstchen` at revision
`c3da414` and the prior repository `warp-ai/wuerstchen-prior` at revision `c41be63`; the prior is
stored in the snapshot's `prior/` subfolder. The reference is diffusers 0.40.0
`WuerstchenCombinedPipeline`. The real-checkpoint tests (`tests/reference/test_ref_wuerstchen.py`)
assert the tolerances below. All 7 real-checkpoint tests passed with exactly 0 difference in every
comparison (see the table), and the run took about 1 h 2 min on an M2 Max (CPU). The random-weight
and tiny-snapshot comparisons need no downloaded weights and also pass with exactly 0 difference.

### Comparisons (CPU, float32)

| Comparison | Tolerance | Measured |
| --- | --- | --- |
| Tensor header check: both text encoders, prior, decoder, VQGAN (names and shapes) | 0 missing, extra or mismatched | 0 missing, extra or mismatched (388 and 517 tensors in the text encoders, 648 prior, 990 decoder, 122 VQGAN) |
| Masked CLIP text states of the prior (ViT-bigG) and of the decoder (ViT-H), prompt and negative | 0 | exactly 0 (max abs and max rel) |
| Prior single step (batch 2, three timestep ratios, real text states) | 0 | exactly 0 at every ratio |
| Decoder single step (two timestep ratios, real text states) | 0 | exactly 0 at both ratios |
| VQGAN decode | 0 | exactly 0 |
| End to end: prior latents / decoder latents / pixels (512×512, 60 and 12 steps, seed 0) | 0 / 0 / 0 | 0 / 0 / 0 |
| End to end, seed 1234 (pixels) | 0 | 0 |

The tensor layout of the published files was also checked against the recorded headers in
`tests/fixtures/wuerstchen_configs/`: 648 tensors for the prior, 990 for the decoder and 122 for
the VQGAN, with no missing, extra or mismatched names or shapes.

Random-weight parity covers the prior and the decoder with tiny configurations and with the
published configurations at small latent sizes, including the optional `x_cat` input and the split
output of the decoder. Tiny-snapshot end-to-end parity against `WuerstchenCombinedPipeline` covers
the defaults, decoder guidance with a negative prompt, decoder guidance with the prior's guidance
switched off, a non-square size, a second seed, and a prompt longer than the 77 tokens the text
encoders accept; it compares the prior's image embeddings, the decoder's latents and the final
bytes.

## PixArt

Validated on CPU in the same environment as above. `pixart-sigma-xl-1024` and
`pixart-alpha-xl-1024` were loaded strictly from `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` at
revision `e102b35` and `PixArt-alpha/PixArt-XL-2-1024-MS` at revision `b89adad`, respectively. The
observed difference in every comparison is **exactly 0**: on CPU in float32, the local
implementation produces a bit-identical result to the reference.

### Comparisons (CPU, float32)

| Comparison | Model | Largest observed difference | Tolerance |
| --- | --- | --- | --- |
| T5 tokenizer ids / attention masks (300 tokens) | `pixart-sigma-xl-1024` | identical / identical | exact / exact |
| T5-v1.1-XXL encoder | `pixart-sigma-xl-1024` | 0 | 0 |
| Transformer single step (1024×1024, batch 2) | `pixart-sigma-xl-1024` | 0 | 0 |
| VAE decode / encode | `pixart-sigma-xl-1024` | 0 / 0 | 0 / 0 |
| DPM-Solver++ timesteps (20 steps) / trajectories | `pixart-sigma-xl-1024` | exact / 0 | exact / 0 |
| End to end: final latent / pixels (1024×1024, 20 steps, guidance 4.5) | `pixart-sigma-xl-1024` | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | `pixart-sigma-xl-1024` | 0 | 0 |
| T5 tokenizer ids / attention masks (120 tokens) | `pixart-alpha-xl-1024` | identical / identical | exact / exact |
| T5-v1.1-XXL encoder | `pixart-alpha-xl-1024` | 0 | 0 |
| Transformer single step (1024×1024, batch 2) | `pixart-alpha-xl-1024` | 0 | 0 |
| VAE decode / encode | `pixart-alpha-xl-1024` | 0 / 0 | 0 / 0 |
| DPM-Solver++ timesteps (20 steps) / trajectories | `pixart-alpha-xl-1024` | exact / 0 | exact / 0 |
| End to end: final latent / pixels (1024×1024, 20 steps, guidance 4.5) | `pixart-alpha-xl-1024` | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | `pixart-alpha-xl-1024` | 0 | 0 |

The end-to-end prompt was "a photograph of an astronaut riding a horse" with fixed initial noise
`seed=7` and scheduler generator `seed=42`. It used the 20-step, guidance 4.5 defaults. The seed
end-to-end check used `seed=42`.

Random-weight parity also covered Sigma- and Alpha-style tiny and medium transformer
configurations, and tiny-snapshot end-to-end parity covered both the PixArt Sigma and PixArt Alpha
pipelines. Every comparison was exact.

The header-only tensor check covered `text_encoder`, `transformer` and `vae`: tensor counts were
219 / 603 / 248 for Sigma and 219 / 612 / 248 for Alpha, with missing, extra and shape-mismatched
tensors all **0**. Record: `tensor_header_check`, [validation.json](validation.json).

Caption cleaning requires `bs4` and `ftfy` and is not implemented. Parity was measured with
`clean_caption=False`, which is what diffusers uses when those packages are unavailable. The
reference and native components were loaded together; T5-XXL used about 19 GB in float32. The
real-checkpoint run for both models (19 tests) took 45 minutes.

## Krea 2

Validated on CPU in the same environment as above. `krea2-turbo` was loaded strictly from
`krea/Krea-2-Turbo` at revision `98e0fe1`. The weights are published in bfloat16 and were upcast
to float32. The observed difference in every comparison is **exactly 0**: on CPU in float32, the
local implementation produces a bit-identical result to the reference.

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| Tensor header names and shapes | identical | exact |
| Prompt template token ids / attention masks | identical / identical | exact / exact |
| Qwen3-VL text encoder (12 selected hidden layers, stacked) | 0 | 0 |
| Transformer single step (1024×1024) | 0 | 0 |
| VAE decode / encode | 0 / 0 | 0 / 0 |
| Flow-match Euler scheduler with exponential dynamic shifting, including `mu` | exact | exact |
| End to end: final latent / pixels (1024×1024, 8 steps, guidance 0.0) | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | 0 | 0 |

The end-to-end prompt was "a photograph of an astronaut riding a horse". The fixed-noise and
seed checks used the 8-step, guidance 0.0, 1024×1024 defaults.

The header-only tensor check covered `text_encoder`, `transformer` and `vae`: tensor counts were
713 / 430 / 194, with missing, extra and shape-mismatched tensors all **0**. Record:
`tensor_header_check`, [validation.json](validation.json).

The reference pipeline calls the text encoder with transformers' default key-value cache. Its
contiguous key/value copies and masked grouped-query attention on hosts with MPS set the SDPA
summation order. The transformer's float32 tensors remain at their file offsets as diffusers loads
them (`clone_shards = False`). Latents are de-normalized by dividing by the reciprocal standard
deviation, matching the reference. The reference transformer was loaded without
`_keep_in_fp32_modules` because that path requires `accelerate`; all components ran in float32,
and the transformer was cast to float32 after loading.

The roughly 13B transformer uses about 52 GB in float32, so the reference and native transformers
were never loaded together. Each comparison ran the reference first, saved its outputs, freed it,
then ran the native model. The full real-checkpoint run on the Apple M2 Max took about 1 hour.

## LongCat-Image

Validated on CPU in the same environment as above. `longcat-image` was loaded strictly from
`meituan-longcat/LongCat-Image` at revision `d2ea50b`. The published bfloat16 weights were upcast to
float32. The observed difference in every comparison is **exactly 0**: on CPU in float32, the local
implementation produces a bit-identical result to the reference.

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| Rewritten prompt string and token ids | identical | exact |
| Qwen2.5-VL text encoder | 0 | 0 |
| LongCat-Image transformer single step (1024×1024) | 0 | 0 |
| VAE decode / encode | 0 / 0 | 0 / 0 |
| Flow-match Euler scheduler with dynamic shifting, including `mu` | exact | exact |
| End to end: final latent / pixels (1024×1024, 8 steps, guidance 4.5) | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | 0 | 0 |

The end-to-end run used 8 steps instead of the 50-step default and fixed initial noise. Prompt
rewrite and CFG renormalization were enabled. The seed end-to-end check used `seed=42`.

Random-weight parity covered the Qwen2.5-VL text model, including per-step generation logits, and
the transformer. Tiny-snapshot end-to-end parity covered configurations with and without prompt
rewrite and CFG renormalization. Every comparison was exact.

The header-only tensor check covered `text_encoder`, `transformer` and `vae`: tensor counts were
729 / 612 / 244, with missing, extra and shape-mismatched tensors all **0**. Record:
`tensor_header_check`, [validation.json](validation.json).

Reference-matching details: rewrite generation uses a key-value cache with one new token per step,
as transformers' `generate` does; recomputing the whole sequence changed logits and flipped tokens.
The Qwen2-VL chat template adds a default system turn. Published Qwen2.5-VL checkpoints use
transformers 4.x `model.layers.*` key names and the legacy `"type": "mrope"`, which are read as
transformers 5 does. The text encoder uses the default key-value cache, with contiguous keys and
values and the masked GQA rule on MPS hosts.

In float32 the text encoder used about 33 GB and the transformer about 25 GB. End-to-end validation
used the save-then-compare sequence. The real-checkpoint runs took about 26, 25 and 47 minutes.

## CogView4

Validated on CPU in the same environment as above. `cogview4-6b` was loaded strictly from
`zai-org/CogView4-6B` at revision `63a52b7`. The published bfloat16 weights were upcast to float32.
The observed difference in every comparison is **exactly 0**: on CPU in float32, the local
implementation produces a bit-identical result to the reference.

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| GLM tokenizer: 8 fixed prompts and 2000 seeded fuzz strings | identical | exact |
| GLM token ids | identical | exact |
| GLM-4-9B text encoder, second-to-last hidden state | 0 | 0 |
| CogView4 transformer single step (1024×1024) | 0 | 0 |
| VAE decode / encode | 0 / 0 | 0 / 0 |
| Flow-match Euler scheduler with linear dynamic shifting, including `mu` | exact | exact |
| End to end: final latent / pixels (1024×1024, 8 steps, guidance 5.0) | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | 0 | 0 |

The end-to-end prompt was "a photograph of an astronaut riding a horse" with fixed initial noise.
It used 8 steps instead of the 50-step default to shorten the CPU run, with guidance 5.0. The seed
end-to-end check used `seed=42`.

Random-weight parity also covered the GLM text model and CogView4 transformer, and tiny-snapshot
end-to-end parity covered the complete pipeline. Every comparison was exact. The reference text
encoder was called exactly as the pipeline calls it: with the default key-value cache, contiguous
keys and values, and the masked grouped-query attention rule.

The header-only tensor check covered `text_encoder`, `transformer` and `vae`: tensor counts were
442 / 408 / 288, with missing, extra and shape-mismatched tensors all **0**. Record:
`tensor_header_check`, [validation.json](validation.json).

The text encoder used about 36 GB and the transformer about 25 GB in float32. End-to-end tests used
the save-then-compare sequence: the reference pipeline ran first, saved its outputs and was freed
before the native pipeline ran. The real-checkpoint component and end-to-end runs took about 31 and
53 minutes, respectively.

## Ideogram 4

`ideogram4-fp8` was loaded strictly from `ideogram-ai/ideogram-4-fp8` at revision `ee79a72` and
validated on CPU in float32. The reference is the official `ideogram-oss/ideogram4` implementation
at commit `990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2`, not diffusers: the model card lists no diffusers
support for the fp8 release. The official reference ran with PyTorch 2.14.1 and transformers 5.18.0
in its own Python 3.13 environment. Every comparison against that official implementation is
**exactly 0**.

The official reference environment was created with:

```bash
uv venv -p 3.13 .venv-ref-ideogram4
uv pip install --python .venv-ref-ideogram4 git+https://github.com/ideogram-oss/ideogram4@990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2 "transformers==5.18.0" pytest numpy
```

The official-reference tests run with:

```bash
IMAGEHUB_REFERENCE_TESTS=1 PYTHONPATH=$PWD .venv-ref-ideogram4/bin/python -m pytest tests/reference/test_ref_ideogram4.py tests/reference/test_ref_tiny_ideogram4.py
```

### Comparisons with the official implementation (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| Published headers: fp8 tensors and per-row scales for every quantized component | identical | exact |
| Prompt token ids | identical | exact |
| Qwen3-VL text features (53,248 wide from stacked hidden states) | 0 | 0 |
| Conditional transformer single step (1024×1024) | 0 | 0 |
| Unconditional transformer single step (1024×1024) | 0 | 0 |
| Autoencoder decode / encode | 0 / 0 | 0 / 0 |
| Official sampler intervals and per-step guidance schedule | identical | exact |
| End to end, seed 0 (`V4_TURBO_12`, 1024×1024 pixels) | 0 | 0 |
| End to end, seed 42 (`V4_TURBO_12`, 1024×1024 pixels) | 0 | 0 |

The published header check covered `text_encoder`, `transformer`, `unconditional_transformer` and
`vae`: 1117 / 669 / 669 / 251 tensors. The three quantized components contained 368 / 211 / 211
float8 e4m3 weights and the same number of float32 scales; the VAE contained neither. imagehub
keeps the published fp8 weights and dequantizes each forward pass as
`weight.to(dtype) * weight_scale.to(dtype).unsqueeze(1)`, exactly like the official implementation.
The float32 result is bit-identical. Record: `tensor_header_check`,
[validation.json](validation.json).

Random-weight parity covered fp8 linears and the transformer, and a tiny end-to-end test covered
the complete pipeline against the official implementation. Every comparison was exact. The
real-checkpoint end-to-end comparisons used the official `V4_TURBO_12` preset rather than the
48-step `V4_QUALITY_48` default to shorten the CPU run.

The official pipeline peaks near 52 GB while loading in float32 and the native pipeline near 40 GB.
The official reference therefore ran in a child process with the same thread count, saved its
outputs and exited before the native outputs were compared. The real-checkpoint run took about two
hours.

## Qwen-Image 2.1

The reference is the implementation the Qwen team contributed to diffusers. No released diffusers
version contains it yet: 0.40.0 predates it. The reference was pinned to commit
`7997e4eabb1f880fa5697e2b24de09a892fffad5` in a separate environment with the same PyTorch 2.14.1
and transformers 5.18.0. The comparison will be re-run when a diffusers release includes the model.

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| Prompt token ids (2 prompts) | identical | exact |
| Qwen3-VL text encoder (last decoder layer before the final norm) | 0 | 0 |
| Transformer cached-flow calls (1024×1024) | 0 | 0 |
| VAE decode / encode | 0 / 0 | 0 |
| Flow-match Euler scheduler with dynamic shifting | 0 | 0 |
| End to end: final latent / pixels (1024×1024, 8 steps, `true_cfg` 1.0) | 0 / 0 | 0 / 0 |
| Guided end to end: final latent / pixels (1024×1024, 4 steps, `true_cfg` 4.0, negative prompt) | 0 / 0 | 0 / 0 |
| End to end, seed 42 (1024×1024, 8 steps, `true_cfg` 1.0; pixels) | 0 | 0 |

The end-to-end prompt was "a photograph of an astronaut riding a horse" with initial noise
`seed=1234`; the guided run used the negative prompt "blurry". The model card uses 40 steps at
2048×2048. The reduced step counts and 1024×1024 size used here only shorten the CPU run.

The header-only tensor check covered `text_encoder`, `transformer` and `vae`: tensor counts
750 / 297 / 238, with missing, extra and shape-mismatched tensors all **0**. Understanding-only
text-encoder tensors were skipped by an explicit allow-list. Record: `tensor_header_check`,
[validation.json](validation.json).

To fit the 96 GB machine, each native/reference component pair was loaded and freed one at a time.
The full validation run took 42 minutes.

## Janus-Pro

Validated on CPU on October 5, 2026, in the same environment as above. `janus-pro-1b` and
`janus-pro-7b` were loaded strictly and compared with transformers 5.18.0
`JanusForConditionalGeneration.generate(generation_mode="image")`. The observed difference in
every numeric comparison is **exactly 0**: on CPU in float32, the local implementation produces
bit-identical image tokens and pixels.

### Comparisons (CPU, float32)

| Comparison | Checkpoint | Largest observed difference | Tolerance |
| --- | --- | --- | --- |
| DeepSeek LLM tokenizer (fixed prompts, seeded fuzz and every code point; processor prompt ids) | janus-pro-1b | identical | exact |
| Language model (prompt prefill and 8 decode steps with the static cache) | janus-pro-1b | 0 | 0 |
| LlamaGen VQ-16 decode | janus-pro-1b | 0 | 0 |
| End to end: 576 image tokens / pixels (1 image, seeded sampling, guidance 5) | janus-pro-1b | identical / identical | exact / exact |
| End to end: 576 image tokens / pixels (batch of 2, seeded sampling, guidance 5) | janus-pro-1b | identical / identical | exact / exact |
| End to end, seed 42 (pixels) | janus-pro-1b | identical | exact |
| DeepSeek LLM tokenizer (13 fixed prompts + 3,000 fuzz + every code point: 3,336,096 strings; 0 mismatches) | janus-pro-7b | identical | exact |
| Language model (prompt prefill and 8 decode steps with the static cache) | janus-pro-7b | 0 | 0 |
| LlamaGen VQ-16 decode | janus-pro-7b | 0 | 0 |
| End to end: 576 image tokens / pixels (1 image, seeded sampling, guidance 5) | janus-pro-7b | identical / identical | exact / exact |
| End to end: 576 image tokens / pixels (batch of 2, seeded sampling, guidance 5) | janus-pro-7b | identical / identical | exact / exact |
| End to end, seed 42 (pixels) | janus-pro-7b | identical | exact |

The end-to-end prompt was "a photograph of an astronaut riding a horse". Both the transformers
reference and imagehub use seeded sampling, guidance 5, temperature 1.0 and 576 autoregressive
image tokens. The header checks covered 964 tensors for `janus-pro-1b` and 1018 tensors for
`janus-pro-7b`. Understanding-only tensors are skipped by an explicit allow-list; missing, extra
and shape-mismatched generation tensors are all **0**. Record: `tensor_header_check`,
[validation.json](validation.json).

**Checkpoint provenance.** The catalog uses the transformers conversions
`deepseek-community/Janus-Pro-1B` and `deepseek-community/Janus-Pro-7B` (provenance
`community-conversion`) because the authors' repositories ship weights for the original `janus`
package. For 1B, all 819 tensors used for image generation are byte-identical to the authors'
`deepseek-ai/Janus-Pro-1B` file; the 145 tensors that differ belong to the understanding-only
`vision_model`. For 7B, 873 of 1018 converted tensors are byte-identical to the authors'
`deepseek-ai/Janus-Pro-7B` shards; the 145 that differ are the understanding-only `vision_model`.

The 7B checkpoint is approximately 28 GB per float32 copy. To fit the 96 GB machine, one reference
and one native copy were loaded at a time. The full 7B validation run took 68 minutes.

## Stable Diffusion 3 and 3.5

Validated on CPU in the same environment as above against diffusers 0.40.0
`StableDiffusion3Pipeline` and transformers 5.18.0. `sd3-medium`, `sd3.5-medium`, `sd3.5-large`
and `sd3.5-large-turbo` were loaded strictly. Every numeric comparison was exactly **0**, so the
native implementation produced bit-identical results to the reference in CPU float32.

### Comparisons (CPU, float32)

| Comparison | Checkpoint | Largest observed difference | Tolerance |
| --- | --- | --- | --- |
| Three tokenizers (3 fixed prompts) | sd3-medium | identical | exact |
| CLIP-L / CLIP-G / T5-XXL text encoders | sd3-medium | 0 / 0 / 0 | 0 |
| 24-block MMDiT without q/k normalization, single-step calls (1024×1024, t = 1000 and 500) | sd3-medium | 0 | 0 |
| VAE decode / encode | sd3-medium | 0 / 0 | 0 |
| Flow-match Euler timesteps / sigmas (28 steps) | sd3-medium | 0 / 0 | 0 |
| End to end: final latent / pixels (1024×1024, 12 steps, guidance 7.0) | sd3-medium | 0 / 0 | 0 / 0 |
| End to end, seed 42 with the same settings (pixels) | sd3-medium | 0 | 0 |
| Three tokenizers (3 fixed prompts) | sd3.5-medium | identical | exact |
| CLIP-L / CLIP-G / T5-XXL text encoders | sd3.5-medium | 0 / 0 / 0 | 0 |
| MMDiT-X single-step calls (1024×1024, t = 1000 and 500) | sd3.5-medium | 0 | 0 |
| VAE decode / encode | sd3.5-medium | 0 / 0 | 0 |
| Flow-match Euler timesteps / sigmas (40 steps) | sd3.5-medium | 0 / 0 | 0 |
| End to end: final latent / pixels (1024×1024, 12 steps, guidance 4.5) | sd3.5-medium | 0 / 0 | 0 / 0 |
| End to end, seed 42 with the same settings (pixels) | sd3.5-medium | 0 | 0 |
| Three tokenizers (3 fixed prompts) | sd3.5-large | identical | exact |
| CLIP-L / CLIP-G / T5-XXL text encoders | sd3.5-large | 0 / 0 / 0 | 0 |
| MMDiT single-step calls (1024×1024, t = 1000 and 500) | sd3.5-large | 0 | 0 |
| VAE decode / encode | sd3.5-large | 0 / 0 | 0 |
| Flow-match Euler timesteps / sigmas (28 steps) | sd3.5-large | 0 / 0 | 0 |
| End to end: final latent / pixels (1024×1024, 8 steps, guidance 3.5) | sd3.5-large | 0 / 0 | 0 / 0 |
| End to end, seed 42 with the same settings (pixels) | sd3.5-large | 0 | 0 |
| Three tokenizers (3 fixed prompts) | sd3.5-large-turbo | identical | exact |
| CLIP-L / CLIP-G / T5-XXL text encoders | sd3.5-large-turbo | 0 / 0 / 0 | 0 |
| MMDiT single-step calls (1024×1024, t = 1000 and 500) | sd3.5-large-turbo | 0 | 0 |
| VAE decode / encode | sd3.5-large-turbo | 0 / 0 | 0 |
| Flow-match Euler timesteps / sigmas (4 steps) | sd3.5-large-turbo | 0 / 0 | 0 |
| End to end: final latent / pixels (1024×1024, 4 steps, guidance 0.0) | sd3.5-large-turbo | 0 / 0 | 0 / 0 |
| Guided end to end: final latent / pixels (1024×1024, 4 steps, guidance 3.5, negative prompt) | sd3.5-large-turbo | 0 / 0 | 0 / 0 |
| End to end, seed 42 with the defaults (pixels) | sd3.5-large-turbo | 0 | 0 |

For all four checkpoints, all three tokenizers matched. The T5 tokenizer additionally passed seeded
fuzzing and an every-code-point check against the snapshot's `tokenizer_3`.

The end-to-end prompt was "a photograph of an astronaut riding a horse" with initial noise
`seed=1234`. The guidance values are the model-card settings. `sd3-medium` used 12 steps instead of
its 28-step default, `sd3.5-medium` used 12 steps instead of its 40-step default, and large used 8
instead of its 28-step default; these reductions only shorten the CPU validation run. Large-turbo
used its defaults of 4 steps and guidance 0.0, with no reduction. Its guided comparison used
guidance 3.5 and the negative prompt "blurry". The seed 42 comparisons used the same size, step
count and guidance.

The header-only checks covered `text_encoder`, `text_encoder_2`, `text_encoder_3`, `transformer`
and `vae`. `sd3-medium` had 197 / 517 / 219 / 683 / 244 tensors; `sd3.5-medium` had
197 / 517 / 219 / 909 / 244; large had 197 / 517 / 219 / 1227 / 244; large-turbo had
197 / 517 / 219 / 1227 / 244. Missing, extra and shape-mismatched tensors were all **0**. Record:
`tensor_header_check`, [validation.json](validation.json).

To fit the 96 GB machine, each native/reference component pair was loaded and freed one at a time.
For end-to-end validation, the reference pipeline ran first and was freed before the native
pipeline ran. The `sd3-medium` run took 13 minutes, the `sd3.5-medium` run took 16 minutes and the
large run took 27 minutes. The large-turbo run took 17 minutes.

## Sana

Validated on CPU in the same environment as above. `sana-1600m-1024` and `sana-600m-1024` are
loaded strictly from `Efficient-Large-Model/Sana_1600M_1024px_diffusers` at revision `ac0da2f` and
`Efficient-Large-Model/Sana_600M_1024px_diffusers` at revision `28f3af7`. Every comparison below is
**exactly 0** (tolerance 0; identical bytes for images). The real-checkpoint run for both weight sets takes about 2 h 15 min on an M2 Max (CPU).

### Comparisons (CPU, float32, both weight sets)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| Tensor header check: text encoder, transformer, DC-AE (names and shapes) | 0 missing, extra or mismatched | 0 |
| Gemma tokenizer ids / attention masks (prompt with the complex human instruction, and `""`) | identical / identical | exact / exact |
| `SanaPipeline.encode_prompt` embeddings and masks (prompt and negative) | 0 | 0 |
| Transformer single step (batch 2, 32×32 latents, two timesteps, real conditioning) | 0 | 0 |
| DC-AE decode (32×32 latents, scaled as the pipeline does) | 0 | 0 |
| End to end: final latent / pixels (1024×1024, 20 steps, guidance 4.5, seed 0) | 0 / 0 | 0 / 0 |
| End to end, seed 1234, 1360×768 request (binning, resize and crop back; pixels) | 0 | 0 |

The end-to-end prompt was "a photograph of an astronaut riding a horse". The reference text encoder
was loaded exactly as `SanaPipeline.from_pretrained(..., torch_dtype=torch.float32)` loads it. The
non-square seed test exercises resolution binning and the resize and centre crop of the decoded
image back to the requested size.

Random-weight parity covered the Gemma-2 text model, the transformer and DC-AE with tiny and with the
published configurations at small spatial sizes, and tiny-snapshot end-to-end parity covered binning
on and off, guidance on and off and non-square sizes. The Gemma tokenizer is additionally covered by
fixed-prompt and seeded fuzz tests. The scheduler is checked against diffusers with flow sigmas and
`flow_prediction`.

The reference and native components were loaded one model id at a time; the float32 reference
needs about 6.4 GB for the transformer and 10.5 GB for Gemma-2.

## Lumina-Image 2.0

`lumina-image-2.0` is loaded strictly from `Alpha-VLLM/Lumina-Image-2.0` at revision `53504ab` and
compared with diffusers 0.40.0 `Lumina2Pipeline` on CPU in float32. Every comparison below measured
exactly 0. The real-checkpoint run takes about 50 minutes on an M2 Max (CPU).

### Comparisons (CPU, float32)

| Comparison | Measured difference |
| --- | --- |
| Tensor header check: text encoder, transformer, autoencoder (names and shapes) | 0 missing, extra or mismatched |
| Gemma tokenizer ids / attention masks (system prompt and prompt, a truncated prompt, and `""`) | exact / exact |
| `Lumina2Pipeline.encode_prompt` embeddings and masks (prompt and negative) | 0 |
| Transformer single step (batch 2, 64×64 latents, two timesteps, real conditioning) | 0 |
| Autoencoder decode (64×64 latents, scaled and shifted as the pipeline does) | 0 |
| End to end: final latent / pixels (1024×1024, 30 steps, guidance 4.0, seed 0) | 0 / 0 |
| End to end, seed 1234, 1360×768 request (pixels) | 0 |

The reference text encoder is loaded exactly as `Lumina2Pipeline.from_pretrained(...,
torch_dtype=torch.float32)` loads it, and its second-to-last hidden state is the conditioning. The
end-to-end reference run decodes its own latents with the pipeline's autoencoder and image processor.

Random-weight parity covers the transformer with tiny and published configurations and the Gemma-2
text model, including every entry of the hidden-state tuple. Tiny-snapshot end-to-end parity covers
the default settings, `cfg_trunc_ratio` below 1, `cfg_normalization` off, guidance 1.0, non-square
sizes, a custom system prompt and negative prompt, and several seeds.

The end to end first differed by about 3e-4 in the latents. The cause was the memory layout of the
loaded weights: when diffusers loads a sharded checkpoint without accelerate, it keeps the shard that
holds the model's first tensor at its file offsets and copies the other shards into aligned buffers,
and the CPU kernels sum in a layout-dependent order. The transformer now loads its shards the same
way (`keep_first_shard_mapped`).

## aMUSEd

`amused-256` and `amused-512` are loaded strictly from `amused/amused-256` at revision `09b6259` and
`amused/amused-512` at revision `6f3f309`, and are compared with diffusers 0.40.0 `AmusedPipeline` on
CPU in float32. The real-checkpoint tests (`tests/reference/test_ref_amused.py`) assert the exact
equalities below for both models. All 14 real-checkpoint tests (7 per model) passed, and every
comparison matched exactly. The real-checkpoint run for both models took about 5 minutes on an
Apple M2 Max (CPU). The tiny-snapshot and random-weight comparisons need no downloaded weights and
pass with exactly 0 difference.

### Comparisons (CPU, float32)

| Comparison | Measured (`amused-256`, `amused-512`) |
| --- | --- |
| Tensor header check: text encoder, transformer, VQGAN (names and shapes) | `amused-256`: 197, 474 and 277 tensors; `amused-512`: 198, 478 and 277 tensors; 0 missing, extra or mismatched in both |
| CLIP tokenizer ids (prompt, `""`, punctuation and case, a truncated prompt) | all 4 cases exact for both |
| Text encoder: projected pooled embedding and second-to-last hidden state (prompt and `""`) | max abs 0, max rel 0 for both |
| One transformer step on a partly masked token grid with micro-conditioning | max abs 0, max rel 0 for both |
| VQGAN decode of fixed token ids (codebook lookup, group normalization) | max abs 0, max rel 0 for both |
| End to end at the model defaults (12 steps, guidance 10, seed 0): final token ids and uint8 image | token ids equal, max pixel difference 0 (256×256 and 512×512) |
| Seeded pipeline run at a different, non-square size (seed 1234): uint8 image | max pixel difference 0 (`amused-256` at 320×192, `amused-512` at 640×384) |

Random-weight comparisons, which pass with exactly 0 difference: the U-ViT with a tiny
configuration (with and without bias, with and without the downsampling and upsampling of the 512
model) and with the published 256 configuration at full size; the scheduler trajectory with seeded
generators (cosine and linear schedules, number and pair temperatures, the generator-free random
state); and the VQGAN with group normalization, with and without the attention block and the
codebook lookup, and with the published configuration. Tiny-snapshot end-to-end runs compare the
token ids and the image bytes for both transformer variants, with the defaults, guidance 1.0 (no
guidance), a negative prompt, micro-conditioning and temperature settings, a larger grid, and several
seeds.

## Kolors

`kolors` is loaded strictly from `Kwai-Kolors/Kolors-diffusers` at revision `7e091c7` and compared
with diffusers 0.40.0 `KolorsPipeline` on CPU in float32, on the float32 upcast of the published
float16 weights (only float16 weights are published). Every comparison below measured exactly 0.
The real-checkpoint run takes about 1 h 20 min on an M2 Max (CPU).

### Comparisons (CPU, float32)

| Comparison | Measured difference |
| --- | --- |
| Tensor header check: text encoder (200 tensors), U-Net (1682), autoencoder (248) (names and shapes) | 0 missing, extra or mismatched |
| ChatGLM tokenizer ids / attention masks (11 prompts, a 2000-prompt fuzz, 3 snapshot cases) | identical / identical |
| `KolorsPipeline.encode_prompt` embeddings and pooled embeddings (prompt and negative) | 0 |
| U-Net single step (real conditioning) | 0 |
| Autoencoder decode | 0 |
| End to end: final latent / pixels (1024×1024, 50 steps, guidance 5.0, seed 0) | 0 / 0 |
| End to end, seed 1234, 1360×768 request (pixels) | 0 |

The reference pipeline was loaded with `variant="fp16"` and `torch_dtype=torch.float32`, so both
sides compute on the same upcast weights.

Random-weight parity covers the ChatGLM encoder in five configurations (the published layout,
post-residual layer norm, multi-head attention without the final norm, linear biases, and a wider
model), on left-padded batches. Tiny-snapshot end-to-end parity covers four cases (the default
settings, guidance 1.0, a non-square size with a negative prompt, and micro-conditioning with
explicit original size, crop and target size) with two sets of random weights each, plus a check of
the latents. The Euler scheduler ignores the configuration keys that diffusers' Euler scheduler does
not read (`thresholding`, `dynamic_thresholding_ratio`, `clip_sample_range`), which this release's
configuration contains.

## AuraFlow

Validated on CPU in the same environment as above. `auraflow-v0.3` is loaded strictly from
`fal/AuraFlow-v0.3` at revision `2cd8588`. Every comparison below is required to be **exactly 0**
(tolerance 0; identical bytes for images). The real-checkpoint run took about 3 h 41 min on an M2 Max (CPU).

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| Tensor header check: text encoder, transformer, VAE (names and shapes) | 0 missing, extra or mismatched | 0 |
| Tokenizer ids / attention masks (prompt and `""`, 256 tokens) | identical / identical | exact / exact |
| `AuraFlowPipeline.encode_prompt` embeddings (prompt and negative) | 0 | 0 |
| Transformer single step (batch 2, 32×48 latents, two timesteps, real conditioning) | 0 | 0 |
| VAE decode (32×48 latents, scaled as the pipeline does) | 0 | 0 |
| End to end: final latent / pixels (1024×1024, 50 steps, guidance 3.5, seed 0) | 0 / 0 | 0 / 0 |
| End to end, seed 1234, 1360×768 (pixels) | 0 | 0 |

The end-to-end prompt was "a photograph of an astronaut riding a horse". The reference was loaded
exactly as `AuraFlowPipeline.from_pretrained(..., torch_dtype=torch.float32)` loads it, with the
default negative prompt.

Random-weight parity covered the UMT5 encoder and the transformer with tiny and with the published
configuration at small spatial sizes, and tiny-snapshot end-to-end parity covered the default
settings, guidance on and off, non-square sizes, a truncated prompt and several seeds. The tokenizer
is additionally covered by fixed-prompt and seeded fuzz tests on a synthetic folder.

The float32 transformer needs about 27 GB, so the reference and native transformers were loaded one
after the other and each was freed before the next.

## OmniGen

`omnigen-v1` is loaded strictly from `Shitao/OmniGen-v1-diffusers` at revision `016e2f6` and is compared
with diffusers 0.40.0 `OmniGenPipeline` on CPU in float32. The published transformer is stored in
bfloat16 and converted to float32 on load by both implementations; the autoencoder is float32. The
real-checkpoint tests are in `tests/reference/test_ref_omnigen.py`; their measurements are pending the
real-checkpoint run, so none are listed here yet. The end-to-end runs use shortened step counts
because a 1024×1024, 50-step run takes hours on CPU.

### Comparisons (CPU, float32)

| Comparison | Measured difference |
| --- | --- |
| Tensor header check: transformer, autoencoder (names and shapes) | pending the real-checkpoint run |
| Tokenizer ids (instruction prompt, edit prompt, long prompt, `""`) and byte equality of the tokenizer files with the fixtures | pending the real-checkpoint run |
| Transformer single step (3 timesteps, with and without an input image, real conditioning) | pending the real-checkpoint run |
| Autoencoder posterior sample and decode | pending the real-checkpoint run |
| End to end text-to-image: final latent / pixels (1024×1024, shortened steps, guidance 2.5, seed 0) | pending the real-checkpoint run |
| End to end with an input image: final latent / pixels (512×512, shortened steps, seed 0) | pending the real-checkpoint run |
| End to end, seed 1234, 1360×768 request (pixels, shortened steps) | pending the real-checkpoint run |

Tests that need no downloaded weights: random-weight parity of the transformer with a tiny and with the
published configuration (two layers), with and without input-image tokens and with sequences long
enough to switch the rotary embedding to its long factors; tokenizer parity with transformers on the
published tokenizer files (fixed prompts, a seeded fuzz test and random Unicode); parity of the image
preprocessing, token layout, attention masks and position ids with the diffusers processor; parity of
the inverted flow-match schedule and Euler steps; and tiny-snapshot end-to-end parity covering the
defaults, guidance 1.0 and above, non-square sizes, several seeds, one step, an input image,
`use_input_image_size_as_output`, a small `max_input_image_size` and RGBA and grayscale inputs.

Where the reference pipeline cannot run on its own, parity is defined as follows. The reference fails on
non-RGB input images, so the comparison uses the RGB conversion, as for FLUX.1 Kontext. The reference
draws the input-image posterior from the global torch generator, so the reference runs call
`torch.manual_seed(seed)` before the pipeline. The reference rotary embedding builds its table from the
first sequence of the batch only; the native module reproduces that.

## Device smoke tests (no parity claim)

| Device | Result | Largest latent difference from CPU | Largest pixel difference | PSNR |
| --- | --- | --- | --- | --- |
| MPS (Apple M2 Max) | finite, 512×512, 20 steps | 2.7e-3 | 1 | 72.8 dB |
| CUDA | not run yet | | | |

## Limits

- The parity claim holds only for float32 on CPU, against diffusers 0.40.0 and transformers 5.18.0.
- MPS and CUDA runs are smoke tests: they check for finite output and the correct size, and record
  the difference from CPU. There is no parity claim on these devices.
- There is no parity claim for float16 or bfloat16.
- There is no safety checker or invisible watermark. The reference was run with `safety_checker=None`.
- Only the scheduler published with each checkpoint is implemented: PNDM (PLMS) for SD 1.x, Euler
  with trailing spacing for SD-Turbo, LCM for LCM Dreamshaper v7, Euler for SDXL base, SSD-1B and Kolors,
  LCM for Hyper-SDXL and DMD2, Euler ancestral for SDXL Turbo, EDM DPM-Solver++ for Playground v2.5,
  flow-match Euler for Z-Image and Stable Diffusion 3 and 3.5, flow-match Euler with a static shift
  for FLUX.1 schnell, Lumina-Image 2.0 and AuraFlow, flow-match Euler with inverted sigmas for OmniGen, and dynamic shifting for FLUX.1 dev, Qwen-Image 2.1 and LongCat-Image, flow-match
  Euler with exponential shifting for Krea 2 and linear shifting for CogView4, DDPM for HunyuanDiT,
  UnCLIP and DDPM for Kandinsky 2.2, DDPM Wuerstchen for Stable Cascade and Wuerstchen v2, DPM-Solver++ for PixArt and Sana (flow sigmas), and
  the cosine-schedule masked-token scheduler for aMUSEd.
- No image quality benchmark (FID, CLIP score) was run.
- Only text-to-image generation is supported. SD 1.x img2img is not implemented.

## Automated tests

- Regular tests: 630 passed, 500 skipped (the skipped ones are the optional reference tests). Tests do
  not download models from the network.
- Reference tests (CPU): 34 passed for SD 1.x and 25 passed, 2 skipped for the SDXL family
  (`sdxl-base-1.0`, `ssd-1b`, `sdxl-turbo`), 12 passed for the three SDXL-Lightning models, 4 passed
  for `hyper-sdxl-1step`, 8 passed for `dmd2-sdxl-4step` and `dmd2-sdxl-1step`,
  8 passed, 1 skipped for `playground-v2.5`, 9 passed for `z-image-turbo`, 8 passed for `z-image`,
  6 passed for `janus-pro-1b`, 9 passed for `janus-pro-7b`, 9 passed for `qwen-image-2.1`, 7 passed
  for `flux2-klein-4b`, 7 passed for `flux2-klein-9b`, 8 passed each for `flux1-schnell`,
  `flux1-dev` and `flux1-krea-dev`, 7 passed for `flux1-kontext-dev`, 15 passed for `hunyuan-dit-v1.1-distilled`, 5 passed for `kandinsky-2.2`, 7 passed each for `stable-cascade` and `stable-cascade-lite`, 19 passed
  for `pixart-sigma-xl-1024` and `pixart-alpha-xl-1024`, 7 passed for `krea2-turbo`, 8 passed for `longcat-image`, 9 passed for
  `cogview4-6b`, 7 passed for `ideogram4-fp8`, 9 passed for `sd-turbo`, 8 passed for `lumina-image-2.0`, 7 passed for `kolors`, 7 passed for `auraflow-v0.3`, 7 passed for `wuerstchen-v2`, 8 passed for `sd-v2-1`, 7 passed each for `sana-600m-1024` and `sana-1600m-1024`, 7 passed each for `amused-256` and
  `amused-512`, and 7 passed for
  `lcm-dreamshaper-v7`. These include structural parity with random weights, end-to-end parity with
  a small model, real-checkpoint parity and tokenizer fuzz tests.
- Stable Diffusion 3 and 3.5 reference tests (CPU): 10 passed for `sd3-medium`, 10 passed each for
  `sd3.5-medium` and `sd3.5-large`, and 11 passed for `sd3.5-large-turbo`.
- MPS smoke test: 1 passed.

## Differences found and fixed during validation

The first run with real checkpoints exposed two tokenizer differences that tests with random weights
could not show. Both were fixed and are guarded by permanent fuzz tests:

1. **The `normalized` flag of special tokens.** `<|startoftext|>` matches in normalized text, while
   `<|endoftext|>` matches only as its exact spelling in the raw text. This is because transformers
   registers special tokens one after another and the last registration takes effect.
2. **Splitting of sequences not recognized as special tokens.** These sequences are split into three
   pieces, `<|`, `endoftext`, `|>`, as in the reference pre-tokenizer.

The SDXL validation found three more differences:

3. **Memory-mapped safetensors files.** CPU BLAS picked a different summation order for
   differently aligned weights, so a 4.8e-7 difference in the pooled text embedding grew to a
   29-pixel difference over sampling. Safetensors files are now memory-mapped, so the weights are
   read from the file at their published alignment.
4. **`null` for disabled options.** The published SDXL config uses `null` for disabled options,
   for example `class_embed_type` and `cross_attention_norm` in the U-Net config.
5. **Special-token metadata in `added_tokens_decoder`.** SDXL Turbo's tokenizer configs define
   special-token normalization there, so those entries now take precedence over the named
   `unk_token`, `bos_token`, `eos_token` and `pad_token` values.
6. **Older Unicode tables in the reference tokenizer.** For 108 combining marks, the reference
   tokenizer treats code points Python recognizes as assigned as unassigned.
7. **Sharded checkpoint alignment.** The reference loaders copy sharded checkpoints into fresh
   64-byte aligned tensors, unlike single-file checkpoints, and BLAS summation order depends on
   alignment. The first run therefore differed by 4.2e-5 in the transformer, 3.1e-3 in unguided
   latents and 1 pixel.
8. **SDPA attention in the Janus reference.** The reference loads the language model with SDPA
   attention. Eager attention differed by 5.5e-6 in the last hidden state.
9. **Janus return-sequence configuration.** In transformers 5.18.0, setting
   `model.generation_config.num_return_sequences` as the model card shows fails validation. The
   equivalent `generate(num_return_sequences=...)` argument is used instead.
10. **Euler initial noise scale with trailing spacing.** With `timestep_spacing="trailing"`, the
    Euler scheduler's initial noise scale is the largest sigma itself, not `sqrt(sigma² + 1)`.
