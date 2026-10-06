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

Validated on CPU on October 5, 2026, in the same environment as above. Four checkpoints, `full`:
`sdxl-base-1.0`, `sdxl-turbo`, `ssd-1b` and `playground-v2.5`. The published tensors were loaded
strictly. A generation at each model's native size was compared against diffusers from the same
initial noise.

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

For `sdxl-base-1.0`, the same header-only tensor check covered the four components
(`text_encoder`, `text_encoder_2`, `unet`, `vae`): tensor counts 196 / 517 / 1680 / 248, and
missing, extra and shape-mismatched tensors are **0**. Record: `tensor_header_check`,
[validation.json](validation.json).

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

## Stable Diffusion 3.5

Validated on CPU in the same environment as above against diffusers 0.40.0
`StableDiffusion3Pipeline` and transformers 5.18.0. `sd3.5-medium`, `sd3.5-large` and
`sd3.5-large-turbo` were loaded strictly. Every numeric comparison was exactly **0**, so the native
implementation produced bit-identical results to the reference in CPU float32.

### Comparisons (CPU, float32)

| Comparison | Checkpoint | Largest observed difference | Tolerance |
| --- | --- | --- | --- |
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

For all three checkpoints, all three tokenizers matched. The T5 tokenizer additionally passed seeded
fuzzing and an every-code-point check against the snapshot's `tokenizer_3`.

The end-to-end prompt was "a photograph of an astronaut riding a horse" with initial noise
`seed=1234`. The guidance values are the model-card settings. Medium used 12 steps instead of its
40-step default, and large used 8 instead of its 28-step default; these reductions only shorten
the CPU validation run. Large-turbo used its defaults of 4 steps and guidance 0.0, with no
reduction. Its guided comparison used guidance 3.5 and the negative prompt "blurry". The seed 42
comparisons used the same size, step count and guidance.

The header-only checks covered `text_encoder`, `text_encoder_2`, `text_encoder_3`, `transformer`
and `vae`. Medium had 197 / 517 / 219 / 909 / 244 tensors; large had
197 / 517 / 219 / 1227 / 244; large-turbo had 197 / 517 / 219 / 1227 / 244. Missing, extra and
shape-mismatched tensors were all **0**. Record: `tensor_header_check`,
[validation.json](validation.json).

To fit the 96 GB machine, each native/reference component pair was loaded and freed one at a time.
For end-to-end validation, the reference pipeline ran first and was freed before the native
pipeline ran. The medium run took 16 minutes and the large run took 27 minutes.
The large-turbo run took 17 minutes.

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
  with trailing spacing for SD-Turbo, Euler for SDXL base and SSD-1B, Euler ancestral for SDXL
  Turbo, EDM DPM-Solver++ for Playground v2.5, flow-match Euler for Z-Image and Stable Diffusion
  3.5, and flow-match Euler with dynamic shifting for Qwen-Image 2.1.
- No image quality benchmark (FID, CLIP score) was run.
- Only text-to-image generation is supported. SD 1.x img2img is not implemented.

## Automated tests

- Regular tests: 396 passed, 227 skipped (the skipped ones are the optional reference tests). Tests do
  not download models from the network.
- Reference tests (CPU): 34 passed for SD 1.x and 25 passed, 2 skipped for the SDXL family
  (`sdxl-base-1.0`, `ssd-1b`, `sdxl-turbo`), 8 passed, 1 skipped for `playground-v2.5`, and 9 passed for
  `z-image-turbo`, 8 passed for `z-image`, 6 passed for `janus-pro-1b`, 9 passed for `janus-pro-7b`,
  9 passed for `qwen-image-2.1`, and 9 passed for `sd-turbo`. These
  include structural parity with random weights, end-to-end parity with a small model,
  real-checkpoint parity and tokenizer fuzz tests.
- Stable Diffusion 3.5 reference tests (CPU): 10 passed each for `sd3.5-medium` and
  `sd3.5-large`, and 11 passed for `sd3.5-large-turbo`.
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
