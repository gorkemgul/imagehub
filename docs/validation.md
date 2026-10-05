# Validation — imagehub 0.1

Validated on CPU on October 4, 2026. Environment: Python 3.13.11 / PyTorch 2.14.1.
Reference: diffusers 0.40.0, transformers 5.18.0. Machine: Apple M2 Max, 96 GB unified memory,
12 CPU threads. Sources, commits and file fingerprints are in [validation.json](validation.json).

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

Validated on CPU on October 5, 2026, in the same environment as above. `z-image-turbo` was loaded
strictly from `Tongyi-MAI/Z-Image-Turbo`. The observed difference in every comparison is **exactly
0**: on CPU in float32, the local implementation produces a bit-identical result to the reference.

### Comparisons (CPU, float32)

| Comparison | Largest observed difference | Tolerance |
| --- | --- | --- |
| Tokenizer token ids and attention masks (3 fixed prompts, seeded fuzz, exhaustive per-code-point check) | identical | exact |
| Qwen3 text encoder (`hidden_states[-2]`) | 0 | 0 |
| Transformer single step (1024×1024, t = 0.5 and 0.25) | 0 | 0 |
| VAE decode / encode | 0 / 0 | 0 |
| Flow-match scheduler timesteps / sigmas | 0 / 0 | 0 |
| End to end: final latent / pixels (1024×1024, 9 steps, guidance 0.0) | 0 / 0 | 0 / 0 |
| Guided end to end: final latent / pixels (1024×1024, 4 steps, guidance 4.0, negative prompt) | 0 / 0 | 0 / 0 |
| End to end, seed 42 (1024×1024, 9 steps, guidance 0.0; pixels) | 0 | 0 |

The end-to-end prompt was "a photograph of an astronaut riding a horse" with initial noise
`seed=1234`; the guided run uses a negative prompt. The tokenizer check covers fixed prompts, seeded
fuzz and every Unicode code point. The reference tokenizer's Unicode tables are older than Python's
for 108 combining marks, which it treats as unassigned.

To fit the 96 GB machine, each component pair (ours and the reference) is loaded and freed one at a
time, and in the end-to-end checks the reference pipeline runs first and is freed before ours runs.

The header-only tensor check covered `text_encoder`, `transformer` and `vae`: tensor counts
398 / 521 / 244, with missing, extra and shape-mismatched tensors all **0**. Record:
`tensor_header_check`, [validation.json](validation.json).

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
  for SDXL base and SSD-1B, Euler ancestral for SDXL Turbo, EDM DPM-Solver++ for Playground v2.5,
  and flow-match Euler for Z-Image.
- No image quality benchmark (FID, CLIP score) was run.
- Only text-to-image generation is supported. SD 1.x img2img is not implemented.

## Automated tests

- Regular tests: 313 passed, 127 skipped (the skipped ones are the optional reference tests). Tests do
  not download models from the network.
- Reference tests (CPU): 34 passed for SD 1.x and 25 passed, 2 skipped for the SDXL family
  (`sdxl-base-1.0`, `ssd-1b`, `sdxl-turbo`), 8 passed, 1 skipped for `playground-v2.5`, and 9 passed for
  `z-image-turbo`. These
  include structural parity with random weights, end-to-end parity with a small model,
  real-checkpoint parity and tokenizer fuzz tests.
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
