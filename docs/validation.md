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

Validated on CPU on October 5, 2026, in the same environment as above. Two checkpoints, `full`:
`sdxl-base-1.0` and `ssd-1b`. The published tensors were loaded strictly. A 1024×1024 generation
from the same initial noise was compared against diffusers.

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
| Tokenizer token ids (2 tokenizers, 18 prompts) | ssd-1b | identical | identical |
| CLIP ViT-L/14 text encoder (penultimate layer) | ssd-1b | 0 | 0 |
| OpenCLIP bigG text encoder (penultimate layer and pooled output) | ssd-1b | 0 / 0 | 0 / 0 |
| U-Net single step (t = 999, 500, 1) | ssd-1b | 0 | 0 |
| VAE decode / encode | ssd-1b | 0 / 0 | 0 |
| Euler timesteps / sigmas (50 and 30 steps) | ssd-1b | 0 / 0 | 0 |
| End to end: final latent / pixels | ssd-1b | 0 / 0 | 0 / 0 |
| End to end, seed 42 (pixels) | ssd-1b | 0 | 0 |

The observed difference in every comparison is **exactly 0**: on CPU in float32, the local
implementation produces a bit-identical image to the one diffusers produces. The same tolerance
rule applies: since every measurement came out 0, no tolerance needed to be loosened.

End-to-end test settings: prompt "a photograph of an astronaut riding a horse", 1024×1024,
guidance 5.0, initial noise `seed=1234` (the seed 42 run uses the same settings). 30 steps are
used instead of the default 50 to keep CPU runtime reasonable.

For `sdxl-base-1.0`, the same header-only tensor check covered the four components
(`text_encoder`, `text_encoder_2`, `unet`, `vae`): tensor counts 196 / 517 / 1680 / 248, and
missing, extra and shape-mismatched tensors are **0**. Record: `tensor_header_check`,
[validation.json](validation.json).

For `ssd-1b`, the same header-only tensor check covered the four components (`text_encoder`,
`text_encoder_2`, `unet`, `vae`): tensor counts 196 / 517 / 944 / 248, and missing, extra and
shape-mismatched tensors are **0**. Record: `tensor_header_check`,
[validation.json](validation.json).

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
- Only the scheduler published with each checkpoint is implemented: PNDM (PLMS) for SD 1.x and Euler for SDXL.
- No image quality benchmark (FID, CLIP score) was run.
- Only text-to-image generation is supported. SD 1.x img2img is not implemented.

## Automated tests

- Regular tests: 133 passed, 36 skipped (the skipped ones are the optional reference tests). Tests do
  not download models from the network.
- Reference tests (CPU): 34 passed. These include structural parity with random weights, end-to-end
  parity with a small model, real-checkpoint parity and tokenizer fuzz tests.
- MPS smoke test: 1 passed.

## Differences found and fixed during validation

The first run with real checkpoints exposed two tokenizer differences that tests with random weights
could not show. Both were fixed and are guarded by permanent fuzz tests:

1. **The `normalized` flag of special tokens.** `<|startoftext|>` matches in normalized text, while
   `<|endoftext|>` matches only as its exact spelling in the raw text. This is because transformers
   registers special tokens one after another and the last registration takes effect.
2. **Splitting of sequences not recognized as special tokens.** These sequences are split into three
   pieces, `<|`, `endoftext`, `|>`, as in the reference pre-tokenizer.

The SDXL validation found two more differences:

3. **Memory-mapped safetensors files.** CPU BLAS picked a different summation order for
   differently aligned weights, so a 4.8e-7 difference in the pooled text embedding grew to a
   29-pixel difference over sampling. Safetensors files are now memory-mapped, so the weights are
   read from the file at their published alignment.
4. **`null` for disabled options.** The published SDXL config uses `null` for disabled options,
   for example `class_embed_type` and `cross_attention_norm` in the U-Net config.
