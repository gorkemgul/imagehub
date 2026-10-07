# PixArt design

- Date: 2026-10-07
- Issue: #37
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`pixart-sigma-xl-1024` (`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`, revision `e102b35`) and `pixart-alpha-xl-1024`
(`PixArt-alpha/PixArt-XL-2-1024-MS`, revision `b89adad`), both CreativeML-OpenRAIL++-M and not gated, text-to-image
only, at `full` validation status, as a new family `pixart`. Out of scope: the 256/512 and 2K releases (they can follow
with the same code), PixArt-LCM and PixArt-DMD, ControlNet, caption cleaning (`clean_caption`, which needs `bs4` and
`ftfy`; diffusers turns it off when they are missing), fp16/bf16 parity.

## Reference

diffusers 0.40.0 `PixArtSigmaPipeline` and `PixArtAlphaPipeline` (`pipelines/pixart_alpha/`),
`PixArtTransformer2DModel` (the `Transformer2DModel` configs load into it), `AutoencoderKL`,
`DPMSolverMultistepScheduler`; transformers 5.18.0 `T5EncoderModel` and `T5Tokenizer` (from `spiece.model`).

## Components

| Part | Status |
| --- | --- |
| T5 tokenizer from `spiece.model` (100 extra ids), 300 (Sigma) / 120 (Alpha) tokens | exists (HunyuanDiT work) |
| T5-v1.1-XXL encoder | exists |
| 4-channel VAE: Sigma uses the SDXL VAE (scaling 0.13025), Alpha the SD VAE (0.18215) | exists |
| `PixArtTransformer2DModel`: 28 blocks, 1152 wide, 16 heads × 72, `ada_norm_single` (timestep modulation shared across blocks), caption projection, 2D sin-cos position embedding with interpolation scale; Alpha adds resolution and aspect-ratio embeddings | **new** |
| DPM-Solver++ multistep (order 2, midpoint, linspace spacing, epsilon prediction, lower-order final) | **new** |

## Pipeline

As the two pipelines' `__call__` for text-to-image with `clean_caption=False`: T5 embeddings with attention masks
(classifier-free guidance with an empty negative prompt, the mask handling each pipeline applies), resolution binning
to the aspect-ratio tables, latents from a seeded CPU generator, the Alpha size conditioning, the learned-sigma split,
DPM-Solver++ steps, VAE decode. Defaults from the pipelines and model cards: 20 steps, guidance 4.5, 1024×1024.

## Memory

fp32: T5-XXL ≈ 19 GB, transformer ≈ 2.4 GB, VAE ≈ 0.3 GB. Reference and native fit together.

## Validation

Random-weight parity for the transformer (Sigma-style and Alpha-style tiny configs exact; full-config layouts),
DPM-Solver++ parity (timesteps and trajectories), a tiny-snapshot end to end against both pipelines, then for each
model: tokenizer and T5 parity, transformer single step at 1024×1024, VAE decode/encode, end to end at the defaults
(latents and pixels) and the seed test, plus the gallery sample image.

## Task breakdown

1. DPM-Solver++ scheduler and the PixArt transformer, with parity tests.
2. Backend (`pixart` family), catalog entries, tiny-snapshot and real-checkpoint tests.
3. Real-checkpoint parity, docs; PR closes #37.
