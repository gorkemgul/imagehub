# FLUX.1 design

- Date: 2026-10-06
- Issue: #16
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`flux1-schnell` (`black-forest-labs/FLUX.1-schnell`, Apache-2.0) and `flux1-dev`
(`black-forest-labs/FLUX.1-dev`, FLUX.1 [dev] Non-Commercial License), both gated, text-to-image only,
at `full` validation status, as a new family `flux`. Out of scope: FLUX.1 Kontext, Fill, Redux, Depth
and Canny variants, ControlNet, IP-Adapter, LoRA, fp16/bf16 parity.

## Reference

diffusers 0.40.0 `FluxPipeline` (`pipelines/flux/pipeline_flux.py`), `FluxTransformer2DModel`,
`AutoencoderKL`, `FlowMatchEulerDiscreteScheduler`; transformers 5.18.0 `CLIPTextModel`,
`T5EncoderModel`, `CLIPTokenizer`, `T5TokenizerFast`.

## Components

| Part | Status |
| --- | --- |
| CLIP-L tokenizer and text encoder (pooled output only) | exists |
| T5 tokenizer (`tokenizer.json`) and T5-XXL encoder | exists (SD 3) |
| 16-channel VAE with `scaling_factor` and `shift_factor`, no quant convs | exists (Z-Image) |
| Flow-match Euler; static shift (schnell) and dynamic shifting with `calculate_shift` (dev) | exists |
| `FluxTransformer2DModel`: 19 double-stream and 38 single-stream blocks, 24 heads × 128, 3-axis RoPE, pooled-projection and (dev) guidance embeddings | **new** |

## Pipeline

As `FluxPipeline.__call__` for text-to-image: CLIP pooled embedding and T5 sequence embeddings
(`max_sequence_length` 256 for schnell, 512 for dev, as the model cards use), text ids, packed 2×2
latents from a seeded CPU generator, latent image ids, `sigmas = linspace(1, 1/steps, steps)` and `mu`
(dev), timesteps divided by 1000 for the transformer, the guidance embedding (dev) or none (schnell),
true CFG only when `true_cfg_scale > 1` with a negative prompt (off by default), unpacking,
`latents / scaling_factor + shift_factor`, VAE decode. Defaults from the model cards: schnell 4 steps,
guidance 0.0, max sequence 256; dev 50 steps, guidance 3.5, max sequence 512; both 1024×1024.

## Memory

The transformer is 11.9B parameters: ≈ 48 GB per fp32 copy, so the reference and the native copy do
not fit together on the 96 GB machine. Every real-checkpoint comparison therefore runs the reference
first, saves its outputs (tensors and pixels) to a temporary file, frees the reference, then loads the
native model and compares against the saved outputs. Component tests do the same per component. A
full pipeline alone (transformer + T5-XXL + CLIP + VAE in fp32) is about 68 GB.

## Validation

Random-weight parity for the transformer (tiny config exact; medium config; full-config layout on the
meta device), a tiny-snapshot end to end against `FluxPipeline` for both a schnell-like and a dev-like
(guidance embedding, dynamic shifting) configuration, then real-checkpoint component parity
(tokenizers, CLIP pooled, T5, transformer one step at 1024×1024, VAE, scheduler) and end-to-end parity
at 1024×1024 (schnell at its 4-step default; dev with a reduced step count, recorded), plus seed tests,
all with the save-then-compare sequence.

## Task breakdown

1. Transformer + random-weight parity.
2. Backend (`flux` family), catalog entries, sequential reference test helpers, tiny-snapshot tests,
   real-checkpoint tests.
3. Real-checkpoint parity (schnell, then dev), docs; PR closes #16.
