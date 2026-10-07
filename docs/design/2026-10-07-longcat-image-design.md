# LongCat-Image design

- Date: 2026-10-07
- Issue: #40
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`longcat-image` (`meituan-longcat/LongCat-Image`, revision `d2ea50b`, Apache-2.0, not gated), text-to-image only, at
`full` validation status, as a new family `longcat-image`. Out of scope: LongCat-Image-Edit, LoRA, fp16/bf16 parity. The
weights are published in bfloat16; parity runs in float32 after upcasting.

## Reference

diffusers 0.40.0 `LongCatImagePipeline` (`pipelines/longcat_image/pipeline_longcat_image.py`),
`LongCatImageTransformer2DModel`, `AutoencoderKL`, `FlowMatchEulerDiscreteScheduler`; transformers 5.18.0
`Qwen2_5_VLForConditionalGeneration`, the Qwen2 tokenizer and `Qwen2VLProcessor` (text only).

## Components

| Part | Status |
| --- | --- |
| Qwen2 tokenizer and the pipeline's prompt templates (encoding and rewriting) | tokenizer exists; templates are new |
| Qwen2.5-VL text model (28 layers, 3584 wide, 28 query / 4 key-value heads, multimodal RoPE with sections 16/24/24) and its LM head | **new** |
| Prompt rewrite: generation with the published generation config (`top_k=1`, repetition penalty 1.05, temperature 0.1, top-p 0.001), deterministic, up to 512 new tokens | **new** |
| `LongCatImageTransformer2DModel`: FLUX-style, 10 double-stream and 20 single-stream blocks, 24 heads × 128, no guidance embedding | **new**; FLUX.1 building blocks are reused where numerics are identical |
| 16-channel FLUX VAE with scaling and shift factors | exists |
| Flow-match Euler with dynamic shifting | exists |
| CFG renormalization (`enable_cfg_renorm`, `cfg_renorm_min`) | **new** (pipeline) |

## Pipeline

As `LongCatImagePipeline.__call__` for text-to-image: optional prompt rewrite, prompt encoding with the pipeline's template
and token layout, empty negative prompt for classifier-free guidance, packed latents from a seeded CPU generator, position
ids, sigmas and `mu`, the loop with CFG and CFG renormalization, unpacking, `latents / scaling_factor + shift_factor`, VAE
decode. Defaults from the pipeline and model card: 50 steps, guidance 4.5, CFG renormalization on, prompt rewrite on,
1024×1024.

## Memory

fp32: transformer about 25 GB, text encoder about 33 GB, VAE 0.3 GB. A reference/native pair of the text encoder or the
transformer fits, but not both full pipelines at once, so end-to-end tests use the save-then-compare sequence.

## Validation

Random-weight parity for the Qwen2.5-VL text model (hidden states, logits and the rewrite generation) and for the
transformer (tiny configs exact; full-config layouts), prompt-template and token parity, a tiny-snapshot end to end against
`LongCatImagePipeline` with and without prompt rewrite, then real-checkpoint component parity (rewritten prompt string
identical, text encoder, transformer single step at 1024×1024, VAE, scheduler) and end to end at 1024×1024 (prompt rewrite
on; the step count used is recorded), plus the seed test and the gallery sample.

## Task breakdown

1. Qwen2.5-VL text model, LM head and the rewrite generation, with parity tests.
2. Transformer + random-weight parity.
3. Backend (`longcat-image` family), prompt templates, CFG renormalization, catalog entry, tiny-snapshot and real tests.
4. Real-checkpoint parity, docs; PR closes #40.
