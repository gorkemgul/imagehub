# CogView4 design

- Date: 2026-10-07
- Issue: #42
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`cogview4-6b` (`zai-org/CogView4-6B`, revision `63a52b7`, Apache-2.0, not gated), text-to-image only, at `full`
validation status, as a new family `cogview4`. Out of scope: CogView4 control pipelines, LoRA, fp16/bf16 parity. The
weights are published in bfloat16; parity runs in float32 after upcasting.

## Reference

diffusers 0.40.0 `CogView4Pipeline` (`pipelines/cogview4/pipeline_cogview4.py`), `CogView4Transformer2DModel`,
`AutoencoderKL`, `FlowMatchEulerDiscreteScheduler`; transformers 5.18.0 `GlmModel` and `PreTrainedTokenizerFast` from
`tokenizer.json`.

## Components

| Part | Status |
| --- | --- |
| GLM-4 tokenizer from `tokenizer.json` with the pipeline's padding to a multiple of 16 | **new** (BPE reader may be shared with the existing byte-level BPE tokenizers) |
| `GlmModel` (GLM-4-9B: 40 layers, 4096 wide, 32 query / 2 key-value heads, partial rotary 0.5, attention bias, RMSNorm eps 1.5625e-7), hidden state used by the pipeline | **new** |
| `CogView4Transformer2DModel`: 28 blocks, 32 heads × 128, 2D RoPE (256/256), timestep + size conditioning | **new** |
| 16-channel `AutoencoderKL` (block widths 128/512/1024/1024, 3 layers per block, no mid-block attention, scaling 1.0, shift 0.0) | exists if `models/autoencoder_kl.py` covers this config; otherwise extended |
| Flow-match Euler with linear dynamic shifting and the pipeline's `calculate_shift` | scheduler exists; the pipeline's sigmas and `mu` are checked |

## Pipeline

As `CogView4Pipeline.__call__` for text-to-image: prompt tokenization and encoding exactly as the pipeline calls the
text encoder, classifier-free guidance with an empty negative prompt, latents from a seeded CPU generator, original size,
target size and crop conditioning, sigmas and `mu`, the loop, VAE decode. Defaults from the pipeline and model card: 50
steps, guidance 5.0, 1024×1024.

## Lessons carried over

From Krea 2: call the reference text encoder exactly as the pipeline does (transformers' default `use_cache=True`, which
stores keys and values as contiguous copies), follow transformers' rule for grouped-query attention in SDPA (masked GQA
on hosts with MPS and torch ≥ 2.13), and match where each loader places tensors in memory (copied versus memory-mapped)
and the order of floating-point operations in pre- and post-processing.

## Memory

fp32: transformer about 25 GB, text encoder about 36 GB, VAE about 0.8 GB. Component pairs fit; end-to-end tests use the
save-then-compare sequence.

## Validation

Random-weight parity for the GLM text model and the transformer (tiny configs exact; full-config layouts), tokenizer
parity (fixed prompts and fuzz), a tiny-snapshot end to end against `CogView4Pipeline`, then real-checkpoint component
parity and end-to-end parity at 1024×1024 (reduced step count recorded), plus the seed test and the gallery sample.

## Task breakdown

1. GLM tokenizer and text model + parity.
2. Transformer + random-weight parity.
3. Backend (`cogview4` family), catalog entry, tiny-snapshot and real tests.
4. Real-checkpoint parity, docs; PR closes #42.
