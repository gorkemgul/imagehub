# Ideogram 4 design

- Date: 2026-10-08
- Issue: #39
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`ideogram4-fp8` (`ideogram-ai/ideogram-4-fp8`, revision `ee79a72`, gated, Ideogram 4 Non-Commercial License), text-to-image
only, at `full` validation status, as a new family `ideogram4`. Out of scope: the nf4 releases (bitsandbytes, CUDA only),
magic prompt and safety moderation (both call external paid APIs), LoRA.

## Weights

Only quantized weights are published. The fp8 release stores every large `Linear` weight of the transformers and of the
text encoder as `float8_e4m3fn` with a per-output-channel `float32` `weight_scale`; other tensors are bfloat16. The
official code keeps the fp8 weights and dequantizes them at every forward pass (`weight.to(dtype) * weight_scale.to(dtype)
.unsqueeze(1)`, then `F.linear`), so it needs no fp8 hardware and runs on CPU. imagehub does the same computation in the
same order, so a float32 run is bit-identical while the weights stay small in memory.

## Reference

The model card lists no diffusers support for the fp8 release, and the nf4 release needs CUDA, so the reference is the
official implementation: `ideogram-oss/ideogram4` at commit `990fe1c`, installed in its own environment
(`.venv-ref-ideogram4`, with transformers 5.18.0, as for Qwen-Image 2.1), run on CPU with compute dtype float32 and fp8
weights. The validation pages state that parity is against the official code, not diffusers.

## Components

| Part | Status |
| --- | --- |
| Qwen2 tokenizer with the official prompt format and the local caption verifier | tokenizer exists; format and verifier are new |
| Qwen3-VL text model (36 layers, 4096 wide, 32 / 8 heads) with fp8 linears; the official code's hidden-state selection | model exists (`models/qwen3_vl_text.py`); fp8 linears and the selection are new |
| `Ideogram4Transformer`: single-stream DiT, 34 layers, 18 heads × 256, adaLN, multimodal RoPE (24/20/20), fp8 linears | **new** |
| Unconditional transformer (same architecture, separate weights) used for classifier-free guidance | **new** (same class) |
| Autoencoder (FLUX.2-style, 32 channels with batch-norm latent normalization) as the official code builds it | compared with `models/autoencoder_kl_flux2.py`; reused only where numerics are identical |
| Official sampler (`scheduler.py`, presets `V4_QUALITY_48` default, `V4_DEFAULT_20`, `V4_TURBO_12`) with a per-step guidance schedule | **new** |

## Pipeline

As the official `Ideogram4Pipeline.__call__` with magic prompt and safety off: caption verification, prompt encoding,
noise from a seeded generator, the sampler preset's step intervals and guidance schedule, conditional and unconditional
transformer passes, latent de-normalization and decode. Defaults: preset `V4_QUALITY_48`, 1024×1024.

## Memory

With fp8 weights kept in memory: transformers about 9.3 GB each, text encoder about 8.8 GB, plus activations. Reference
and native fit together for components; end-to-end tests run the reference first and free it.

## Validation

fp8 linear parity (dequantization order), random-weight parity for the transformer (tiny configs exact; full-config
layout), the sampler's intervals and guidance schedule, a tiny end to end against the official pipeline, then
real-checkpoint component parity and end-to-end parity at 1024×1024 with a reduced preset (`V4_TURBO_12`, recorded), the
seed test and the gallery sample.

## Task breakdown

1. fp8 linear, transformer, sampler + parity against the official code.
2. Text encoder with fp8 linears, prompt format, caption verifier, autoencoder check.
3. Backend (`ideogram4` family), catalog entry, tiny end to end and real tests.
4. Real-checkpoint parity, docs; PR closes #39.
