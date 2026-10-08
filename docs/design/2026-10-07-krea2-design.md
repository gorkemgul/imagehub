# Krea 2 design

- Date: 2026-10-07
- Issue: #38
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`krea2-turbo` (`krea/Krea-2-Turbo`, revision `98e0fe1`, gated, Krea 2 Community License), text-to-image only, at
`full` validation status, as a new family `krea2`. Out of scope: the single-file `turbo.safetensors` for the official
codebase (the same transformer in another layout), `Krea-2-Raw` (can follow with the same code), LoRA, fp16/bf16 parity.
The weights are published in bfloat16; parity runs in float32 after upcasting, as for the other models.

## Reference

diffusers 0.40.0 `Krea2Pipeline` (`pipelines/krea2/pipeline_krea2.py`), `Krea2Transformer2DModel`,
`AutoencoderKLQwenImage`, `FlowMatchEulerDiscreteScheduler`; transformers 5.18.0 `Qwen3VLModel` and the Qwen2 tokenizer.

## Components

| Part | Status |
| --- | --- |
| Qwen2 tokenizer with the Krea 2 prompt template (system prefix, fixed-length padding, suffix tokens) | tokenizer exists; the template and layout are new |
| Qwen3-VL text model (36 layers, 2560 wide); hidden states of 12 selected layers stacked | exists (`models/qwen3_vl_text.py`, Qwen-Image 2.1); the 3-axis text position ids and layer selection are new |
| `Krea2Transformer2DModel`: 28 blocks, 48 query / 12 key-value heads × 128, 3-axis RoPE (32/48/48, θ 1000), layerwise and refiner text blocks over the 12 text layers | **new** |
| `AutoencoderKLQwenImage` (16 channels, causal 3D convolutions, per-channel latent mean and std) | **new**: the Qwen-Image 2.1 VAE in imagehub is a different variant |
| Flow-match Euler with exponential dynamic shifting | exists |

## Pipeline

As `Krea2Pipeline.__call__` for text-to-image: prompt embeddings and masks, packed 2×2 latents from a seeded CPU
generator, position ids for text and image tokens, sigmas and `mu`, the denoising loop (classifier-free guidance only when
the pipeline applies it; Turbo is distilled and the model card uses guidance 0), unpacking, latent de-normalization, VAE
decode. Defaults from the model card: 8 steps, guidance 0.0, 1024×1024.

## Memory

The transformer has about 13B parameters: about 52 GB per float32 copy, so the reference and native copies never load
together. As for FLUX.1, every real-checkpoint comparison runs the reference first, saves its outputs, frees it, then
runs the native model. A full float32 pipeline alone (transformer, text encoder about 18 GB, VAE) is about 71 GB.

## Validation

Random-weight parity for the transformer and the VAE (tiny configs exact; full-config layouts), prompt-template and token
parity, text-encoder parity with the selected layers, scheduler parity including `mu`, a tiny-snapshot end to end against
`Krea2Pipeline`, then real-checkpoint component parity and end-to-end parity at 1024×1024 with the defaults, plus the
seed test and the gallery sample, all with the save-then-compare sequence.

## Task breakdown

1. Transformer + random-weight parity.
2. Qwen-Image VAE + random-weight parity.
3. Backend (`krea2` family), prompt encoding, catalog entry, tiny-snapshot and real-checkpoint tests.
4. Real-checkpoint parity, docs; PR closes #38.
