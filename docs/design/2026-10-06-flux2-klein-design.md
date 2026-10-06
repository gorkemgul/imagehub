# FLUX.2 klein design

- Date: 2026-10-06
- Issue: #33
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`flux2-klein-4b` (`black-forest-labs/FLUX.2-klein-4B`, revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`, Apache-2.0), text-to-image only, at `full` validation
status, as a new family `flux2`; `flux2-klein-9b` (gated) follows with the same code. Out of scope:
image conditioning and editing (`image` input, the inpaint and KV-cache pipelines), FLUX.2-dev (32B,
beyond fp32 parity on the 96 GB machine), fp16/bf16 parity.

## Reference

diffusers 0.40.0 `Flux2KleinPipeline` (`pipelines/flux2/pipeline_flux2_klein.py`),
`Flux2Transformer2DModel`, `AutoencoderKLFlux2`, `FlowMatchEulerDiscreteScheduler`; transformers 5.18.0
`Qwen3ForCausalLM` and the Qwen2 tokenizer. Every numeric detail follows the installed source.

## Components

| Part | Status |
| --- | --- |
| Qwen2 tokenizer + chat template (`enable_thinking=False`) | exists (`tokenization_qwen`); the prompt string is pinned by a parity test |
| Qwen3 text encoder; hidden states of layers 9, 18, 27 stacked to 3 × 2560 = 7680 | exists (`models/qwen3.py`); the layer selection and stacking are new |
| `Flux2Transformer2DModel` (5 double-stream and 20 single-stream blocks, 24 heads × 128, 4-axis RoPE 32/32/32/32 with θ 2000, no guidance embedding for klein) | **new** |
| `AutoencoderKLFlux2` (32 latent channels, 2×2 patchified latents, batch-norm running statistics used to normalize latents, quant convs) | **new** |
| Flow-match Euler with dynamic shifting and the pipeline's `compute_empirical_mu` | scheduler exists; the empirical `mu` is new |

## Pipeline

As `Flux2KleinPipeline.__call__` for text-to-image: prompt embeddings and text ids, latents from a
seeded CPU generator in the patchified 128-channel layout, latent ids, sigmas and `mu`, the denoising
loop (with classifier-free guidance only when the pipeline applies it for this checkpoint:
`is_distilled` is true), unpacking, batch-norm de-normalization, unpatchify, VAE decode. Defaults
from the model card: 4 steps, guidance 1.0, 1024×1024.

## Memory

fp32: transformer ≈ 15.5 GB and text encoder ≈ 16 GB per copy; reference and native fit together.

## Validation

Random-weight parity for the transformer and the VAE (tiny configs exact; full-config layouts on the
meta device), prompt-string and token parity, scheduler parity including `mu`, a tiny-snapshot end to
end against `Flux2KleinPipeline`, then real-checkpoint component parity and end-to-end parity at
1024×1024 with the defaults, plus the seed test.

## Task breakdown

1. Transformer + random-weight parity.
2. VAE + random-weight parity.
3. Backend (`flux2` family), prompt encoding, empirical `mu`, catalog entry, tiny-snapshot tests,
   real-checkpoint tests.
4. Real-checkpoint parity, docs; PR closes #33 (klein 4B; 9B as a follow-up in the same issue).
