# Lumina-Next design

- Date: 2026-10-10
- Issue: #92
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

One weight set of a new family `lumina-next`, text-to-image, catalog id `lumina-next`, by the authors, not gated,
Apache-2.0 (model card):

| Model id | Repository | Revision | Transformer |
| --- | --- | --- | --- |
| `lumina-next` | `Alpha-VLLM/Lumina-Next-SFT-diffusers` | `0ee5ec90043acf5cb41fe96274af36eb7fad8d95` | 1.75B, one bfloat16 file |

The text encoder (Gemma 2B, two bfloat16 shards) and the transformer are published in bfloat16 only; the autoencoder
(the SDXL one) is float32. No float32 copy exists, so the float32 reference upcasts the bfloat16 tensors exactly.
Lumina-Image 2.0 is a different architecture and stays in the family `lumina2`.

Out of scope: `clean_caption` (needs `beautifulsoup4` and `ftfy`; the pipeline falls back to lower-casing and stripping
without them, which is what the backend does), the pipeline's `max_sequence_length` argument (it is accepted but
ignored; the prompt length is fixed at 256), LoRA, reduced-precision parity, batches, caller-supplied latents.

## Reference

diffusers 0.40.0 `LuminaPipeline` (`pipelines/lumina/pipeline_lumina.py`), `LuminaNextDiT2DModel`
(`LuminaAttnProcessor2_0`, `LuminaFeedForward`, `LuminaRMSNormZero`, `LuminaLayerNormContinuous`,
`get_2d_rotary_pos_embed_lumina`), `AutoencoderKL`, `FlowMatchEulerDiscreteScheduler` (static shift 1.0); transformers
5.18.0 `GemmaModel`, `GemmaTokenizerFast`.

## Components

| Part | Status |
| --- | --- |
| Gemma tokenizer | exists; the published `tokenizer.json` appends the end token, which the existing tokenizer already follows from `tokenizer_config.json` |
| Gemma (first generation) 2B text model: no extra norms, no sliding window, one key-value head | **new** `models/gemma.py`, reusing the Gemma-2 norm, MLP and attention modules (the attention scale is `head_dim ** -0.5` through `query_pre_attn_scalar`) |
| `LuminaNextDiT2DModel` (gated text cross-attention beside self-attention, 2D RoPE, pooled-caption modulation) | **new** |
| `AutoencoderKL` (SDXL, quant convolutions, no shift) | exists |
| Flow-match Euler with a static shift | exists; the default timestep path (no custom sigmas) is checked against diffusers |

The published configuration says `hidden_act="gelu"`; transformers maps it to the tanh approximation, and so does the
native model.

## Pipeline

As `LuminaPipeline.__call__` for one prompt:

- The prompt is lower-cased and stripped; tokenization to 256 tokens including the start and end tokens, padded on the
  right to a multiple of eight; the masked second-to-last hidden state is the conditioning. The negative prompt (default
  `""`) is used as written and padded or truncated to the prompt's length. It is only encoded when guidance exceeds 1.
- Positive and negative inputs share one batch (`[prompt, negative]`), as in the reference, so the CPU kernels see the
  same shapes.
- Timesteps come from `set_timesteps(steps)` without custom sigmas. The model timestep is `1 - t / 1000`.
- Time-aware scaled RoPE: with `scaling_factor = sqrt(width * height / 1024²)`, the rotary table (384×384, computed in
  float32) uses `linear_factor = scaling_factor` and `ntk_factor = 1` while the reversed timestep is below
  `scaling_watershed`, and the opposite afterwards. The comparison is made in float32 as in the reference.
- Proportional attention: self-attention logits use the scale `sqrt(log_4096(tokens)) * head_dim ** -0.5`; the
  cross-attention keeps the plain scale.
- The model returns eight channels; the first four are kept. Guidance is applied to the first three channels only
  (`uncond + g * (cond - uncond)`), the fourth stays conditional. The prediction is negated before the scheduler step.
- Latents `randn(1, 4, H/8, W/8)`; `vae.decode(latents / 0.13025)`; uint8 as diffusers.

## Memory

fp32: transformer 7.0 GB, Gemma 2B 10.0 GB, autoencoder 0.3 GB. Reference and native fit together.

## Validation

Random-weight parity for the Gemma model (tiny and the published attention geometry, padding sides, every entry of the
hidden-state tuple), the transformer (tiny and published widths, both attention scalings, three rotary scalings) and the
rotary tables; tiny end to end against `LuminaPipeline` (guidance on and off, negative prompts, non-square sizes, five
watershed values, proportional attention off, seeds); tokenizer and scheduler checks; real-checkpoint component and end
to end tests plus a shortened watershed test and the seed test. Results are recorded after the real-checkpoint run.

## Task breakdown

1. Config fixtures and the Gemma model with parity.
2. The transformer with header and parity tests.
3. Backend, catalog entry, registry, unit tests and the tiny end to end.
4. Real-checkpoint tests and docs; the pull request closes #92.
