# CogView3-Plus design

- Date: 2026-10-10
- Issue: #89
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

One weight set of a new family `cogview3`, text-to-image, catalog id `cogview3-plus`, by the authors, not gated,
Apache-2.0 (model card tag; the awesome catalog lists Apache-2.0 for the weights):

| Model id | Repository | Revision | Transformer |
| --- | --- | --- | --- |
| `cogview3-plus-3b` | `zai-org/CogView3-Plus-3B` | `5d70e40732ac0efac98524c51a7fa9c82707f1e5` | 2.85B, two bfloat16 shards |

The repository publishes only these weights: the transformer and the VAE are bfloat16 (there is no float32
copy), the T5-XXL text encoder is float32 in four shards. Parity is measured in float32 after upcasting the
bfloat16 tensors, as diffusers does with `torch_dtype=torch.float32`. The tokenizer is the SentencePiece model
(`spiece.model`) with `tokenizer_config.json`, `added_tokens.json` and `special_tokens_map.json`; there is no
`tokenizer.json`.

Out of scope: `CogVideoXDPMScheduler`, stochastic sampling (`eta` above 0), several images per prompt, prompt
embeddings passed in directly, reduced-precision parity.

## Reference

diffusers 0.40.0 `CogView3PlusPipeline` (`pipelines/cogview3/pipeline_cogview3plus.py`),
`CogView3PlusTransformer2DModel`, `AutoencoderKL`, `CogVideoXDDIMScheduler`; transformers 5.18.0
`T5EncoderModel` and `T5Tokenizer`.

## Components

| Part | Status |
| --- | --- |
| T5-XXL encoder (`imagehub/models/t5.py`) | existing; header and parity are checked for this config |
| T5 tokenizer built from `spiece.model` (`imagehub/tokenization_t5.py`) | existing; parity is checked against this repository's tokenizer files |
| `AutoencoderKL` (16 channels, no quant convolutions, no mid-block attention, `layers_per_block` 3) | existing; same layout as the CogView4 VAE |
| `CogView3PlusTransformer2DModel` (joint text/image attention without RoPE, AdaLN-Zero with 12 chunks, a fixed 2D sine-cosine position buffer, SDXL-style size conditions) | **new**; reuses only the timestep/size embedding of the CogView4 module, which is an exact match |
| `CogVideoXDDIMScheduler` (shifted SNR, zero terminal SNR, trailing spacing, v-prediction, `eta` 0) | **new** (`imagehub/schedulers/ddim_cogvideox.py`) |

CogView3-Plus differs from CogView4 in the text encoder, the attention (no RoPE, query/key LayerNorm with
epsilon 1e-6), the position embedding (added to the patch embedding), the modulation layer (the SiLU is inside
the norm and the chunk order differs), one shared feed-forward over the concatenated tokens, and the scheduler
(DDIM, not flow matching). Only the embedding classes are shared.

## Pipeline

As `CogView3PlusPipeline.__call__` for one prompt: `max_sequence_length` 224 tokens, padded with `<pad>` and
truncated keeping the final `</s>`; the T5 encoder is called without an attention mask; guidance is applied when
`guidance_scale > 1` (default 5.0); without a negative prompt the unconditional features are zeros (an explicit
negative prompt, even `""`, is encoded); the unconditional and conditional rows go through the transformer as
one batch of two; `original_size` defaults to `(height, width)`, `target_size` is `(height, width)` and the crop
is `(0, 0)`; 50 steps, 1024×1024, latents `randn(1, 16, H/8, W/8)`; `noise_pred.float()`; the scheduler step is
taken with `eta` 0; `vae.decode(latents / scaling_factor)`; uint8 as diffusers. Width and height must be
multiples of 16 (the reference accepts multiples of 8 but fails in the patch embedding otherwise) and at most
2048 (`pos_embed_max_size` 128 patches).

## Memory

fp32: T5-XXL encoder 19 GB, transformer 11.4 GB, VAE 0.8 GB. The reference pipeline holds all of them at once, so
the real-checkpoint end-to-end tests compute and save the reference result and free it before loading the native
model.

## Validation

Scheduler parity (timesteps, tables and trajectories for all variants), tokenizer fuzz against the published
SentencePiece model, random-weight transformer parity (tiny, wider and published configuration), tiny end to end
against `CogView3PlusPipeline`, header checks, and the real-checkpoint file `tests/reference/test_ref_cogview3_plus.py`
(tokens, text encoder, transformer at two timesteps, VAE decode/encode, scheduler, end to end and seed at 8 steps
because of the 7.6B parameters in total).

## Task breakdown

1. Config and header fixtures; scheduler with parity.
2. Transformer with header and random-weight parity.
3. Backend, catalog entry, registry, unit tests and the tiny end to end.
4. Tokenizer parity, real-checkpoint tests and docs; the pull request closes #89.
