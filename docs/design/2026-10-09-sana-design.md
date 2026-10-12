# Sana design

- Date: 2026-10-09
- Issue: #20
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Four weight sets of the `sana` family, text-to-image, at `full` validation status, catalog id `sana`, by the authors,
not gated, Apache-2.0 (model cards):

| Model id | Repository | Revision | Transformer |
| --- | --- | --- | --- |
| `sana-1600m-1024` | `Efficient-Large-Model/Sana_1600M_1024px_diffusers` | `ac0da2ff55fbe434795be0dce883042e4d49e2fc` | 1.6B, two float32 shards |
| `sana-600m-1024` | `Efficient-Large-Model/Sana_600M_1024px_diffusers` | `28f3af7689de15f3883d5863059a2fca0aa9b829` | 0.6B, one float32 file |
| `sana-sprint-1.6b-1024` | `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` | `19683c58b7ea290e55cedd8950ae1d86ada7ef96` | 1.6B, SCM scheduler, guidance embeds |
| `sana-1.5-4.8b-1024` | `Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers` | `9468102c3cebb657f8c4b5f1e5a71e989a15f10d` | 4.8B, two float32 shards, qk norm |

The 1600M repository also holds an unsharded float16 transformer file with legacy tensor names; diffusers loads the
float32 shards through the index, so the catalog lists the shards and the index only. Both repositories share the same
text encoder (Gemma-2 2B, stored in bfloat16), tokenizer, DC-AE and scheduler configuration; the 600M entry uses its own
repository for every file.

Out of scope: 2K/4K checkpoints, ControlNet, PAG, `clean_caption` (needs external packages),
reduced-precision parity.

## Reference

diffusers 0.40.0 `SanaPipeline` (`pipelines/sana/pipeline_sana.py`), `SanaSprintPipeline`
(`pipelines/sana/pipeline_sana_sprint.py`), `SanaTransformer2DModel`, `AutoencoderDC`,
`DPMSolverMultistepScheduler` (flow sigmas, `flow_prediction`, `flow_shift` 3.0, midpoint, order 2),
`SCMScheduler` (`scheduling_scm`, trigflow, `sigma_data` 0.5); transformers 5.18.0 `Gemma2Model`,
`GemmaTokenizerFast`.

## Components

| Part | Status |
| --- | --- |
| Gemma tokenizer (sentencepiece BPE from `tokenizer.json`, byte fallback, special tokens) | **new** |
| Gemma-2 2B text model (soft-capped attention, alternating sliding-window layers, `(1 + w)` RMSNorm, scaled embeddings), with an attention mask | **new** |
| `SanaTransformer2DModel` (linear attention, GLUMBConv feed-forward, cross-attention to the masked text, AdaLN-single; optional guidance embedding and qk norm) | **new** |
| `AutoencoderDC` (DC-AE, 32× compression, 32 latent channels), decoding | **new** |
| DPM-Solver++ with flow sigmas and `flow_prediction` | scheduler exists; the flow options are checked or added |
| SCM scheduler with trigflow parameterization | **new** |

## Pipeline

As `SanaPipeline.__call__` for one prompt: the "complex human instruction" is prepended to the prompt (not to the
negative prompt); tokenization to `max_sequence_length + len(instruction tokens) - 2` and selection of the first token
plus the last 299; the masked Gemma-2 last hidden state as conditioning; classifier-free guidance 4.5 with the negative
prompt `""`; resolution binning on (the request is snapped to the closest aspect-ratio bin, generated there, and the
decoded image is resized and centre-cropped back to the request); 20 steps; 1024×1024; latents `randn(B, 32, H/32, W/32)`;
`vae.decode(latents / scaling_factor)`; uint8 as diffusers.

## Memory

fp32: transformer 6.4 GB (600M: 2.4 GB), Gemma-2 2B 10.5 GB, DC-AE 1.3 GB. Reference and native fit together.

## Validation

Tokenizer parity on fixed prompts plus a fuzz test; random-weight parity for Gemma-2, the transformer and DC-AE (tiny
and with the published configs at small spatial sizes); scheduler parity for DPM-Solver++ and SCM; tiny end to end
against `SanaPipeline` and `SanaSprintPipeline` (binning on and off, guidance on and off, non-square); real-checkpoint
component and end-to-end parity for the 1600M and 600M weight sets, plus the seed test. Sprint and 1.5 are validated
through scheduler/config parity and tiny-snapshot end-to-end parity.

## Task breakdown

1. Gemma tokenizer and Gemma-2 text model, with parity.
2. `SanaTransformer2DModel` and `AutoencoderDC`, with header checks and random-weight parity.
3. Scheduler flow options, backend `sana`, both catalog entries, unit tests and the tiny end to end.
4. Real-checkpoint parity and docs; the pull request closes #20.
