# Kandinsky 2.2 design

- Date: 2026-10-08
- Issue: #43
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`kandinsky-2.2` = the decoder `kandinsky-community/kandinsky-2-2-decoder` (revision `9ae140d`) with its prior
`kandinsky-community/kandinsky-2-2-prior` (revision `9fc51ad`), both Apache-2.0 and not gated, text-to-image only, at
`full` validation status, as a new family `kandinsky22`. Out of scope: img2img, inpainting, ControlNet, prior
interpolation, fp16 parity.

## Reference

diffusers 0.40.0 `KandinskyV22CombinedPipeline` (`pipelines/kandinsky2_2/pipeline_kandinsky2_2_combined.py`), which runs
`KandinskyV22PriorPipeline` and then `KandinskyV22Pipeline`; `PriorTransformer`, `UnCLIPScheduler`,
`UNet2DConditionModel`, `VQModel` (MoVQ), `DDPMScheduler`; transformers 5.18.0 `CLIPTextModelWithProjection`,
`CLIPVisionModelWithProjection`, `CLIPTokenizer`.

## Components

| Part | Status |
| --- | --- |
| CLIP tokenizer and CLIP text model with projection (prior) | exist (`tokenization.py`, `models/clip_text.py`; projection checked) |
| CLIP vision model with projection, used by the prior for the zero image embedding of the negative branch | **new** |
| `PriorTransformer` and `UnCLIPScheduler` (sample prediction, `fixed_small_log`, clipping range 10) | **new** |
| Decoder `UNet2DConditionModel` with image conditioning (`addition_embed_type="image"`, `encoder_hid_dim_type="image_proj"`, 8 output channels) | U-Net exists; the image conditioning is new |
| MoVQ `VQModel` decoder (spatially conditioned normalization) | **new** |
| DDPM with linear betas, `clip_sample=true`, epsilon prediction, and the pipeline's handling of the learned-variance output channels | scheduler exists (HunyuanDiT); the options are new |
| Two repositories in one catalog entry | **new**: the snapshot holds the decoder at its root and the prior repository in a subfolder |

## Pipeline

As `KandinskyV22CombinedPipeline.__call__` for text-to-image: prior with classifier-free guidance (negative branch from the
zero image embedding), its UnCLIP sampling from a seeded CPU generator, then the decoder with image-embedding conditioning
and classifier-free guidance, DDPM steps, MoVQ decode. Defaults: prior 25 steps and guidance 4.0, decoder 100 steps and
guidance 4.0, 512×512.

## Memory

fp32: prior about 4.1 GB, text encoder about 2.8 GB, image encoder about 3.7 GB, decoder U-Net about 5 GB, MoVQ 0.3 GB.
Reference and native fit together.

## Validation

Random-weight parity for the prior transformer, the CLIP vision model, the image-conditioned U-Net and MoVQ; scheduler
parity (UnCLIP and the DDPM options) including generator noise; a tiny end to end against the combined pipeline; then
real-checkpoint component parity and end-to-end parity with the defaults, plus the seed test and the gallery sample.

## Task breakdown

1. Prior: CLIP vision with projection, `PriorTransformer`, `UnCLIPScheduler` + parity.
2. Decoder: image-conditioned U-Net, MoVQ, DDPM options + parity.
3. Catalog support for a second repository, backend (`kandinsky22`), catalog entry, tiny and real tests.
4. Real-checkpoint parity, docs; PR closes #43.
