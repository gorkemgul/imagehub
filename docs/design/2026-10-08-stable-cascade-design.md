# Stable Cascade design

- Date: 2026-10-08
- Issue: #41
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Two weight sets of one new family `stable_cascade`, text-to-image only, at `full` validation status, with catalog id
`wuerstchen`:

| Model id | Prior (stage C) | Decoder (stage B) |
| --- | --- | --- |
| `stable-cascade` | `prior/` (3.6B) | `decoder/` (1.5B) |
| `stable-cascade-lite` | `prior_lite/` (1B) | `decoder_lite/` (700M) |

Weights: `stabilityai/stable-cascade` (revision `a89f66d459ae653e3b4d4f992a7c3789d0dc4d16`) and
`stabilityai/stable-cascade-prior` (revision `7ca32c21c3b4d4e35bbb94fcfedfb4fa2259bd91`), both by the authors, not gated,
license `stable-cascade-nc-community` (non-commercial). Out of scope: image variation (the prior's CLIP image encoder),
the decoder repository's own text encoder (unused by the combined text-to-image path, see below), the stage-A encoder,
ControlNet, the previewer, bf16 parity.

## Reference

diffusers 0.40.0 `StableCascadeCombinedPipeline` (`pipelines/stable_cascade/pipeline_stable_cascade_combined.py`), which
runs `StableCascadePriorPipeline` and then `StableCascadeDecoderPipeline`; `StableCascadeUNet`
(`models/unets/unet_stable_cascade.py`), `DDPMWuerstchenScheduler`, `PaellaVQModel`
(`pipelines/deprecated/wuerstchen/modeling_paella_vq_model.py`); transformers 5.18.0 `CLIPTextModelWithProjection`,
`CLIPTokenizer`.

## Components

| Part | Status |
| --- | --- |
| CLIP tokenizer and CLIP ViT-bigG text model with projection | exist; the prior calls the text model **with the attention mask**, which is new |
| `StableCascadeUNet` (prior: text, pooled text and image conditioning; decoder: effnet and pixel mappers) | **new**, one module for both stages |
| `DDPMWuerstchenScheduler` (cosine schedule, `s=0.008`, `scaler=1.0`) | **new** |
| `PaellaVQModel` (stage A), decoding only, without quantization | **new** |
| Snapshot with the decoder repository at the root and the prior repository in `prior/` | exists (`subfolders`, Kandinsky 2.2) |

## Pipeline

As `StableCascadeCombinedPipeline.__call__` for text-to-image, with one seeded CPU generator shared by both stages:

1. Prior text encoding: tokenizer padded to 77 with the attention mask; `prompt_embeds = hidden_states[-1]` (before the
   final layer norm), `prompt_embeds_pooled = text_embeds.unsqueeze(1)`. The negative prompt (default `""`) is encoded
   the same way when the guidance scale is above 1.
2. Prior: image embeddings are zeros `(B, 1, 768)` for both branches; latents `randn(B, 16, ceil(H/42.67),
   ceil(W/42.67))`; timesteps `linspace(1, 0, steps + 1)[:-1]`; the prior predicts with the branches concatenated as
   `[text, uncond]` and combines them with `torch.lerp(uncond, text, scale)`; each scheduler step draws noise from the
   generator, including the last.
3. Decoder: conditioned on the prior's latents (`effnet`) and the **prior's** pooled text embedding (the combined
   pipeline passes the prior's prompt embeddings on, so the decoder repository's text encoder is never called);
   latents `randn(B, 4, int(h * 10.67), int(w * 10.67))`; guidance only when the scale is above 1 (with zero effnet
   for the negative branch).
4. `vqgan.decode(latents * 0.3764)`, clamp to `[0, 1]`, then `(x * 255).round()` to uint8.

Defaults (combined pipeline): 512×512, prior 60 steps and guidance 4.0, decoder 12 steps and guidance 0.0.

## Memory

fp32: prior 14.4 GB (lite 4.1 GB), decoder 6.3 GB (lite 2.8 GB), text encoder 2.8 GB, stage A 0.07 GB. Reference and
native fit together on a 96 GB host when the stages are compared one at a time.

## Validation

Random-weight parity for both U-Net configurations (prior and decoder, tiny and with the published layer layout), the
scheduler including generator noise, Paella decoding and the masked text encoder; header checks for all four U-Nets
and stage A from the published safetensors headers (fixtures in `tests/fixtures/stable_cascade_configs/`); a tiny end
to end against the combined pipeline; then real-checkpoint component and end-to-end parity for both weight sets with
the defaults, plus the seed test and the gallery sample.

## Task breakdown

1. CLIP text attention mask and `DDPMWuerstchenScheduler`, with parity.
2. `StableCascadeUNet` with header checks and random-weight parity.
3. `PaellaVQModel` with header check and random-weight parity.
4. Backend `stable_cascade`, both catalog entries, unit tests and the tiny end to end.
5. Real-checkpoint parity and docs; the pull request closes #41.
