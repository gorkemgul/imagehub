# Kandinsky 5 design

- Date: 2026-10-12
- Issue: #116
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

One weight set of a new family `kandinsky5`, text-to-image, at `full` validation status, catalog id `kandinsky-5-t2i-lite`,
by the authors, not gated, license from the model card:

| Model id | Repository | Revision | Transformer |
| --- | --- | --- | --- |
| `kandinsky-5-t2i-lite` | `kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers` | `25da1e82e6adefdbc83c02a71f762287c5e471a4` | 6.2B, one bfloat16 file |

Out of scope: image-to-image and video pipelines, sparse attention modes (`nablaT_v2`), the pretrain checkpoint, reduced-
precision parity beyond upcasting the published weights to float32.

## Reference

diffusers 0.40.0 `Kandinsky5T2IPipeline` (`pipelines/kandinsky5/pipeline_kandinsky_t2i.py`),
`Kandinsky5Transformer3DModel` (`models/transformers/transformer_kandinsky.py`), `AutoencoderKL` (FLUX VAE),
`FlowMatchEulerDiscreteScheduler`; transformers 5.18.0 `Qwen2_5_VLForConditionalGeneration`, `CLIPTextModel`.

## Components

| Part | Status |
| --- | --- |
| Qwen2.5-VL 7B text model (`imagehub.models.qwen2_5_vl_text`) | **existing** |
| CLIP-L text encoder (`imagehub.models.clip_text`) | **existing**, loaded from a full `CLIPModel` checkpoint by skipping vision/projection tensors |
| FLUX VAE (`imagehub.models.autoencoder_kl`) | **existing** |
| Flow-match Euler scheduler (`imagehub.schedulers.flow_match_euler`) | **existing**, `shift=5.0` |
| Qwen2 / Qwen2.5-VL byte-level BPE tokenizer (`imagehub.tokenization_qwen`) | **existing** |
| CLIP BPE tokenizer (`imagehub.tokenization`) | **existing** |
| `Kandinsky5Transformer3DModel` (text encoder blocks, visual decoder blocks, 3D/1D RoPE, AdaLN modulation) | **new** |

## Pipeline

As `Kandinsky5T2IPipeline.__call__` for one prompt: the fixed system prompt is prepended, Qwen2.5-VL tokenizes it and the
first 41 template tokens are dropped; CLIP tokenizes the raw prompt to 77 tokens and returns the EOS hidden state as the
pooled embedding; classifier-free guidance with `guidance_scale` when it exceeds 1.0 and a negative prompt is supplied;
resolution list `[(1024,1024), (640,1408), (1408,640), (768,1280), (1280,768), (896,1152), (1152,896)]` (the request is
snapped to the closest aspect ratio); 50 steps by default; latents `randn(B, 1, H/8, W/8, 16)`; the transformer predicts the
velocity, the scheduler denoises, the VAE decodes the reshaped and scaled latents; uint8 as diffusers. The transformer
keeps time/modulation modules in float32 internally even when weights are loaded in bfloat16; the native model reproduces
that numerics pattern.

## Memory

fp32: transformer ~6.2B params (~24.8 GB), Qwen2.5-VL 7B (~28 GB), CLIP-L (~0.5 GB), VAE (~0.5 GB). The reference end-to-end
run is shortened and the reference is freed before loading the native model.

## Validation

Tokenizer parity on fixed prompts; random-weight parity for the transformer (tiny and published config at small spatial
sizes), text encoder outputs, VAE decode; scheduler parity; tiny end-to-end against `Kandinsky5T2IPipeline` (defaults,
guidance on/off, non-square, two seeds); real-checkpoint component parity (shortened steps) and seed test, skipping cleanly
without the snapshot.

## Task breakdown

1. `Kandinsky5Transformer3DModel` with header checks and random-weight parity.
2. Backend `kandinsky5`, catalog entry and registry wiring.
3. Unit tests and tiny end-to-end reference parity.
4. Real-checkpoint test file, docs update and validation record placeholders.
