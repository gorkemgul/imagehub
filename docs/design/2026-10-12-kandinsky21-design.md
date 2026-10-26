# Kandinsky 2.1 design

- Date: 2026-10-12
- Issue: #115
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

One text-to-image checkpoint, `kandinsky-2-1`, in the new `kandinsky21` family. The decoder is
`kandinsky-community/kandinsky-2-1` at revision
`04b366659663bf60deacb96fd3e38f1cb39a3be4`; the prior is
`kandinsky-community/kandinsky-2-1-prior` at revision
`b726af62d69111bbf62b5d71bb9d90d9867fbce4`. Both are author repositories with Apache-2.0
weights and publish float32 safetensors.

Out of scope: image-to-image, inpainting, interpolation, ControlNet, safety checking, watermarking,
and fp16/bfloat16 parity.

## Reference

diffusers 0.40.0 `KandinskyPriorPipeline`, `KandinskyPipeline`, and
`KandinskyCombinedPipeline`; transformers 5.18.0 XLM-RoBERTa and CLIP components.

## Components

| Part | Status |
| --- | --- |
| XLM-RoBERTa tokenizer from `tokenizer.json` | new pure-Python implementation |
| XLM-RoBERTa encoder and MultilingualCLIP projection | new native modules |
| PriorTransformer and CLIP text/image encoders | reused |
| Text-image-conditioned U-Net | shared U-Net extended with published conditioning paths |
| MoVQ decoder | reused from Kandinsky 2.2 |
| UnCLIP prior scheduler | reused |
| DDIM decoder scheduler | reused; accepts the pipeline generator argument |

## Pipeline

The prompt is encoded by the CLIP text encoder for the diffusion prior. The prior runs for 25
steps with guidance 4.0 and produces a 768-dimensional image embedding. With no explicit negative
prompt, the negative image embedding comes from a zero image; an explicit negative prompt is run
through the prior as a second prompt. The decoder uses the MultilingualCLIP encoder at the
reference pipeline's fixed length of 77, conditions the text-image U-Net, runs DDIM for 100 steps,
and decodes with MoVQ. The combined reference applies `negative_prompt` only in the prior; decoder
classifier-free guidance uses empty text.

## Memory

The complete model is about 3.3B parameters. CPU float32 component checks load reference and native
components together; end-to-end validation saves and frees the reference result before loading the
native pipeline.

## Validation

Published safetensors headers are checked on the meta device. Random-weight tests compare the new
text encoder, text-image U-Net, DDIM trajectory, tokenizer, and a tiny combined pipeline exactly.
The tiny pipeline covers guidance on and off, an explicit negative prompt, non-square output, and
two seeds. The real-checkpoint suite covers all six weight-bearing components, default end-to-end
generation, and seed parity on CPU float32.
