# ERNIE-Image design

- Date: 2026-10-10
- Issue: #88
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

One weight set of a new family `ernie-image`, text-to-image, catalog id `ernie-image`, by the authors, not gated,
Apache-2.0 (model card and the sibling catalog):

| Model id | Repository | Revision | Transformer |
| --- | --- | --- | --- |
| `ernie-image` | `baidu/ERNIE-Image` | `5346b31d68c9c23758ba56ef8be5e9dc174c7f99` | 8.03B, two bfloat16 shards |

Only bfloat16 files are published, so parity is measured on the weights upcast to float32. The catalog lists the
transformer shards and index, the text encoder (a full `Mistral3Model`, 7.7 GB), the tokenizer, the VAE and the scheduler.

Out of scope: the prompt enhancer (`pe`, `pe_tokenizer`), image input, LoRA, reduced-precision parity. The
reference pipeline enables the enhancer by default (`use_pe=True`); it rewrites the prompt by sampling a 3B causal
language model with the global random generator (temperature 0.6, top-p 0.95), so it cannot be tied to the request seed.
imagehub reproduces the pipeline with `use_pe=False` and its snapshot omits the 7.7 GB `pe` weights.

## Reference

diffusers 0.40.0 `ErnieImagePipeline` (`pipelines/ernie_image/`), `ErnieImageTransformer2DModel`,
`AutoencoderKLFlux2`, `FlowMatchEulerDiscreteScheduler`; transformers 5.18.0 `Mistral3Model` /
`Ministral3Model` and `TokenizersBackend`.

## Components

| Part | Status |
| --- | --- |
| Byte-level BPE tokenizer from `tokenizer.json` (isolated Unicode regex split, `ignore_merges`, plain added tokens, `<s>` template, truncation to 2048) | **new** |
| Ministral 3 decoder (GQA, YaRN rotary, position-dependent query scale), loaded from the language model of the `Mistral3Model` file; vision tensors are dropped | **new** |
| `ErnieImageTransformer2DModel` (single stream, shared AdaLN, 3-axis RoPE, ragged text via masking) | **new** |
| FLUX.2 VAE (`AutoencoderKLFlux2`), latent batch-norm statistics | exists |
| Flow-match Euler with a static shift and caller-supplied sigmas | exists |

## Pipeline

As `ErnieImagePipeline.__call__` for one prompt with `use_pe=False`: the prompt is tokenized with the `<s>` prefix and
encoded alone; the second-to-last hidden state of the language model is the conditioning. Guidance above 1.0 also
encodes the negative prompt (default `""`, which is just `<s>`) and runs both in one batch (negative first).
Latents `randn(B, 128, H/16, W/16)` in the model dtype from a CPU generator; sigmas `linspace(1, 0, steps + 1)[:-1]`
with shift 4.0; the transformer gets float timesteps `sigma * 1000`; `pred = uncond + g * (cond - uncond)`. Decoding
de-normalises with the VAE batch-norm statistics using epsilon `1e-5` (the pipeline's constant, not the configured
`1e-4`), unpatchifies and decodes.

## Memory

fp32: transformer 32 GB, text encoder 13.7 GB (language model only), VAE 0.3 GB.

## Validation

Tokenizer parity on fixed prompts plus a fuzz test (also run over every Unicode code point against the real
`tokenizer.json`); random-weight parity for the transformer and the text model (tiny and published-style, with YaRN
positions beyond the original window); tiny end to end against `ErnieImagePipeline` (guidance on and off, non-square,
two seeds, batch of two); real-checkpoint header, tokenizer, text encoder, one transformer step, VAE decode,
scheduler, end to end and seed tests. The end-to-end runs use 4 steps and save the reference result before loading
the native model.
