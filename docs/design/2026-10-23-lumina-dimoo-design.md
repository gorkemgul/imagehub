# Lumina-DiMOO design

- Date: 2026-10-23
- Issue: #179
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`lumina-dimoo` is the author checkpoint `Alpha-VLLM/Lumina-DiMOO` at revision
`f014549e17679e41b119959754cdd9320af9bdc8`, licensed Apache-2.0. The reference is the official
`LLaDAForMultiModalGeneration` and `generate_image` implementation at
`Alpha-VLLM/Lumina-DiMOO@81344e60545ae8d70f11039d3450717d974f831b`.

The repository publishes an 8B LLaDA checkpoint in four BF16 shards and a full-precision
aMUSEd `VQModel`. Text-to-image generation is in scope. Image editing and understanding, the
experimental max-logit cache, batching, training, and reduced-precision parity are out of scope.

## Components

| Part | Status |
| --- | --- |
| NFC byte-level BPE tokenizer and added multimodal tokens | reuse `Qwen2Tokenizer` |
| Bidirectional LLaDA Llama block, SwiGLU MLP, RoPE and vocabulary head | new |
| aMUSEd VQ decoder, 16x compression and 8,192-entry codebook | reuse `VQModel` |
| Cosine MaskGIT sampling with Gumbel sampling and classifier-free guidance | new backend loop |

## Pipeline

The backend formats the author system/user prompt, appends `<answer><image>`, a rectangular grid
of mask tokens with newline separators, and closing image/answer tokens. Each step predicts all
masked image positions, applies classifier-free guidance when requested, samples with Gumbel-max,
and remasks the least-confident samples according to the cosine schedule. The final image-token
ids are offset by 126356, decoded by the VQ model, clipped to `[0, 1]`, and converted to uint8.

Defaults are 1,024x1,024, 64 steps, guidance 4.0 and temperature 1.0. Tests and validation use
short runs and small images because each LLaDA forward covers the complete image-token sequence.

## Memory and validation

The language model is stored in BF16 (about 15 GB) and the VQ model in float32 (about 0.6 GB).
Real comparisons run the official reference in a subprocess and free it before native loading.
Validation covers published headers, tokenizer output, random-weight LLaDA and VQ parity, exact
sampling, a tiny end-to-end snapshot, real component checks, and separate seed-0 and seed-42
public-pipeline image comparisons.
