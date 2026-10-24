# Emu3-Gen design

- Date: 2026-10-24
- Issue: #180
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Add the author checkpoint `BAAI/Emu3-Gen` at revision
`40b3bb89a9cee99ab0b9deb35dead2e50d6a0d72` and its required
`BAAI/Emu3-VisionTokenizer` at revision `c81f916ad371289e205310a7539255e8a9396488`.
Both weight sets and the official code are Apache-2.0. The official code reference is
`baaivision/Emu3` revision `dbbf9858194d70b8c58293e219ecffe22df0f9c7`.

Training, understanding and video generation are out of scope. The backend supports one text
prompt and image generation through the common `t2i` API.

## Reference and components

The parity target is the official `Emu3ForCausalLM.generate` path driven by `Emu3Processor`,
followed by `Emu3VisionVQModel.decode`. New native components are the tiktoken-compatible Emu3
tokenizer, causal Emu3 decoder, constrained autoregressive sampler and spatiotemporal VQ decoder.
The published language model has 32 Llama-style layers (4096 hidden units, 32 query heads and
eight key/value heads); the separate tokenizer has a 32,768-entry four-channel codebook.

## Pipeline

The prompt is prefixed by the published BOS marker and suffixed by the image-start marker and
token-grid dimensions. Classifier-free guidance evaluates the negative prompt separately.
Generation samples only vision tokens at image positions, inserts row/end markers deterministically,
and stops at EOS. The visual token grid is decoded to RGB and converted from `[-1, 1]` to uint8.
Defaults retain the published 518,400-pixel area, guidance 3 and top-k 2048. Width and height are
rounded to the tokenizer's spatial factor of eight.

## Memory and validation

The generator is about 8.5B parameters in float32 and the vision tokenizer about 270M parameters.
Real comparisons therefore run the official side in a subprocess and free it before native loading.
Published-header tests cover every shard and all vision-tokenizer tensors. Exact random-weight tests
cover tokenization, transformer caching, constrained sampling, VQ decoding and a tiny whole pipeline.
Real-checkpoint tests separately record headers, tokenizer, one language-model step, VQ decode,
and shortened seed-0 and seed-42 end-to-end runs through `pipeline(...)`.

