# HunyuanImage 2.1 design

- Date: 2026-10-10
- Issue: #93
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

One weight set of a new family `hunyuanimage`, text-to-image, catalog id `hunyuanimage-2-1`, base
(non-distilled) model, Tencent Hunyuan Community License (model card and catalog):

| Model id | Repository | Revision | Transformer |
| --- | --- | --- | --- |
| `hunyuanimage-2-1` | `hunyuanvideo-community/HunyuanImage-2.1-Diffusers` | `7e7b7a177de58591aeaffca0929f4765003d7ced` | 17B, four bfloat16 shards |

The repository is a community conversion of `tencent/HunyuanImage-2.1` to the diffusers layout that
`HunyuanImagePipeline` loads. The transformer, the Qwen2.5-VL text encoder and the VAE are published
in bfloat16 only (the ByT5 encoder in float32); parity is validated in float32 after upcasting.

Out of scope: the refiner (`HunyuanImageRefinerPipeline`), the distilled variants (guidance embeddings,
mean-flow timesteps), PAG, reduced-precision parity, sizes that are not a multiple of 32.

## Reference

diffusers 0.40.0 `HunyuanImagePipeline`, `HunyuanImageTransformer2DModel`,
`AutoencoderKLHunyuanImage`, `AdaptiveProjectedMixGuidance`, `FlowMatchEulerDiscreteScheduler`;
transformers 5.18.0 `Qwen2_5_VLForConditionalGeneration`, `T5EncoderModel`, `Qwen2Tokenizer`,
`ByT5Tokenizer`.

## Components

| Part | Status |
| --- | --- |
| Qwen2.5-VL text model (`Qwen25VLTextModel`) | existing; used unchanged |
| T5 encoder (`T5EncoderModel`) at the ByT5 configuration | existing; used unchanged |
| Qwen2 tokenizer | existing; now also reads `vocab.json`, `merges.txt` and the added tokens of the config when there is no `tokenizer.json` |
| ByT5 tokenizer (bytes + 3, added tokens with whitespace stripping, one reserved end position) | **new** |
| `HunyuanImageTransformer2DModel` (dual and single blocks with a boolean key mask, token refiner, ByT5 projection) | **new**, built on the FLUX layers |
| `AutoencoderKLHunyuanImage` (32x, 64 channels, decoder and encoder) | **new** |
| Flow-match Euler with shift 5.0 and explicit sigmas | existing |
| Adaptive projected mix guidance | **new** (in the backend), checked against the diffusers guider |

## Pipeline

As `HunyuanImagePipeline.__call__` for one prompt: the prompt goes into the system/user template; the
Qwen tokens are padded to 1034 (right padding); the embedding is the third-from-last hidden state with
the first 34 positions dropped. Quoted passages (single, double and Chinese quotes) are formatted as
`Text "..." . ` and encoded by ByT5 to 128 tokens; without quotes the glyph embedding and mask are
zeros, and the plain guider is used, otherwise the OCR guider. The negative prompt (default empty) is
encoded the same way with a zero glyph stream. Latents `randn(1, 64, H/32, W/32)` from a CPU generator;
sigmas `linspace(1, 0, steps + 1)[:-1]`; timesteps are the 0-1000 scheduler values. At every step the
conditional and unconditional predictions are computed; steps up to the guider's start step use
classifier-free guidance (and fill the momentum buffer), later steps use adaptive projected guidance
(momentum -0.5, norm threshold 10, scale 10, eta 0). Guidance scale 1 runs the conditional branch only.
Decode: `vae.decode(latents / 0.75289)`, then `(x / 2 + 0.5)` clamped, times 255, rounded, uint8.

In the transformer, the valid ByT5 tokens, the valid Qwen tokens, then the padding of both streams are
concatenated; the key mask is true for all image tokens and for the valid text tokens, and the
rotary embedding is applied to image tokens only.

## Memory

Float32: transformer 68 GB, Qwen2.5-VL 30 GB, VAE 3 GB, ByT5 1 GB. The backend loads the stages one at a
time and frees each model before the next. The real test computes each reference stage in a subprocess,
saves it and lets the process exit before loading the native model; end-to-end runs are shortened
(8 steps at 512x512, and 3 steps at 256x256 for the seed test).

## Validation

Header checks against the fixtures; tokenizer fuzz tests; random-weight exact parity for the
transformer, VAE, ByT5 encoder, guider and schedule; a tiny snapshot compared with the diffusers
pipeline (defaults, guidance on and off, non-square, quoted text, two seeds); the real-checkpoint file
`tests/reference/test_ref_hunyuanimage.py`.
