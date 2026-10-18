# Meissonic design

## Scope

Add native text-to-image inference for the `meissonic` catalog model. The supported task is
single-prompt masked-token generation and VQ decoding through the public `pipeline("t2i", ...)`
API. Training, image-to-image, inpainting, and ControlNet inputs are out of scope.

## Published artifacts

- Weights: `MeissonFlow/Meissonic` at revision
  `7ee5068bcebaaf6165e5d77c2969aa3b38e17b38`, Apache-2.0, author release.
- Reference: `viiika/Meissonic` at revision
  `a9de06ca8c02c467aec3da275605fc59380299e2`.
- Reference pipeline and transformer classes: `src.pipeline.Pipeline` and
  `src.transformer.Transformer2DModel`.
- Full-precision files are used. The three published model components total about 6.0 GB.

## Components

The existing native `CLIPTextModelWithProjection`, `CLIPTokenizer`, and `VQModel` match the
published configurations. The new `MeissonicTransformer2DModel` keeps the published tensor
names while composing the existing FLUX joint-attention and single-attention blocks with the
existing masked-token embedding, RMS normalization, and convolutional resampling primitives.
The new scheduler preserves Meissonic's sampling and remasking order.

The transformer embeds a 64x64 token grid, downsamples it to 32x32, applies 14 joint
text/image blocks and 28 single-stream blocks with three-axis rotary embeddings, upsamples,
and predicts 8192 codebook logits. Five sinusoidal micro-conditions are concatenated with the
projected CLIP pooled state.

## Pipeline behavior

Generation begins with mask token 8255. Classifier-free guidance concatenates negative and
positive text states. Each scheduler step samples candidate tokens and remasks the lowest
confidence unknown positions using Gumbel noise and a cosine schedule. The final token grid is
decoded without requantization and rounded to uint8 after clipping to `[0, 1]`.

Defaults follow the release: 1024x1024, 48 steps, guidance 9, aesthetic score 6, crop `(0, 0)`,
and temperature `(2, 0)`. Smaller multiples of 32 are accepted for tests and constrained
hardware. Real-checkpoint validation uses few steps at a small valid size and is intentionally
not a quality benchmark.

## Validation

Offline tests cover published tensor headers, exact random-weight component parity, tokenizer
and scheduler parity, tiny end-to-end parity, option validation, deterministic seeds, and
metadata. Opt-in real-checkpoint tests cover each component and two separate public-pipeline
end-to-end runs with seeds 0 and 42.

