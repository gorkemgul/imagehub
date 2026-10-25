# Infinity design

- Date: 2026-10-25
- Issue: #19
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Add the author-published Infinity-2B text-to-image checkpoint as `infinity-2b`, linked to catalog
id `infinity`, with native CPU float32 inference and strict loading. The checkpoint is
`FoundationVision/Infinity` at revision `e7108923249fb3c836ebfbc3f4b4d318d6131293`:
`infinity_2b_reg.pth` and its bitwise tokenizer `infinity_vae_d32reg.pth`. The text encoder and
tokenizer are `google/flan-t5-xl` at revision
`7d6315df2c2fb742f0f5b556879d730926ca9001`. Both code and weights are MIT licensed.

Out of scope: the 125M and 8B checkpoints, training, video, FlexAttention and fused kernels,
spatial patchification, prompt rewriting, watermarking, and reduced-precision parity.

## Reference

The original `FoundationVision/Infinity` repository at commit
`9f9fcd1de7ead377ddf1106db1249a35f3a59143`: `Infinity`, `vae_model`,
`encode_prompt`, and `Infinity.autoregressive_infer_cfg`. The published 2B configuration is 32
transformer blocks, width 2048, 16 heads, d32 binary spherical quantization, 16x VAE compression,
Flan-T5-XL conditioning, per-scale CFG 3, temperature 0.5, top-k 900, and top-p 0.97.

## Components

| Part | Status |
| --- | --- |
| Flan-T5 tokenizer and encoder | reuse `T5Tokenizer` and `T5EncoderModel` |
| Infinity cross-attention transformer, 2D RoPE, adaptive layer norm | new |
| d32 binary spherical quantizer and VAE decoder | new |
| bitwise top-k/top-p sampling and next-scale generation loop | new backend |

The native modules keep the published tensor names. Optional reference dependencies and fused
operations are replaced by the corresponding plain PyTorch operations; runtime imports remain
limited to PyTorch and the standard library.

## Pipeline

Tokenize one prompt to 512 tokens, compact its non-padding Flan-T5 hidden states, and predict a
pyramid of 32 independent bits per spatial token. Each sampled scale is converted to signed,
unit-normalized codes and added into the full latent grid; the accumulated grid supplies the next
scale and is finally decoded by the VAE. Classifier-free guidance uses the learned unconditional
text tensor. Images use the reference BGR-to-RGB conversion and uint8 truncation.

The public default is 1024x1024 with the full published scale schedule. Real-checkpoint tests use
the smallest officially supported `0.06M` schedule (seven scales ending at a 16x16 latent grid,
decoded to 256x256), so validation finishes in minutes rather than running the 1024px schedule.

## Memory

The Infinity archive is 8.82 GB, the d32 VAE is 1.56 GB, and Flan-T5-XL is 11.40 GB on disk.
Component validation loads the reference and native implementations in separate subprocesses.
The native pipeline loads components sequentially where possible and uses CPU float32 for parity.

## Validation

Published header checks cover all three components. Random-weight parity covers transformer
conditioning, cached self-attention, bit sampling, quantizer conversion, and VAE decode. Tiny
snapshot tests cover exact end-to-end generation, option validation before loading, repeatability,
request isolation, and autograd restoration. Real-checkpoint tests cover tokenizer/text features,
one transformer scale, VAE decode, and separate seed-0 and seed-42 public-pipeline runs through the
shortest schedule.

