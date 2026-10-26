# LDM design

- Date: 2026-10-13
- Issue: #124

## Scope

`ldm-text2im-large-256` from `CompVis/ldm-text2im-large-256` at revision
`30de525ca11a880baea4962827fb6cb0bb268955`, author provenance, Apache-2.0, and float32
tensor-only `.bin` weights. The catalog entry includes tokenizer, LDM BERT, U-Net, VQ-VAE, and
DDIM files.

## Components and validation

The existing U-Net and f=8 KL autoencoder are reused. The legacy U-Net config omits its 1280-wide
cross-attention dimension, which the loader supplies. LDMBert is native and uses the existing BERT
WordPiece tokenizer. The backend follows diffusers 0.40.0 `LDMTextToImagePipeline`: 50 DDIM steps,
guidance 1.0, 256×256, eta 0, CPU-seeded noise, VQ-VAE decoding and uint8 rounding. Random BERT
parity is covered; snapshot-gated tests cover all components, end-to-end output and seed parity.

## Out of scope

Safety checkers, watermarks, img2img, other schedulers, reduced precision and nonzero eta.
