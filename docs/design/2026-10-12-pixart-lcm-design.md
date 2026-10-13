# PixArt-LCM design

## Scope

Add `pixart-lcm-xl-1024` for the published PixArt-LCM XL 2-step checkpoint at revision
`e153060b10990994d2ed31304b3224da47fbd740`. It reuses the PixArt transformer, T5 encoder,
T5 tokenizer, AutoencoderKL, and LCM scheduler.

## Pipeline

The checkpoint uses `PixArtAlphaPipeline` with an `LCMScheduler`. Its defaults are four steps,
guidance 0.0, and 1024 by 1024 output. At guidance above 1.0 it retains PixArt's
classifier-free-guidance path. The seeded CPU generator supplies both initial latents and the
LCM scheduler's between-step noise, matching the reference call order.

## Weights and validation

The repository publishes float32 transformer and VAE safetensors plus four float32 T5 shards.
Fixtures retain the published configs and safetensors headers. Unit and tiny-reference tests cover
defaults, scheduler selection, strict meta loading, and exact tiny end-to-end LCM behavior.
The real-checkpoint reference tests are extended but skip without a local snapshot. No weights are
downloaded by tests.

## Out of scope

ControlNet, LCM-LoRA, caption cleaning, safety checking, watermarking, and reduced-precision
parity remain unsupported.
