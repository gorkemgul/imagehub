# Ovis-Image design

## Scope

Add the `ovis-image-7b` text-to-image checkpoint from `ATH-MaaS/Ovis-Image-7B` at revision
`41be1c5821a92c970d63d7eb595a2fd3fe32b22e`. The repository publishes float32 text and
transformer shards and a bfloat16 VAE; the catalog preserves those published precisions.

## Components

The implementation reuses the native Qwen3 encoder, Qwen tokenizer, AutoencoderKL, and
FlowMatchEulerDiscreteScheduler. Ovis-Image's 6 joint and 27 single transformer blocks are new.
The backend follows diffusers 0.40.0 `OvisImagePipeline`: it templates the one user message,
drops the first 28 text positions, packs 2x2 VAE latents, applies dynamic flow shift, and decodes
with scaling factor 0.3611 and shift factor 0.1159.

## Validation and limits

Offline coverage verifies the full published transformer header and a tiny forward pass. Real
checkpoint validation is opt-in and must compare tokenizer, text encoder, denoiser, VAE and two
separate public-API seed runs, freeing the reference before native loading. CPU float32 is the
parity target. LoRA, callbacks, embeddings supplied by callers, and image conditioning are out of
scope.
