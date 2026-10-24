# DreamShaper 8 design

- Date: 2026-10-21
- Issue: #172
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Add the `dreamshaper-8` text-to-image checkpoint from `Lykon/dreamshaper-8` at revision
`a7e52b98680b1ba8ff7bce97c7f9f2e2e5337917`. The model is an author-published Stable Diffusion
1.5 fine-tune in diffusers format, is not gated, and is licensed under CreativeML OpenRAIL-M.

The catalog uses the full-precision safetensors components. Reduced-precision duplicates, the
single example image, feature extractor and safety checker are excluded.

## Reference

diffusers 0.40.0 `StableDiffusionPipeline`, `UNet2DConditionModel`, `AutoencoderKL`,
`DEISMultistepScheduler`; transformers 5.18.0 `CLIPTextModel` and `CLIPTokenizer`.

## Components and pipeline

The published CLIP text encoder, U-Net and KL autoencoder tensor layouts are the existing Stable
Diffusion 1.x layouts. The checkpoint's DEIS configuration is already supported by the shared
Stable Diffusion backend. Generation defaults to 50 steps and guidance 7.5 through the public API;
the checkpoint scheduler, 512-pixel default size, prompt encoding, classifier-free guidance,
latent scaling, VAE decoding and uint8 conversion remain unchanged.

No native model, tokenizer, scheduler, backend or registry changes are required.

## Memory and validation

The full-precision checkpoint is about 4.3 GB; native and reference components fit in memory on the
validation machine. Validation covers published safetensors headers, tokenizer, text encoder,
U-Net, VAE decoder, DEIS trajectory, and separate end-to-end runs at seeds 0 and 42 through
`pipeline(...)`. Fast tests cover the pinned catalog entry and published component layouts.

Out of scope: safety checking, watermarking, img2img and reduced-precision parity.
