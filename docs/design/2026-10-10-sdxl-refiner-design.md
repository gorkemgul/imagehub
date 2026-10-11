# SDXL refiner design

- Date: 2026-10-10
- Issue: #85
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`sdxl-refiner-1.0` = `stabilityai/stable-diffusion-xl-refiner-1.0` (revision
`5d4cfe854c9a9a87939ff3653551c2b3c99a4356`), by the authors, not gated, CreativeML Open RAIL++-M, catalog id
`sdxl`, family `stable-diffusion-xl`. Float32 files only: `text_encoder_2`, `unet`, `vae` and the second
tokenizer. The repository's `vae_1_0/` and fp16 files are not listed. It uses the image input
(`image_input="required"`): `image` and `strength` are accepted for this entry only; every other SDXL entry still
rejects them.

Out of scope: running the refiner after the base model in latent space (`denoising_end` / `denoising_start`),
other schedulers, several input images, IP-Adapter, fp16 parity.

## Reference

diffusers 0.40.0 `StableDiffusionXLImg2ImgPipeline`, one prompt and one image. Defaults from its signature: 50 steps,
guidance 5.0, strength 0.3, aesthetic score 6.0 / negative 2.5.

## Components

| Part | Status |
| --- | --- |
| Second text encoder and tokenizer, VAE, Euler scheduler | exist; the published configs equal SDXL base |
| U-Net (four levels, 4 transformer layers, cross-attention 1280, five time ids) | exists unchanged; random-weight parity with the published config |
| VAE posterior sampling (`encode_sample`, `encode_moments`) | **new**, additive |
| Euler `set_begin_index` and `add_noise` | **new**, additive |
| Backend mode `pipeline: refiner`, no first text encoder, preprocessing, strength-truncated timesteps | **new** |

## Pipeline details that matter for parity

- Preprocessing: round the size down to a multiple of 8, Lanczos resize, `[-1, 1]` with the reference's memory layout.
- The VAE encode samples the posterior (`mean + std * randn`) and consumes the seeded generator first; the noise for
  `add_noise` is drawn next. Latents are scaled by 0.13025 (no mean/std).
- Timesteps: `set_timesteps(steps)`, keep `scheduler.timesteps[steps - int(steps * strength):]`, begin index set to the
  first kept index; noise is added at the sigma of that index. No `init_noise_sigma` scaling.
- Time ids: `original_size + crop + (aesthetic_score,)`, negative with the negative size/crop/score.
- `force_zeros_for_empty_prompt` is false: an empty negative prompt is encoded. The context is only text encoder 2's
  penultimate hidden state (1280 channels).

## Memory

fp32: U-Net about 9.4 GB, text encoder 2 about 2.8 GB, VAE 0.3 GB; the reference and native models fit together.

## Validation

See `docs/validation.md`: random-weight and tiny end-to-end parity, then the real-checkpoint run at the defaults with a
synthetic 1024x1024 input and the seed test.
