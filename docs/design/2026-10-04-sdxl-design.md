# SDXL family design

- Date: 2026-10-04
- Roadmap step: 3 in `.ai/GOAL.md`
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Phase A (this design's first deliverable): `sdxl-base-1.0` (`stabilityai/stable-diffusion-xl-base-1.0`),
text-to-image only, at full `full` validation status.

Phase B (follow-up pull requests reusing phase A): `sdxl-turbo`, `ssd-1b`, `playground-v2.5`. Each needs
only a catalog entry, possibly a scheduler and its own parity run.

Out of scope: the refiner, img2img/inpainting, watermarking, LoRA/IP-Adapter, fp16 parity.

## Reference

diffusers 0.40.0 `StableDiffusionXLPipeline`, `UNet2DConditionModel`, `AutoencoderKL`,
`EulerDiscreteScheduler`; transformers 5.18.0 `CLIPTextModel`, `CLIPTextModelWithProjection`,
`CLIPTokenizer`. Every numeric detail follows the installed source of these versions.

## Design

Extend the SD 1.x code instead of forking it. Each extension is driven by the published config, so
SD 1.x behaviour and tensor names stay identical (its parity tests must keep passing unchanged).

### U-Net (`models/unet_2d.py`)

- Per-block `transformer_layers_per_block` (int or list) and per-block head counts
  (`attention_head_dim` list, or `num_attention_heads` when set).
- `use_linear_projection`: `proj_in`/`proj_out` as `nn.Linear` applied after flattening, instead of
  1×1 convolutions.
- Down/up block type lists of any length (SDXL has three levels and a first block without attention).
- `addition_embed_type = "text_time"`: `add_time_proj` (sinusoidal over each time id with
  `addition_time_embed_dim`) and `add_embedding` (`TimestepEmbedding` over
  `projection_class_embeddings_input_dim`), added to the time embedding. The forward takes optional
  `text_embeds` (pooled) and `time_ids` (6 values per sample).
- Unsupported config options (class embeddings, other addition types, attention masks) raise
  `ValueError` instead of being ignored.

### Text encoders (`models/clip_text.py`)

- `hidden_act` `"gelu"` in addition to `"quick_gelu"`.
- Optional access to intermediate hidden states (SDXL uses the penultimate layer's output, before the
  final layer norm).
- `CLIPTextModelWithProjection`: same transformer plus `text_projection` (no bias); its pooled output is
  the final-layer-normed hidden state at the first EOS position (`argmax` of input ids when
  `eos_token_id == 2` in the legacy config, matching transformers' rule) projected by `text_projection`.
  Exact pooling rule: read transformers 5.18.0 `CLIPTextTransformer.forward`.

### Scheduler (`schedulers/euler.py`)

`EulerDiscreteScheduler` with the SDXL release config (scaled_linear betas, leading spacing,
steps_offset 1, epsilon prediction, no Karras sigmas). `init_noise_sigma` is not 1: noise scaling,
`scale_model_input` and the step formula follow the reference exactly. Unsupported options raise
`ValueError`.

### Backend (`backends/stable_diffusion_xl.py`)

- Components: `tokenizer`, `tokenizer_2`, `text_encoder`, `text_encoder_2`, `unet`, `vae`, `scheduler`.
- Prompt encoding as in the reference: penultimate hidden states of both encoders concatenated on the
  channel axis (768 + 1280 = 2048), pooled embeds from encoder 2. Negative prompt: zeros when
  `force_zeros_for_empty_prompt` is set and no negative prompt is given; otherwise encode it.
- `add_time_ids = original_size + crops_coords_top_left + target_size` (defaults: size, (0, 0), size).
- Defaults from the reference pipeline: 50 steps, guidance 5.0, 1024×1024.
- Decoding: VAE with `scaling_factor` 0.13025. The reference upcasts the VAE to float32
  (`force_upcast`); in float32 runs this is a no-op.
- Same option validation, metadata and error behaviour as the SD 1.x backend; shared helpers move into
  a common module only where both backends use them unchanged.

### Catalog and download

`sdxl-base-1.0`: `catalog_id` `sdxl`, `provenance` `author`, pinned revision, license
`CreativeML Open RAIL++-M`, files: configs, both tokenizers, fp32 safetensors of `text_encoder`,
`text_encoder_2`, `unet`, `vae` (no fp16 variants, no `vae_1_0`, no ONNX/OpenVINO/Flax files).

## Validation

Follow the parity ladder in the skill: header check, random-weight structural parity (tiny and full
SDXL configs) for each new or changed module, tokenizer parity and fuzz for both tokenizers, Euler
scheduler parity, a tiny-snapshot end-to-end comparison with `StableDiffusionXLPipeline`, then real
checkpoint component and end-to-end parity at 1024×1024 on CPU float32, and the seed test.

Memory: 2.6B U-Net + 0.8B encoders, fp32, two copies ≈ 28 GB — within budget.

## Task breakdown

1. U-Net generalization + random-weight parity (SD 1.x tests unchanged).
2. Text encoders (gelu, hidden states, projection model) + random-weight parity.
3. Euler scheduler + parity.
4. SDXL backend, catalog entry, tiny-snapshot unit and reference tests.
5. Real-checkpoint download, parity run, validation records and docs.
