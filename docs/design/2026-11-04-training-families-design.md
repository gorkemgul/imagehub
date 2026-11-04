# Training more families design

## Scope

Extend LoRA training (#194, #200) from Stable Diffusion 1.x / 2.x to the families for which
diffusers 0.40.0 ships a LoRA training script, so every family is checked against a reference
training step:

| Family | Catalog checkpoints | Objective | Reference script |
| --- | --- | --- | --- |
| Stable Diffusion XL | `sdxl-base-1.0` and fine-tunes | DDPM | `examples/text_to_image/train_text_to_image_lora_sdxl.py` |
| Stable Diffusion 3 / 3.5 | `sd3-medium`, `sd3.5-medium`, `sd3.5-large` | flow matching | `examples/dreambooth/train_dreambooth_lora_sd3.py` |
| FLUX.1 | `flux1-dev`, `flux1-krea-dev`, `flux-1-lite-8b`, `flux1-kontext-dev` | flow matching | `train_dreambooth_lora_flux.py`, `..._flux_kontext.py` |
| Sana | `sana-600m-1024`, `sana-1600m-1024`, `sana-1.5-4.8b-1024` | flow matching | `train_dreambooth_lora_sana.py` |
| Qwen-Image | `qwen-image`, `qwen-image-2.1` | flow matching | `train_dreambooth_lora_qwen_image.py` |
| Z-Image | `z-image` | flow matching | `train_dreambooth_lora_z_image.py` |
| Lumina 2 | `lumina-image-2.0` | flow matching | `train_dreambooth_lora_lumina2.py` |
| HiDream | `hidream-i1-dev`, `hidream-i1-full` | flow matching | `train_dreambooth_lora_hidream.py` |
| FLUX.2 klein | `flux2-klein-4b`, `flux2-klein-9b` | flow matching | `train_dreambooth_lora_flux2_klein.py` |
| Krea 2, Ideogram 4 | `krea2-turbo`, `ideogram4-fp8` | flow matching | `train_dreambooth_lora_krea2.py`, `..._ideogram4.py` |

Few-step distilled checkpoints (SDXL Lightning / Hyper / DMD2 / Turbo, `sd3.5-large-turbo`,
`flux1-schnell`, `sana-sprint`, `z-image-turbo`, `hidream-i1-fast`) stay `inference_only`, as
`sd-turbo` does: their distillation objectives are not the denoising objective. Whether a turbo
checkpoint with a reference script (`krea2-turbo`) is trainable is decided in its issue.

Out of scope: prior preservation (DreamBooth class images), text-encoder LoRA, full fine-tuning,
image-to-image training scripts, and families without a reference training script.

## Family adapters

The trainer stops depending on Stable Diffusion 1.x. A `TrainingFamily` object supplies what
differs per family; the trainer keeps the loop, data order, accumulation, clipping, precision,
evaluation, sample images, distributed processes, trackers, checkpoints and resume.

```python
class TrainingFamily:
    name: str                       # "stable-diffusion", "stable-diffusion-xl", ...
    prefix: str                     # adapter key prefix: "unet" or "transformer"
    target_modules: tuple[str, ...] # default LoRA targets, from the reference script
    def load(self, checkpoint, *, device, dtype, model_id) -> Components: ...
    def trainable(self, components) -> nn.Module          # the module that gets LoRA
    def frozen(self, components) -> list[nn.Module]
    def collator(self, components, args, generator) -> Collator
    def objective(self, components, args) -> Objective
    def sample(self, components, prompt, args, device) -> Tensor   # (H, W, 3) uint8
```

An `Objective` is called as `objective(batch, generator) -> loss`, where `batch` is the collator's
dictionary moved to the device. It also exposes `draw(batch, generator)` (the random tensors of a
step in the reference script's order) and `loss_from(batch, draws)`, which the reference tests and
the distributed replay use. Profiles (`TrainingSpec`) name their family; `get_training_spec` keeps
its output, plus the family's prefix in the adapter metadata.

Sample images reuse each backend's own denoising functions, so `validation_prompts` output equals
`pipeline(..., lora=adapter)`; LoRA loading for a family is added in the same change as its
training, with diffusers `load_lora_weights` parity, so a trained adapter always loads back.

## Objectives

**DDPM** (SD 1.x / 2.x, SDXL): unchanged from #194. SDXL adds both CLIP encoders (penultimate
hidden states concatenated, pooled output of the second), the `time_ids` of original size, crop
top-left and target size recorded by the collator, and the fp32 VAE the script requires.

**Flow matching** (all others), as in `train_dreambooth_lora_sd3.py` and its siblings:

- latents `(vae_sample - shift_factor) * scaling_factor` (FLUX and FLUX.2 also pack 2x2 patches
  and build image ids);
- `u` from `compute_density_for_timestep_sampling(weighting_scheme, logit_mean, logit_std,
  mode_scale)`, `timesteps = scheduler.timesteps[(u * num_train_timesteps).long()]`, sigmas looked
  up from the training scheduler;
- `noisy = (1 - sigma) * latents + sigma * noise`; target `noise - latents`, or `latents` with
  `precondition_outputs`;
- loss `mean(weighting * (pred - target)^2)` per sample, then over the batch, with
  `compute_loss_weighting_for_sd3`;
- per-family conditioning: text encoders and their prompt formatting, guidance embedding (FLUX.1
  dev, FLUX.2), timestep scaling, attention masks, and the transformer's call signature.

`TrainingArguments` gains `weighting_scheme`, `logit_mean`, `logit_std`, `mode_scale` and
`precondition_outputs`, with the reference scripts' defaults; they are rejected for DDPM families.

## Memory

Each family loads only what training needs on the training device: text encoders and the VAE are
frozen and run without gradients. Families whose reference scripts precompute prompt embeddings
and offload text encoders (FLUX.1, Qwen-Image, HiDream) do the same: embeddings are computed per
batch and the encoders can stay on the CPU (`text_encoder_device`). Gradient checkpointing for the
trainable module comes with SDXL, the first family where it matters.

## Validation

Per family, on CPU in float32 against diffusers 0.40.0 and peft 0.21.2:

- tiny random snapshot: the first step's loss and every LoRA gradient equal the script's step, for
  each weighting scheme the family's script offers;
- a saved adapter loads with `pipeline(..., lora=)` and diffusers `load_lora_weights` to identical
  images; LoRA loading parity for the family's diffusers and kohya layouts;
- the published checkpoint's first step where both copies fit in memory; the pull request states
  which checkpoints were not run this way.

No GPU is needed.

## Work order

One issue and one pull request each, under the tracking issue:

1. Family adapters (no behavior change).
2. SDXL.
3. Flow-matching objective and Stable Diffusion 3 / 3.5.
4. FLUX.1 (with Kontext).
5. Sana.
6. Qwen-Image.
7. Z-Image.
8. Lumina 2.
9. HiDream.
10. FLUX.2 klein.
11. Krea 2 and Ideogram 4.
