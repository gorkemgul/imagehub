# LoRA loading and LoRA training design

## Scope

Add low-rank adapters (LoRA) to imagehub in two pieces:

1. **Loading at inference (#193).** Load diffusers- and kohya-format LoRA files into the native
   Stable Diffusion 1.x and SDXL models, with output parity against diffusers
   `load_lora_weights`.
2. **Training (#194).** An optional `imagehub.training` subpackage, installed with
   `pip install "imagehub[training]"`, that trains LoRA adapters for the Stable Diffusion 1.x
   U-Net and saves them in a file the loader above (and diffusers) reads back.

Both pieces follow the approach of kadirnar's voicehub: its own LoRA layers and its own trainer,
no PEFT or Accelerate dependency, a per-model training profile registry and an optional
`training` extra. The core package stays torch-only. Model code stays in one place and is shared
by inference and training.

Other families (FLUX.1, SD 3.5, Sana, PixArt, Lumina 2, Z-Image, Qwen-Image), SDXL training,
distributed training, text-encoder training, DoRA and LyCORIS (LoHa, LoKr) adapters are later
work and are listed at the end.

## Reference: voicehub

- `voicehub/optimization/lora.py` defines `LoRAConfig` (rank, alpha, dropout, glob target
  patterns, `freeze_base`, seed), `LoRALinear` (wraps an `nn.Linear` and adds
  `scaling * B @ A @ x`, with `merge`/`unmerge`), `inject_lora` (atomically replaces matching
  linears) and `LoRAInjection` (adapter state dict, strict loading, exact `restore`).
- Model-specific files (`architectures/voxcpm2/lora.py`, `models/qwen3tts/lora.py`) choose the
  target modules for one model and read or write its adapter files.
- Training is exposed through `get_training_spec(model)` / `list_training_specs()`, which
  return a profile with a `support` level and a family, and an HF-style
  `Trainer(model, args=TrainingArguments(...), train_dataset=...)` with `train()` and
  `save_model()`.

imagehub keeps the same shapes. It differs where image models differ: published LoRA files
exist in diffusers and kohya layouts and must be read, kohya adapters also patch 1x1
convolutions, and several adapters are commonly stacked at inference.

## Package layout

```
imagehub/lora.py               LoRAConfig, LoRALinear, LoRAConv2d, inject_lora, LoRAInjection
imagehub/lora_formats.py       diffusers / legacy diffusers / kohya key conversion, file loading
imagehub/backends/stable_diffusion.py      lora=, lora_scale=, lora_fuse= options
imagehub/backends/stable_diffusion_xl.py   same options
imagehub/training/__init__.py  public training API
imagehub/training/specs.py     TrainingSupport, TrainingSpec, get_training_spec, list_training_specs
imagehub/training/arguments.py TrainingArguments
imagehub/training/data.py      ImageFolderDataset (metadata.jsonl), collation
imagehub/training/trainer.py   Trainer: loop, optimizer, schedule, checkpoints, resume, logging
imagehub/training/sd1.py       Stable Diffusion 1.x LoRA objective
```

`imagehub/lora.py` and `imagehub/lora_formats.py` are part of the core package because inference
uses them. `imagehub.training` imports pillow and raises an install hint when the extra is
missing.

## LoRA core (`imagehub/lora.py`)

`LoRAConfig` mirrors voicehub: `rank`, `alpha`, `dropout`, `target_modules` (glob patterns or
leaf names), `freeze_base`, `seed`, with `scaling = alpha / rank`.

`LoRALinear` and `LoRAConv2d` wrap a base `nn.Linear` / `nn.Conv2d`. Each wrapper holds one or
more **named adapters**. An adapter has a down matrix `A` (`rank x in`, or a `rank x in x k x k`
convolution), an up matrix `B` (`out x rank`, or a `out x rank x 1 x 1` convolution) and a
scale. The forward pass is

```
base(x) + sum over active adapters of scale_i * B_i(A_i(dropout(x)))
```

New adapters start with `A` uniform in `[-1/sqrt(in), 1/sqrt(in)]` from a seeded generator and
`B = 0`, so injection does not change the output. `merge`/`unmerge` add or subtract
`scale * B @ A` to the base weight exactly once. `inject_lora` replaces matching modules
atomically and raises with the available names when nothing matches. `LoRAInjection` exposes
`parameters()`, `adapter_state_dict()`, strict `load_adapter_state_dict()`, `set_scale(name,
value)`, `merge()`, `unmerge()` and `restore()`, which puts the original modules and their
`requires_grad` flags back.

Differences from voicehub, all needed by published image adapters:

- a convolution wrapper, because kohya adapters patch the Stable Diffusion 1.x `proj_in` /
  `proj_out` 1x1 convolutions;
- several named adapters per layer with independent scales, for stacking styles and
  characters;
- per-adapter scales that fold in the file's own alpha (`alpha / rank`) and the user's
  `lora_scale`.

## File formats (`imagehub/lora_formats.py`)

Native modules already use the diffusers and transformers state-dict names (`to_q`,
`attn1`, `text_model.encoder.layers.N.self_attn.q_proj`, ...), so most conversion is a prefix
change. The reader detects one of three layouts:

| Layout | Example key | Alpha |
| --- | --- | --- |
| diffusers / PEFT | `unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight` | safetensors `lora_adapter_metadata` header, else `rank` |
| legacy diffusers | `unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.to_q_lora.down.weight` | `rank` |
| kohya | `lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight` | per-module `.alpha` tensor |

Component prefixes map to backend components: `unet` / `lora_unet_` to the U-Net,
`text_encoder` / `lora_te_` / `lora_te1_` to the first CLIP text encoder and `text_encoder_2` /
`lora_te2_` to the SDXL second encoder. kohya names replace dots with underscores; they are
resolved against the model's own module names (dots replaced the same way), so no per-family
lookup table is needed and an ambiguous or unknown name is an error.

Files are read with the existing native safetensors reader; a small helper also returns the
header `__metadata__`. A file with DoRA (`dora_scale`), LoHa/LoKr (`hada_`, `lokr_`) or keys
that match no module is rejected with a message naming the first offending keys; nothing is
skipped silently.

## Backend integration

The Stable Diffusion 1.x and SDXL backends accept three construction options:

```python
with pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5",
              lora="style.safetensors", lora_scale=0.8) as pipe:
    image = pipe("a watercolor fox", seed=0)

pipeline(..., lora=[("style.safetensors", 0.8), ("character.safetensors", 0.5)])
pipeline(..., lora="style.safetensors", lora_fuse=True)
```

- `lora`: a path, or a list of paths or `(path, scale)` pairs.
- `lora_scale`: the scale for a single path; default 1.0.
- `lora_fuse`: merge the adapters into the base weights after loading; default `False`.

Options are validated before any weights load; files are read and injected at the end of
`_load`. Unloading the backend drops the modified modules with the rest of the model, so no
`restore` is needed there. Programmatic `load`/`set_scale`/`fuse`/`restore` stay available
through `imagehub.lora` for users who manage modules themselves.

## Training API (`imagehub.training`)

```python
from imagehub.training import (ImageFolderDataset, Trainer, TrainingArguments,
                               get_training_spec)

spec = get_training_spec("sd-v1-5")
print(spec.support.value, spec.family)        # "lora" "stable-diffusion"

trainer = Trainer(
    model="sd-v1-5",
    checkpoint="checkpoints/sd-v1-5",
    args=TrainingArguments(output_dir="runs/my-dog", max_steps=1000),
    train_dataset=ImageFolderDataset("my_images"),
)
result = trainer.train()                       # trainer.train(resume_from_checkpoint=True)
trainer.save_model("runs/my-dog/final")       # adapter.safetensors + adapter_config.json

pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5",
         lora="runs/my-dog/final/adapter.safetensors")
```

**Profiles.** `TrainingSupport` has two values to start: `lora` (a verified LoRA objective) and
`inference_only`. `TrainingSpec` records the catalog id, family, support level and the default
LoRA target modules. Every catalog model has a profile; the ones served by the Stable Diffusion
1.x backend are `lora`, except the few-step distilled `sd-turbo`, `lcm-dreamshaper-v7` and
`sdxs-512-0.9`; the rest are `inference_only`. `list_training_specs(support=...)` filters
them and feeds a training-support table in the docs. `Trainer` refuses an `inference_only`
model with a message naming the supported ones.

**Arguments.** `TrainingArguments` requires `output_dir` and `max_steps`; everything else has a
default close to the diffusers `train_text_to_image_lora.py` script:

| Argument | Default |
| --- | --- |
| `learning_rate` | `1e-4` |
| `train_batch_size` | `1` |
| `gradient_accumulation_steps` | `1` |
| `lr_scheduler`, `lr_warmup_steps` | `"constant"`, `0` (also `linear`, `cosine`) |
| `max_grad_norm` | `1.0` |
| `adam_beta1`, `adam_beta2`, `adam_weight_decay`, `adam_epsilon` | `0.9`, `0.999`, `1e-2`, `1e-8` |
| `mixed_precision` | `"no"` (also `"fp16"`, `"bf16"`) |
| `resolution`, `center_crop`, `random_flip` | `512`, `False`, `False` |
| `lora_rank`, `lora_alpha`, `lora_dropout` | `4`, `4`, `0.0` |
| `lora_target_modules` | the profile default (`to_q`, `to_k`, `to_v`, `to_out.0`) |
| `noise_offset`, `snr_gamma`, `prediction_type` | `0.0`, `None`, the scheduler's |
| `logging_steps`, `save_steps` | `10`, `500` |
| `report_to` | `"none"` (also `"wandb"`, imported only when requested) |
| `seed` | `0` |
| `device` | `"cpu"` |

**Data.** `ImageFolderDataset(path)` reads the Hugging Face imagefolder layout:

```
my_images/
  metadata.jsonl      {"file_name": "0001.png", "text": "a photo of sks dog"}
  0001.png
```

Missing files, missing captions and unreadable images fail when the dataset is created. Images
are loaded in the training process; there are no data-loader workers. Images
are resized to `resolution` on the short side, cropped (center or seeded random), optionally
flipped and normalized to `[-1, 1]`. `train_dataset` also accepts any map-style dataset whose
items are `{"image": PIL.Image | Tensor, "text": str}`, so a Hugging Face `datasets` dataset
works without imagehub depending on it.

**Objective (Stable Diffusion 1.x).** The VAE and text encoder are frozen and run without
gradients. Each step encodes images to latents (`latent_dist.sample() * scaling_factor`),
samples noise (plus `noise_offset`) and a uniform timestep, adds noise with the DDPM schedule
from the snapshot's scheduler config, and computes the MSE between the U-Net prediction and the
`epsilon` or `v_prediction` target, with optional min-SNR weighting. Only the U-Net LoRA
parameters train, with AdamW. New adapters use PEFT's gaussian initialization, as the script does.
The script's gradient clipping never runs (it receives an exhausted parameter generator);
imagehub clips as documented.

**Precision.** With `mixed_precision`, frozen components run in the requested dtype and the
U-Net forward runs under autocast; LoRA parameters and optimizer state stay float32. fp16 uses a
gradient scaler. Parity is checked only in float32.

**Checkpoints and resume.** Every `save_steps` the trainer writes
`output_dir/checkpoint-N/` with `adapter.safetensors`, `adapter_config.json` and
`trainer_state.pt` (optimizer, scheduler, gradient scaler, step, data-order seed and RNG
states). `train(resume_from_checkpoint=True)` picks the latest checkpoint;
a path picks a specific one. `trainer_state.pt` is a pickle and is read with
`torch.load(weights_only=True)`; adapter weights are always safetensors.

**Saved adapters.** `save_model(path)` writes `adapter.safetensors` in the diffusers / PEFT
layout (`unet.<module>.lora_A.weight`, `unet.<module>.lora_B.weight`) with the rank, alpha and
target modules in the `lora_adapter_metadata` header, which diffusers reads, plus a readable
`adapter_config.json`. The native safetensors writer gains an optional metadata argument.

**Logging.** Loss, learning rate and step time are printed every `logging_steps` and returned in
`TrainOutput(global_step, training_loss, metrics)`. `report_to="wandb"` also sends them to
Weights & Biases when the user has it installed.

## Validation

All parity tests run on CPU in float32 against diffusers 0.40.0 / transformers 5.18.0. The
reference extra adds a pinned `peft`, which diffusers needs for `load_lora_weights` and its
LoRA training path; the core and training extras do not.

Loading (#193):

- unit tests: injection, zero-init no-op, merge/unmerge round trip, restore, strict state
  loading, multi-adapter scaling, rejected formats;
- tiny random Stable Diffusion 1.x and SDXL models with random adapters written in each of the
  three layouts, covering U-Net and text encoders: imagehub component outputs and a shortened
  end-to-end run match diffusers after `load_lora_weights` (and after `fuse_lora` for
  `lora_fuse=True`);
- one real-weights reference check: `sd-v1-5` with a small published LoRA, shortened steps.

Training (#194):

- one step on a tiny Stable Diffusion 1.x model with identical initial LoRA weights, latents,
  noise and timesteps: the loss and every LoRA gradient match a diffusers U-Net with the same
  adapter added through `add_adapter`; parameters also match after several AdamW steps with a
  warmup schedule and gradient clipping;
- `epsilon` and `v_prediction`, `noise_offset` and `snr_gamma` paths;
- resume: training N steps in one run equals training k steps, resuming and training N-k;
- gradient accumulation equals the larger batch within float32 tolerance;
- a bf16 smoke run on CPU completes with finite loss;
- a saved adapter loads with `imagehub.lora_formats` and with diffusers `load_lora_weights`, and
  both produce the same output;
- dataset errors (missing file, missing caption, bad image) are reported at construction.

No GPU is needed for these tests.

## Pull requests

1. **LoRA loading**: `imagehub/lora.py`, `imagehub/lora_formats.py`, the two backends, docs and
   this design note. Closes #193.
2. **LoRA training**: `imagehub.training`, the `training` extra, docs and a training-support
   table. Closes #194.

## Later

- LoRA loading for FLUX.1, SD 3.5, Sana, PixArt, Lumina 2, Z-Image and Qwen-Image (their
  diffusers-format files already share module names; kohya FLUX files need their own map).
- Training for SDXL (size/crop conditioning, two text encoders, gradient checkpointing), then
  the DiT families.
- Text-encoder LoRA training, DreamBooth prior preservation, EMA.
- Distributed training, which needs multi-GPU hardware to validate.
- A `python -m imagehub.train` command-line wrapper over the same API.
- DoRA and LyCORIS adapters.
