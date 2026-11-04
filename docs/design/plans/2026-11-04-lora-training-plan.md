# LoRA training implementation plan

**Goal:** An optional `imagehub.training` subpackage that trains LoRA adapters for the Stable
Diffusion 1.x U-Net, with the voicehub-style API (`get_training_spec`, `Trainer`,
`TrainingArguments`) and a first step equal to the diffusers `train_text_to_image_lora.py` step
on CPU float32 (#194).

**Architecture:** `imagehub/training/specs.py` derives a profile for every catalog model.
`arguments.py` validates settings. `data.py` reads the `metadata.jsonl` image folder and applies
the torchvision-equivalent transforms with pillow. `sd1.py` holds the pure objective (VAE and
text encoder frozen, DDPM noise, epsilon / v-prediction, min-SNR). `trainer.py` owns the loop:
seeded sampling, AdamW, LR schedules, accumulation, clipping, mixed precision, checkpoints, exact
resume, adapter export in the diffusers layout and optional wandb.

**Tech stack:** torch; pillow through the new `training` extra. Reference tests use diffusers
0.40.0, transformers 5.18.0, peft 0.21.2 and torchvision.

Design: `docs/design/2026-11-04-lora-and-training-design.md`. Depends on #193 (branch
`feat/lora-loading`); this branch starts from it and is rebased on `main` once #193 merges.

## Global constraints

- The core package stays torch-only; `imagehub.training` imports pillow lazily and raises an
  install hint (`pip install "imagehub[training]"`) without it.
- Parity on CPU float32; tolerance 0 unless a measured difference forces the next power of ten.
- English repository text; commits without tool or assistant names; checks chained with `&&`.
- Branch `feat/lora-training`, worktree `/Users/gorkem.gul/Personal/imagehub-wt/lora-training`;
  PR to `main` with `Closes #194` after #193 merges.

## Reference behavior (diffusers 0.40.0 `train_text_to_image_lora.py`)

- LoRA: `LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
  target_modules=["to_k", "to_q", "to_v", "to_out.0"])` on the U-Net only; gaussian init is
  `A ~ N(0, 1/r)`, `B = 0`.
- Data: `Resize(resolution, bilinear)` on the short side, `CenterCrop` or `RandomCrop`, optional
  `RandomHorizontalFlip`, `ToTensor`, `Normalize([0.5], [0.5])`; captions tokenized to 77 ids.
- Step: `latents = vae.encode(x).latent_dist.sample() * scaling_factor`; `noise = randn_like`;
  `noise += noise_offset * randn(b, c, 1, 1)`; `t = randint(0, num_train_timesteps)`;
  `noisy = scheduler.add_noise(latents, noise, t)`; `ehs = text_encoder(ids)[0]`; target is
  `noise` or `get_velocity(latents, noise, t)`; loss `mse(pred.float(), target.float())`, or with
  `snr_gamma` the per-sample mean times `min(snr, gamma) / snr` (`/ (snr + 1)` for v-prediction).
- AdamW on the LoRA parameters; `get_scheduler(name, warmup, total)`.
- The script passes an already-consumed parameter generator to `clip_grad_norm_`, so it never
  clips. imagehub clips as documented; parity runs keep gradient norms below `max_grad_norm` or
  compare against an explicitly clipped reference.

## File map

| File | Responsibility |
| --- | --- |
| `imagehub/lora.py` | `LoRAConfig.init` (`"uniform"` / `"gaussian"`) and gaussian init in `add_adapter` |
| `imagehub/training/__init__.py` | public API |
| `imagehub/training/specs.py` | `TrainingSupport`, `TrainingSpec`, `get_training_spec`, `list_training_specs` |
| `imagehub/training/arguments.py` | `TrainingArguments` |
| `imagehub/training/data.py` | `ImageFolderDataset`, `preprocess_image` |
| `imagehub/training/schedules.py` | `lr_lambda(name, warmup, total)` matching diffusers |
| `imagehub/training/sd1.py` | `StableDiffusionLoRAObjective` |
| `imagehub/training/trainer.py` | `Trainer`, `TrainOutput` |
| `pyproject.toml` | `training = ["pillow>=10"]` |
| `tests/test_training_*.py` | unit tests without diffusers |
| `tests/reference/test_ref_training.py` | parity with diffusers + peft + torchvision |
| `docs/training.md`, `mkdocs.yml`, `README.md` | guide and training-support table |

### Task 1: gaussian LoRA init

- `LoRAConfig.init: str = "uniform"`, validated in `("uniform", "gaussian")`.
- `LoRALayer.add_adapter(..., init="uniform")`: with no tensors, `"gaussian"` fills `A` with
  `torch.randn(shape, generator=generator) * (1 / rank)` and zeroes `B`.
- Tests: config rejects other values; gaussian `A` is seeded, has std close to `1/rank` on a
  large layer, `B` is zero.

### Task 2: profiles and arguments

`specs.py`:

```python
class TrainingSupport(str, Enum):
    LORA = "lora"
    INFERENCE_ONLY = "inference_only"

@dataclass(frozen=True)
class TrainingSpec:
    model_id: str
    family: str
    support: TrainingSupport
    target_modules: tuple[str, ...] = ()

def get_training_spec(model_id: str) -> TrainingSpec: ...          # KeyError naming close ids
def list_training_specs(*, support: TrainingSupport | str | None = None) -> tuple[TrainingSpec, ...]: ...
```

Catalog models of family `stable-diffusion` are `lora` with targets
`("to_q", "to_k", "to_v", "to_out.0")`, except `sdxs-512-0.9`, whose tiny autoencoder has no
posterior; every other model is `inference_only`. Order follows `MODEL_CATALOG`.

`arguments.py`: a frozen dataclass with the fields and defaults of the design note (minus
`dataloader_num_workers`, which the in-process loader does not need), validated in
`__post_init__` (positive ints, finite non-negative floats, known enum strings, `output_dir`
converted to `Path`). `lora_target_modules: tuple[str, ...] | None = None` means the profile's.

Tests: profile counts, `sd-v1-5` is `lora`, `flux1-dev`-style ids are `inference_only`, unknown id
error, filter by support; argument validation table.

### Task 3: data

- `ImageFolderDataset(root)`: reads `root/metadata.jsonl` (one JSON object per line with
  `file_name` and `text`), checks every file exists and opens with pillow (`Image.open(...).verify()`),
  and every caption is a non-empty string; errors name the line. `__getitem__` returns
  `{"image": PIL.Image (RGB), "text": str}`.
- `preprocess_image(image, *, resolution, center_crop, random_flip, generator) -> Tensor`:
  RGB; short side resized to `resolution` with `Image.BILINEAR` and the long side to
  `int(resolution * long / short)` (torchvision's rule, antialiased by pillow); center crop with
  `round((h - th) / 2)` offsets, or random crop offsets `torch.randint(0, h - th + 1, (1,),
  generator=generator)` then the same for width; flip when `torch.rand(1, generator=generator)
  < 0.5`; tensor `(C, H, W)` float in `[0, 1]`, then `(x - 0.5) / 0.5`. Accepts tensors `(C, H, W)`
  in `[0, 1]` as already-loaded images (no resize when the size already matches).
- Tests: metadata errors, a 2-image folder, preprocessing shapes and determinism; reference test
  compares center-crop preprocessing with torchvision bit for bit.

### Task 4: objective and schedules

`schedules.py`: `lr_lambda(name, warmup, total) -> Callable[[int], float]` for `constant`,
`constant_with_warmup`, `linear`, `cosine` with the diffusers formulas.

`sd1.py`:

```python
class StableDiffusionLoRAObjective:
    def __init__(self, components: StableDiffusionComponents, *, prediction_type: str | None,
                 snr_gamma: float | None, noise_offset: float) -> None: ...
    def encode_text(self, input_ids: Tensor) -> Tensor: ...        # no grad
    def encode_images(self, pixels: Tensor, latent_noise: Tensor) -> Tensor: ...   # no grad, scaled
    def loss(self, latents: Tensor, encoder_hidden_states: Tensor, noise: Tensor,
             timesteps: Tensor) -> Tensor: ...
    def draw(self, latents_shape, generator) -> tuple[Tensor, Tensor, Tensor]: ...  # latent_noise, noise(+offset), timesteps
```

Uses `imagehub.schedulers.ddpm.DDPMScheduler(scheduler_config)` for `alphas_cumprod` and
`add_noise`; velocity and SNR follow diffusers `get_velocity` / `compute_snr`. The prediction type
is the argument or the scheduler config's.

Reference test: tiny SD snapshot; diffusers U-Net with `add_adapter(LoraConfig(...))`, ours with
`inject_lora`; identical non-zero A/B, pixels, ids, latent noise, noise and timesteps → loss and
every LoRA gradient equal; parametrized over epsilon, v-prediction, `snr_gamma=5`,
`noise_offset=0.1`. Then 4 AdamW steps with `cosine` warmup 1 and explicit clipping on both
sides (`torch.nn.utils.clip_grad_norm_` on the reference parameters) → parameters equal.
`lr_lambda` equals `diffusers.optimization.get_scheduler` over 20 steps for each name.

### Task 5: trainer

```python
@dataclass
class TrainOutput:
    global_step: int
    training_loss: float
    metrics: dict[str, float]
    log_history: list[dict[str, float]]

class Trainer:
    def __init__(self, model: str, checkpoint: str | os.PathLike[str], args: TrainingArguments,
                 train_dataset: Any) -> None: ...
    def train(self, resume_from_checkpoint: bool | str | os.PathLike[str] = False) -> TrainOutput: ...
    def save_model(self, path: str | os.PathLike[str] | None = None) -> Path: ...
```

- Construction validates the profile (refuses `inference_only`), the dataset (`len` and item
  shape) and nothing heavy; `train` loads components lazily in `weight_dtype`, injects the
  adapter (`LoRAConfig(rank, alpha, dropout, targets, init="gaussian", seed=seed)`), casts LoRA
  parameters to float32, builds AdamW, `LambdaLR` and, for fp16, `torch.amp.GradScaler`.
- Sampling: a data generator (seed) gives per-epoch permutations and crop/flip draws; a noise
  generator (seed + 1) gives latent noise, noise, offset noise and timesteps, drawn on CPU and
  moved to the device. Batches take the next `train_batch_size` indices across epoch borders.
- Step: `gradient_accumulation_steps` micro-batches, each loss divided by the count; unscale and
  clip with `max_grad_norm` (skip when `0`); optimizer, scheduler, zero grads; log every
  `logging_steps` (`step`, `loss`, `lr`) to stdout, `log_history` and wandb when requested.
- Checkpoints every `save_steps` and at the end: `output_dir/checkpoint-N/adapter.safetensors`,
  `adapter_config.json`, `trainer_state.pt` (optimizer, scheduler, scaler, step, epoch, cursor,
  both generator states, log history). Resume reads the latest `checkpoint-*` (or a path) with
  `torch.load(weights_only=True)`.
- `save_model(path)` writes `adapter.safetensors` with `unet.<module>.lora_A.weight` /
  `lora_B.weight` and the `lora_adapter_metadata` header (`unet.r`, `unet.lora_alpha`,
  `unet.target_modules`), plus `adapter_config.json`; returns the safetensors path.
- Unit tests on a tiny snapshot and 3-image folder: loss is finite and decreases over 15 steps at
  a high LR; only LoRA parameters change; 6 straight steps equal 3 + resume + 3 bit for bit;
  accumulation 2 x batch 1 equals batch 2 within 1e-6 when both draw the same samples; bf16 run
  finishes with finite loss; `save_model` output loads through `pipeline(..., lora=)`; refusing an
  `inference_only` model; missing pillow message.

### Task 6: reference parity, docs and PR

- `tests/reference/test_ref_training.py`: Task 3/4 reference tests, plus a trained-and-saved
  adapter that diffusers `load_lora_weights` and imagehub both load to bit-identical images.
- Optional real-weights check (skipped without `IMAGEHUB_SD_V1_5`): first-step loss and LoRA
  gradients on `sd-v1-5` at 256x256 equal diffusers.
- `docs/training.md` (guide + training-support table generated from `list_training_specs`,
  with a test that the table matches), `mkdocs.yml` nav, README "Train" section, design note
  amendment (no `dataloader_num_workers`; clipping note), `docs/validation.md` rows.
- Check chain, push, PR with `Closes #194.` once #193 is merged and the branch is rebased.
