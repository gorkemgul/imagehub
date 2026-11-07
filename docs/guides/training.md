# LoRA training

`imagehub.training` fine-tunes a LoRA adapter on your own images. It ships its own LoRA layers and
training loop (no PEFT or Accelerate) and is an optional install; the core package stays
PyTorch-only.

## 1. Install and download

```bash
pip install "imagehub[training]"
```

```bash
python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5
```

## 2. Prepare the images

Put the images in one folder with a `metadata.jsonl` file that has one line per image:

```
my_images/
  metadata.jsonl
  0001.png
  0002.jpg
```

```json
{"file_name": "0001.png", "text": "a photo of sks dog on the beach"}
{"file_name": "0002.jpg", "text": "a photo of sks dog in the snow"}
```

Missing files, empty captions and unreadable images are reported when the dataset is created.
Any sized dataset whose items are `{"image": PIL image, "text": str}` also works, for example a
Hugging Face `datasets` dataset; imagehub does not depend on that package.

## 3. Train

```python
from imagehub.training import ImageFolderDataset, Trainer, TrainingArguments

trainer = Trainer(
    model="sd-v1-5",
    checkpoint="checkpoints/sd-v1-5",
    args=TrainingArguments(output_dir="runs/my-dog", max_steps=1000),
    train_dataset=ImageFolderDataset("my_images"),
)
result = trainer.train()
print(result.training_loss)
trainer.save_model("runs/my-dog/final")
```

The loss is printed every `logging_steps`. Every `save_steps` steps, and at the end, the trainer
writes `output_dir/checkpoint-N/` with the adapter and the full training state. If a run stops,
continue it from the latest checkpoint:

```python
trainer.train(resume_from_checkpoint=True)
```

A resumed run gives exactly the same adapter as an uninterrupted one.

## 4. Use the adapter

```python
from imagehub import pipeline

with pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5",
              lora="runs/my-dog/final/adapter.safetensors") as pipe:
    image = pipe("a photo of sks dog in a bucket", seed=0).images[0]
```

`adapter.safetensors` uses the diffusers layout, so diffusers `load_lora_weights` reads it too.

## Evaluation and sample images

```python
trainer = Trainer(
    model="sd-v1-5",
    checkpoint="checkpoints/sd-v1-5",
    args=TrainingArguments(
        output_dir="runs/my-dog",
        max_steps=1000,
        eval_strategy="steps",
        eval_steps=200,
        validation_prompts=["a photo of sks dog in a bucket"],
    ),
    train_dataset=ImageFolderDataset("my_images"),
    eval_dataset=ImageFolderDataset("my_val_images"),
)
trainer.train()
print(trainer.evaluate())   # {"eval_loss": 0.1234}
```

`eval_loss` is the training loss without gradients over the evaluation images. Diffusion loss
depends on the random noise and timestep, so these are seeded by image position and the images
are center-cropped: evaluating the same adapter twice gives the same number, and the curve over
training follows the adapter, not the dice. At every evaluation point (and at the end) each
validation prompt is rendered with the current adapter to `output_dir/samples/step-N/<i>.png`,
exactly as `pipeline(..., lora=...)` would render it with `seed`, `validation_num_inference_steps`
and `validation_guidance_scale`.

## Custom batches

`data_collator` turns a list of dataset items into a batch. The default resizes, crops, flips and
normalizes the images and tokenizes the captions. A custom one returns
`{"pixel_values": (B, 3, H, W) in [-1, 1], "input_ids": (B, 77)}`:

```python
trainer = Trainer(..., data_collator=my_collator)
```

Exact resume is guaranteed for the default collator, whose random state the checkpoint holds.

## Several GPUs

Run the same script with `torchrun`; nothing in it changes:

```bash
torchrun --nproc_per_node=4 train.py
```

Each process trains on its own share of every batch and the adapter gradients are averaged, so all
processes keep the same adapter. The effective batch is `per_device_train_batch_size ×
gradient_accumulation_steps × number of processes`. Only the first process prints, writes
checkpoints, samples and `save_model` output. A checkpoint resumes with the same number of
processes. Distributed runs use CUDA (NCCL) or the CPU (gloo); MPS cannot.

## Settings

Only `output_dir` and `max_steps` are required. The defaults follow diffusers'
`train_text_to_image_lora.py`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `learning_rate` | `1e-4` | AdamW learning rate |
| `per_device_train_batch_size`, `per_device_eval_batch_size` | `1`, `1` | images per micro-batch on each process |
| `gradient_accumulation_steps` | `1` | micro-batches per optimizer step |
| `lr_scheduler`, `lr_warmup_steps` | `"constant"`, `0` | also `constant_with_warmup`, `linear`, `cosine` |
| `max_grad_norm` | `1.0` | gradient clipping; `0` turns it off |
| `adam_beta1`, `adam_beta2`, `adam_weight_decay`, `adam_epsilon` | `0.9`, `0.999`, `1e-2`, `1e-8` | AdamW settings |
| `mixed_precision` | `"no"` | `"fp16"` or `"bf16"`; the adapter and optimizer stay float32 |
| `resolution`, `center_crop`, `random_flip` | `512`, `False`, `False` | image transform |
| `lora_rank`, `lora_alpha`, `lora_dropout` | `4`, `4`, `0.0` | adapter size and scale (`alpha / rank`) |
| `lora_target_modules` | the family's | layers that get an adapter: `to_q`, `to_k`, `to_v`, `to_out.0` for U-Nets, the eight joint-attention projections for SD 3, and `to_k`, `to_q`, `to_v` for Sana |
| `gradient_checkpointing` | `False` | recompute U-Net or transformer blocks in the backward pass to save memory |
| `noise_offset`, `snr_gamma`, `prediction_type` | `0.0`, `None`, the scheduler's | U-Net (DDPM) objective options |
| `weighting_scheme`, `logit_mean`, `logit_std`, `mode_scale` | the family's (`"logit_normal"` for SD 3, `"none"` for FLUX.1 and Sana), `0.0`, `1.0`, `1.29` | flow-matching timestep sampling: also `mode`, `sigma_sqrt`, `cosmap`, `none` |
| `precondition_outputs` | `True` | flow matching: compare the preconditioned output with the latents instead of `noise - latents` |
| `max_sequence_length` | `77` (SD 3), `512` (FLUX.1), `300` (Sana) | text prompt length |
| `complex_human_instruction` | `None` | Sana's optional prompt-enhancement instruction; other families reject it |
| `guidance_scale` | `3.5` | FLUX.1: the guidance embedding's value in training, on checkpoints that have one |
| `logging_steps`, `save_steps` | `10`, `500` | how often to log and to checkpoint |
| `eval_strategy`, `eval_steps` | `"no"`, `logging_steps` | `"steps"` evaluates every `eval_steps` |
| `validation_prompts` | none | prompts rendered at each evaluation point |
| `validation_num_inference_steps`, `validation_guidance_scale` | `25`, `7.5` | sampling settings for those images |
| `report_to`, `run_name` | `"none"`, `None` | `"wandb"`, `"clearml"` or both as a list: logs `train_loss`, `lr`, `eval_loss` and the sample images (the package must be installed) |
| `seed`, `device` | `0`, `"cpu"` | random seed; `"cuda"` or `"mps"` to train on a GPU |

## Choosing rank and alpha

The adapter's update is scaled by `lora_alpha / lora_rank`. Diffusion tools use small ranks and
scales near 1, while language and speech trainers tend to use larger ones:

| Source | Rank | Alpha | Scale |
| --- | --- | --- | --- |
| imagehub default, diffusers training scripts (SD 1.5, SDXL, FLUX.1) | 4 | 4 | 1 |
| kohya sd-scripts default | 4 | 1 | 0.25 |
| PEFT default | 8 | 8 | 1 |

Rank 4 is enough for a style or a simple subject. For more detail, raise the rank to 16-32 and keep
alpha equal to the rank, so the scale and a good learning rate stay the same.

## Supported models

`get_training_spec(model_id)` tells whether a model can be trained; `list_training_specs()` lists
every catalog model.

```python
from imagehub.training import get_training_spec

spec = get_training_spec("sd-v1-5")
print(spec.support.value, spec.methods)   # native ('lora',)
```

`native` means the model trains from images and captions with an objective checked against the
reference training script; `inference_only` means there is no training path yet. `methods` lists
what can be trained (`lora` today).

<!-- training-support:start -->
| Model | Support | Methods |
| --- | --- | --- |
| `sd-v1-1` | native | lora |
| `sd-v1-2` | native | lora |
| `sd-v1-3` | native | lora |
| `sd-v1-4` | native | lora |
| `sd-v1-5` | native | lora |
| `dreamshaper-8` | native | lora |
| `segmind-small-sd` | native | lora |
| `segmind-tiny-sd` | native | lora |
| `bk-sdm-base` | native | lora |
| `bk-sdm-small` | native | lora |
| `bk-sdm-tiny` | native | lora |
| `sd-v2-1` | native | lora |
| `bk-sdm-v2-base` | native | lora |
| `bk-sdm-v2-small` | native | lora |
| `bk-sdm-v2-tiny` | native | lora |
| `sdxl-base-1.0` | native | lora |
| `juggernaut-xl-v9` | native | lora |
| `animagine-xl-4.0` | native | lora |
| `dreamshaper-xl-1.0` | native | lora |
| `sd3-medium` | native | lora |
| `sd3.5-medium` | native | lora |
| `sd3.5-large` | native | lora |
| `flux1-dev` | native | lora |
| `flux1-krea-dev` | native | lora |
| `flux-1-lite-8b` | native | lora |
| `flux1-kontext-dev` | native | lora |
| `sana-1600m-1024` | native | lora |
| `sana-600m-1024` | native | lora |
| `sana-1.5-4.8b-1024` | native | lora |
| `ssd-1b` | native | lora |
| `segmind-vega` | native | lora |
| `koala-700m` | native | lora |
| `koala-1b` | native | lora |
| `playground-v2` | native | lora |
<!-- training-support:end -->

Every other model is `inference_only` for now: the few-step distilled checkpoints (`sd-turbo`,
`lcm-dreamshaper-v7`, `sdxs-512-0.9`, SDXL Lightning, Hyper-SD, DMD2, `sdxl-turbo`, KOALA
Lightning), the SDXL refiner (image to image), Playground v2.5 (EDM, not DDPM) and
`sd3.5-large-turbo` (few-step distilled), and `sana-sprint-1.6b-1024` (distilled). More
transformer families follow (#202).

Sana trains its transformer only, using the script's `to_k`, `to_q`, `to_v` targets. Its DC-AE
has deterministic latents, so the training random stream draws noise then the flow timestep; the
autoencoder stays float32 under mixed precision. Gemma 2 prompt conditioning uses Sana's attention
mask and 300-token default; `complex_human_instruction` is optional and defaults to `None` as in
the script. Gradient checkpointing recomputes the Sana transformer blocks.

For SDXL the transform resizes, flips and then crops, as diffusers' SDXL script does, and records
each image's original size and crop corner as the U-Net's size conditioning. The autoencoder stays
in float32 under mixed precision. `gradient_checkpointing=True` recomputes the U-Net's transformer
and resnet blocks in the backward pass to save memory; the gradients are the same.

SD 3 / 3.5 train with the flow-matching objective of diffusers' `train_dreambooth_lora_sd3.py`:
timesteps drawn by `weighting_scheme` from the shifted schedule, the noisy latents
`(1 - sigma) * latents + sigma * noise`, and preconditioned outputs compared with the latents. The
prompt joins the CLIP-L and CLIP-G penultimate states with T5, as at inference. DDPM options
(`snr_gamma`, `prediction_type`, `noise_offset`) are refused for SD 3, and the flow-matching
options for the U-Net families.

FLUX.1 dev, Krea dev and Lite train with the corresponding diffusers FLUX DreamBooth step:
float32 VAE latents are shifted and scaled, packed with FLUX image ids, and compared directly with
`noise - latents`. The CLIP pooled projection and T5 prompt use a 512-token default. The guidance
embedding uses `guidance_scale` only on checkpoints that expose it. The adapter targets cover both
double-stream and single-stream blocks. `flux1-kontext-dev` trains the same step, as the Kontext
script does without condition images, and also targets the single-stream `proj_mlp` layers;
image-conditioned Kontext training is not included. `flux1-schnell` is few-step distilled and
stays `inference_only`.

```python
trainer = Trainer(
    model="sd3.5-medium",
    checkpoint="checkpoints/sd3.5-medium",
    args=TrainingArguments(
        output_dir="runs/my-dog-sd35",
        max_steps=1000,
        resolution=1024,
        mixed_precision="bf16",
        gradient_checkpointing=True,
        weighting_scheme="logit_normal",
    ),
    train_dataset=ImageFolderDataset("my_images"),
)
```

## What is checked

On CPU in float32, the first step's loss and every LoRA gradient equal the diffusers training step
exactly, on tiny models and on `sd-v1-5`; four trainer steps with warmup, a cosine schedule and
clipping equal a diffusers + PEFT loop; and a saved adapter gives the same image in imagehub and
diffusers. See the [validation record](../validation.md#lora-training).
