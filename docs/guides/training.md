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

## Settings

Only `output_dir` and `max_steps` are required. The defaults follow diffusers'
`train_text_to_image_lora.py`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `learning_rate` | `1e-4` | AdamW learning rate |
| `train_batch_size` | `1` | images per micro-batch |
| `gradient_accumulation_steps` | `1` | micro-batches per optimizer step |
| `lr_scheduler`, `lr_warmup_steps` | `"constant"`, `0` | also `constant_with_warmup`, `linear`, `cosine` |
| `max_grad_norm` | `1.0` | gradient clipping; `0` turns it off |
| `adam_beta1`, `adam_beta2`, `adam_weight_decay`, `adam_epsilon` | `0.9`, `0.999`, `1e-2`, `1e-8` | AdamW settings |
| `mixed_precision` | `"no"` | `"fp16"` or `"bf16"`; the adapter and optimizer stay float32 |
| `resolution`, `center_crop`, `random_flip` | `512`, `False`, `False` | image transform |
| `lora_rank`, `lora_alpha`, `lora_dropout` | `4`, `4`, `0.0` | adapter size and scale (`alpha / rank`) |
| `lora_target_modules` | `to_q`, `to_k`, `to_v`, `to_out.0` | U-Net layers that get an adapter |
| `noise_offset`, `snr_gamma`, `prediction_type` | `0.0`, `None`, the scheduler's | objective options |
| `logging_steps`, `save_steps` | `10`, `500` | how often to log and to checkpoint |
| `report_to`, `run_name` | `"none"`, `None` | `"wandb"` also logs to Weights & Biases if installed |
| `seed`, `device` | `0`, `"cpu"` | random seed; `"cuda"` or `"mps"` to train on a GPU |

## Supported models

`get_training_spec(model_id)` tells whether a model can be trained; `list_training_specs()` lists
every catalog model.

```python
from imagehub.training import get_training_spec

spec = get_training_spec("sd-v1-5")
print(spec.support.value, spec.family)   # lora stable-diffusion
```

<!-- training-support:start -->
| Model | Support |
| --- | --- |
| `sd-v1-1` | lora |
| `sd-v1-2` | lora |
| `sd-v1-3` | lora |
| `sd-v1-4` | lora |
| `sd-v1-5` | lora |
| `dreamshaper-8` | lora |
| `segmind-small-sd` | lora |
| `segmind-tiny-sd` | lora |
| `bk-sdm-base` | lora |
| `bk-sdm-small` | lora |
| `bk-sdm-tiny` | lora |
| `sd-v2-1` | lora |
| `bk-sdm-v2-base` | lora |
| `bk-sdm-v2-small` | lora |
| `bk-sdm-v2-tiny` | lora |
<!-- training-support:end -->

Every other model is `inference_only` for now, including the few-step distilled `sd-turbo`,
`lcm-dreamshaper-v7` and `sdxs-512-0.9`. SDXL and the transformer families come next.

## What is checked

On CPU in float32, the first step's loss and every LoRA gradient equal the diffusers training step
exactly, on tiny models and on `sd-v1-5`; four trainer steps with warmup, a cosine schedule and
clipping equal a diffusers + PEFT loop; and a saved adapter gives the same image in imagehub and
diffusers. See the [validation record](../validation.md#lora-training).
