# Training runtime design

## Scope

Extend `imagehub.training` (#194) with the runtime concepts that a
fine-tuning user expects (#200): support levels, per-device batch sizes, a data collator,
evaluation and validation images, and distributed data-parallel training. Names follow the
Hugging Face trainer where a concept exists there. The objective, the LoRA layers and
the parity results of #194 do not change.

Not included: a `preprocessed` support level (training from cached latents and text embeddings)
and a `custom` level, because no model would use them yet; FSDP and DeepSpeed; epochs as an
alternative to `max_steps`.

## Support levels

`TrainingSupport` has `native` (images and captions in, objective verified against the reference
training script) and `inference_only`. `TrainingSpec` gains `methods: tuple[str, ...]`, `("lora",)`
for native models and `()` otherwise, so a later full fine-tune is a new method rather than a new
level. The training-support table shows both columns.

## Batches and the data collator

`train_batch_size` becomes `per_device_train_batch_size`; `per_device_eval_batch_size` is added
(default 1). The effective batch is `per_device_train_batch_size * gradient_accumulation_steps *
world_size`.

`Trainer(..., data_collator=None)`. A collator maps a list of dataset items to
`{"pixel_values": (B, 3, H, W) float in [-1, 1], "input_ids": (B, 77) long}`. The default
`StableDiffusionCollator(tokenizer, resolution=..., center_crop=..., random_flip=...,
generator=...)` does what the trainer did inline: the torchvision-equivalent transform with a
seeded generator, and CLIP tokenization. A custom collator replaces both; exact resume is promised
for the default collator, whose random state the checkpoint holds.

## Evaluation

`Trainer(..., eval_dataset=None)`; `TrainingArguments` gains `eval_strategy` (`"no"` or
`"steps"`), `eval_steps` (defaults to `logging_steps`) and `per_device_eval_batch_size`.

`trainer.evaluate(eval_dataset=None) -> {"eval_loss": float}` runs the training objective without
gradients over the whole dataset in order, with the U-Net in eval mode. Diffusion loss depends on
the sampled noise and timesteps, so every evaluation draws them from a generator reset to the same
seed (and uses a center-cropping collator with the same seed): two evaluations of the same adapter
give the same number, and the curve over training reflects the adapter only. The loss is the mean
over images.

`validation_prompts` (default none), `validation_num_inference_steps` (25) and
`validation_guidance_scale` (7.5): at each evaluation point, and at the end of training, the main
process generates one image per prompt with the checkpoint's scheduler and the current adapter
(seed `seed`) and saves `output_dir/samples/step-N/<index>.png`. Generation reuses the inference
pipeline functions, so it matches `pipeline(..., lora=adapter)`.

Eval loss is added to `log_history`. `report_to` takes `"wandb"`, `"clearml"` or a list of both;
each tracker receives `train_loss` (named as in the diffusers scripts), `lr`, `eval_loss` and the
validation images. The packages stay optional and are imported only when requested.

## Distributed data-parallel training

The same script runs under `torchrun --nproc_per_node=N train.py`. When `WORLD_SIZE > 1`, the
trainer initializes the default process group (`nccl` on CUDA, `gloo` on CPU), places each process
on `cuda:LOCAL_RANK` (or the CPU), and:

- **Data:** every process draws the same epoch permutation; each micro-batch takes the next
  `per_device_train_batch_size * world_size` indices and process `r` keeps its own slice. Crop,
  flip and noise draws come from per-process generators seeded from `seed` and the rank.
- **Gradients:** after the accumulation micro-batches, every LoRA gradient is summed across
  processes and divided by the world size, before unscaling, clipping and the optimizer step. The
  adapter is small, so this explicit all-reduce is used instead of wrapping the U-Net in
  `DistributedDataParallel`; the result is the same averaged gradient. Every process then holds
  identical adapter weights.
- **Logging and files:** the logged loss is the mean over processes; only rank 0 prints, reports to
  wandb, writes checkpoints, samples and `save_model` output; processes meet at a barrier after
  each write.
- **Checkpoints:** rank 0 gathers every process's generator states into `trainer_state.pt`, with
  the world size. Resume requires the same world size and raises otherwise.
- **Evaluation:** each process evaluates a contiguous shard; loss sums and image counts are
  all-reduced.

MPS does not support process groups; distributed runs use CUDA or the CPU.

## LoRA rank and alpha

Defaults stay at rank 4, alpha 4 (scale 1), the diffusers training scripts' choice and the basis of
the parity tests. Diffusion tools use small ranks and scales near 1 (kohya sd-scripts: rank 4,
alpha 1), while language and speech trainers use larger ones (PEFT: 8 / 8). The
guide lists these and suggests raising the rank to 16-32 for more detail, keeping alpha equal to
the rank.

## Validation

All on CPU in float32, without GPUs:

- unit tests for the profiles, renamed arguments, the default collator (equal to the previous
  inline batches) and a custom collator;
- `evaluate()` is repeatable, independent of the training random state, and changes after
  training; validation images are written and equal `pipeline(..., lora=)` output;
- two-process gloo runs started with `torch.multiprocessing`: both processes end with identical
  adapters; replaying each process's recorded batches and noise in one process with averaged
  gradients gives the same adapter; resume from a two-process checkpoint equals an uninterrupted
  run; resuming with another world size raises; only one checkpoint folder is written.

A multi-GPU NCCL run is not part of this change; it needs rented hardware and is done only with
the user's go-ahead.
