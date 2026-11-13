# Changelog

## 0.1.0

The first public release.

### Models

- 123 checkpoints in 53 families behind one `pipeline("t2i", backend=..., model=...)` API, from
  Stable Diffusion 1.x to FLUX.1, FLUX.2 klein, SD3, Sana, PixArt, Qwen-Image, HiDream-I1, Krea 2,
  Ideogram 4 and autoregressive models such as LlamaGen, Janus-Pro and Infinity.
- Every architecture is reimplemented in native PyTorch; PyTorch is the only required runtime
  dependency.
- Some backends take an optional `image=` input for image-conditioned generation and editing
  (FLUX.1 Kontext, Qwen-Image-Edit, OmniGen, VisualCloze and others).
- `python -m imagehub.pretrained <model-id> <folder>` downloads a checkpoint at the revision pinned
  in the catalog.
- Creating a pipeline for a checkpoint without commercial-use rights emits an
  `ImageHubLicenseWarning`.

### Validation

- Every listed checkpoint is checked against its reference implementation (diffusers,
  transformers or the authors' code). CPU float32 image outputs match bit for bit. The validation
  record lists the settings and results for each checkpoint.

### LoRA and training

- LoRA adapters load with `lora=` and `lora_scale=` on 12 backends: Stable Diffusion 1.x and 2.x,
  SDXL, SD3, FLUX.1, FLUX.2 klein, Sana, Qwen-Image, Z-Image, Lumina 2, HiDream-I1, Krea 2 and
  Ideogram 4.
- `imagehub[training]` adds LoRA fine-tuning (`Trainer`, `TrainingArguments`,
  `ImageFolderDataset`) for eleven families; the training guide lists every trainable checkpoint.

### Documentation

- A documentation site in English and Turkish: model pages, the model catalog, the validation
  record and a side-by-side gallery of reference and imagehub outputs.

### Known limits

- Validation runs on CPU in float32. GPU and half-precision outputs are not validated yet.
- Lumina-mGPT 2.0 is not included yet.
