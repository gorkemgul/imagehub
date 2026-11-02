# imagehub Goal

imagehub runs open-weight image generation architectures behind one API, with every
architecture natively reimplemented in PyTorch and proven to reproduce the original
implementation's output.

## Scope

- Source of truth for which architectures exist: the
  [awesome-text-to-image-architectures](https://github.com/gorkemgul/awesome-text-to-image-architectures)
  catalog (`data/models.json`). Every architecture there with released weights is in scope,
  including ones that diffusers or transformers already support.
- Families: latent U-Net diffusion, diffusion transformers, autoregressive token models,
  continuous autoregressive models, masked generative models, unified understanding and
  generation models, pixel-space diffusion and efficient variants.
- Out of scope until the goal says otherwise: training, serving, CUDA-specific kernels,
  safety checkers and watermarking.

## Priorities

When they conflict, the earlier one wins:

1. **Correctness.** A model produces the same output as its reference implementation. Published
   revisions and file hashes are pinned. No model ships without parity evidence.
2. **Readability.** Plain PyTorch modules that keep the published tensor names, one clear
   responsibility per file, and every architecture linked to its catalog entry.
3. **Ease of use.** One `pipeline(task, backend=..., model=...)` API, one short page per model,
   and a PyPI release at each milestone.

## What "supported" means

A model is listed in `imagehub/catalog.py` with a validation status:

| Status | Evidence required |
| --- | --- |
| `full` | Strict load of the published tensors, exact or measured-tolerance parity of every component and of end-to-end generation against the reference on CPU float32, recorded in `docs/validation.json`. |
| `header_only` | Every published tensor name and shape matches the native model (meta device); generation not compared. |
| `experimental` | Architecture and unit tests exist; the original weights were not tested. |

Only `full` models make parity claims. Other devices and dtypes get smoke tests, never parity
claims, unless parity is measured and recorded for them.

## Invariants

- The only runtime dependency is `torch`. Everything else in `imagehub` is the Python
  standard library; optional packages are imported lazily behind an extra (`image` -> Pillow).
- Reference packages (diffusers, transformers, original repositories) are used only in
  `tests/reference/` and never imported by the library.
- `pipeline()` never touches the network. Downloads happen only through
  `python -m imagehub.pretrained`, pinned to the validated commit and verified file by file.
- Checkpoint loading is strict: missing, unexpected or mis-shaped tensors are errors, and
  nothing is randomly initialized.
- Model code has no device-specific branches and no float64; noise is drawn on CPU.

## Completion criteria for a milestone

1. Every new model is `full`, with its measurements in `docs/validation.json` and a summary in
   `docs/validation.md`.
2. Unit tests pass on the newest Python/PyTorch and on the minimum supported versions
   (Python 3.11, `torch==2.2.*`) from an installed wheel without optional extras.
3. ruff and mypy are clean.
4. The catalog page, the model page and the README model table are updated.
5. The sdist and wheel contain only the published files.
6. CUDA smoke tests and the PyPI release happen with the maintainer's approval.

## Roadmap

The order weighs two things: demand (Hugging Face downloads, likes and trending, audited
2026-10-04; fine-tunes and conversions count toward their base architecture) and reuse (each
step builds infrastructure a later one needs). CPU float32 parity keeps the native and the
reference model in memory at once (about parameters × 4 bytes × 2), so on the 96 GB development
machine models up to roughly 10B parameters are validated end to end; larger ones need
component-by-component parity or a rented high-memory machine.

1. Stable Diffusion 1.x — done (`sd-v1-1` … `sd-v1-5`).
2. Shared-core prerequisites — done: non-persistent buffers that survive meta-device construction,
   checkpoint containers that nest weights (for example under `model` or `ema`), and tokenizer
   special tokens read from the tokenizer config.
3. SDXL family (2.6B) — done: SDXL base, SDXL Turbo, SSD-1B, Playground v2.5.
4. Z-Image-Turbo (6.2B) — done: the diffusion-transformer template.
5. Janus-Pro (1B, 7B) — done: the autoregressive and unified template.
6. Stable Diffusion 3.5 (medium, large, large-turbo) — done: MMDiT and the native T5 encoder.
7. Qwen-Image 2.1 (7.1B) — done, against a pinned diffusers commit in a separate reference
   environment (no released diffusers version contained it yet).

Re-audited 2026-10-05 against Hugging Face downloads and the awesome catalog; the per-model issues
are tracked in the roadmap issue on GitHub.

8. Checkpoints of existing or nearly existing families: SD-Turbo, SDXL-Lightning, LCM,
   Stable Diffusion 3 medium, Z-Image (base).
9. New architectures with high demand that fit the memory budget: FLUX.2 klein (4B, 9B),
   HunyuanDiT; then FLUX.1 schnell and dev (11.9B), which need a component-level or streamed
   parity approach.
10. Second tier: Hyper-SD, DMD2, PixArt-Sigma and PixArt-Alpha, Krea-2, Ideogram 4, LongCat-Image,
    Stable Cascade, CogView4, Kandinsky 2.2.
11. On hold: image editing (FLUX.1 Kontext; needs a new task), Sana, a masked model (Meissonic or
    aMUSEd), Infinity, LlamaGen. The remaining catalog architectures (mostly research models with
    low demand) follow in demand order.
