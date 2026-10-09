# Image input and FLUX.1 Kontext design

- Date: 2026-10-08
- Issue: #44
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Decision: image input is a backend option of `t2i`, not a new task

imagehub keeps one generation task, `t2i`. A model that can take an image accepts it as a backend option, in the same
way backends already accept `steps`, `guidance` or `seed`:

```python
from PIL import Image
from imagehub import pipeline

source = Image.open("cat.png")

edit = pipeline("t2i", model="flux1-kontext-dev")
result = edit("put a red hat on the cat", image=source, seed=0)
```

Reasons:

- The catalog this project links to describes such models with the same entry as their text-to-image use: inputs
  `T, I`, output `I`, interaction `editing`. One task with an optional image input mirrors that.
- The pipeline API does not grow: the request, defaults and option precedence (`default_options` < request parameters
  < call options) stay as they are, and `image` travels as an ordinary option.
- Instruction editing (this design) and classic image-to-image with a `strength` (possible later for SD 1.x, SDXL and
  FLUX.1) fit the same shape; they differ only in the options a backend accepts.

Rules:

- Only backends that implement an image input accept `image`; every other backend keeps rejecting it as an unknown
  option with `TypeError`, before loading.
- `image` accepts a `PIL.Image.Image` (the `image` extra) or an `imagehub.Image`. The backend converts it exactly as the
  reference pipeline preprocesses it; any resizing is part of the parity target.
  Non-RGB PIL images (RGBA, L, …) are converted with `.convert("RGB")` first; the reference pipeline does not convert
  them and fails on them, so parity is defined on the converted image.
- Whether `image` is optional or required is a property of the model and is documented. For Kontext it is optional:
  without an image the reference pipeline runs plain text-to-image, and so does ours.
- Result metadata records whether an input image was used and its preprocessed size; it never stores the image.
- The catalog records the input per model: `ModelSpec.image_input` is `None` (text only, the default), `"optional"` or
  `"required"`, so the image-input table and a programmatic query come from one place.

## Documentation

Both forms:

- every model page in `docs/models.md` states in prose which image options it takes (or that it takes none);
- a table at the top of `docs/models.md` lists the models that accept an image:

| Model | Image input | Use | Options |
| --- | --- | --- | --- |
| `flux1-kontext-dev` | optional | instruction editing | `image` |

The README model table gets an input column (`T` or `T, I`) matching the catalog's labels.

## Model: FLUX.1 Kontext [dev]

`flux1-kontext-dev` = `black-forest-labs/FLUX.1-Kontext-dev` (revision `24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d`), by
the authors, gated, `FLUX.1-dev Non-Commercial License`, catalog id `flux-1-kontext`, family `flux1` (the existing
backend gains the image path; text-to-image stays unchanged).

Reference: diffusers 0.40.0 `FluxKontextPipeline` (`pipelines/flux/pipeline_flux_kontext.py`).

| Part | Status |
| --- | --- |
| CLIP-L and T5-XXL text encoders, tokenizers | exist (FLUX.1) |
| FLUX.1 transformer | exists unchanged: the published config and all 1160 tensor names equal FLUX.1-dev's. The pipeline appends the image tokens to the sequence with their own position ids (first id component 1) and keeps only the generated tokens of the output |
| VAE encoder (mode of the posterior, shift and scale) | the encoder exists; the FLUX latent normalization on encode is checked |
| Input preprocessing: snap to the preferred Kontext resolution closest to the input's aspect ratio (`PREFERRED_KONTEXT_RESOLUTIONS`, `max_area = 1024²`, multiples of 16), resize as diffusers' `VaeImageProcessor`, normalize to `[-1, 1]` | **new** |
| Output size: `height`/`width` (default 1024×1024), rescaled to `max_area` at the same aspect ratio and rounded down to multiples of 16, independent of the input image size | **new** |

Defaults as the `FluxKontextPipeline` signature: 28 steps, guidance 3.5, `max_sequence_length` 512, `true_cfg_scale`
1.0 (no negative prompt). The model card's examples use guidance 2.5; that stays a call option.

Out of scope: inpainting (`FluxKontextInpaintPipeline`), several input images, IP-Adapter, true CFG with a negative
prompt, fp16/bf16 parity.

## Memory

fp32: transformer about 47.6 GB, T5-XXL about 19 GB, CLIP-L 0.5 GB, VAE 0.3 GB (as FLUX.1-dev). The reference and the
native model cannot both hold the transformer in one process on a 96 GB host: compare the transformer save-then-compare
or in a child process, as for FLUX.1-dev and Ideogram 4.

## Validation

Random-weight parity for the image-token path of the transformer (tiny, with image tokens and position ids) and for
preprocessing over several input sizes and aspect ratios (bytes of the resized image and the normalized tensor); a tiny
end to end against `FluxKontextPipeline` with and without an input image; then real-checkpoint component and
end-to-end parity at the defaults with a fixed input image, plus the seed test and the gallery sample. Existing FLUX.1
tests must stay exact.

## Task breakdown

1. Image input plumbing (the `image` option, `imagehub.Image`/PIL conversion, `ModelSpec.image_input`, rejection
   elsewhere), Kontext preprocessing and the VAE encode, with parity.
2. The Kontext path in the FLUX.1 backend (image tokens and ids, output slicing, size defaults); catalog entry; tiny end
   to end with and without an image.
3. Real-checkpoint parity; docs (the image-input table, the README input column, the model page); the pull request
   closes #44.
