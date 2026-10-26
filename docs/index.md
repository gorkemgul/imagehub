---
title: imagehub
hide:
  - navigation
  - toc
---

# imagehub

<div class="ih-hero" markdown>

<!-- ih:banner -->

<p class="ih-eyebrow">Open-weight text-to-image models · pure PyTorch</p>

## Model catalog and pipeline API

<p class="ih-lead">
imagehub reimplements open text-to-image architectures natively in PyTorch and checks every
model against its reference implementation. On CPU in float32 the output is
bit-identical to the recorded reference implementation.
</p>

<div class="ih-cta" markdown>
[Get started](quickstart.md){ .md-button .md-button--primary }
[Browse the models](models/index.md){ .md-button }
[See the parity record](parity/index.md){ .md-button }
</div>

</div>

<div class="ih-stats" markdown>
<div class="ih-stat" markdown><strong>{{ ih.models }}</strong><span>published checkpoints</span></div>
<div class="ih-stat" markdown><strong>{{ ih.families }}</strong><span>architecture families</span></div>
<div class="ih-stat" markdown><strong>{{ ih.validated }}</strong><span>with recorded parity</span></div>
<div class="ih-stat" markdown><strong>1</strong><span>runtime dependency: PyTorch</span></div>
</div>

<div class="ih-split" markdown>
<div markdown>

## Pipeline interface

The model id selects the architecture; the folder selects the weights. Switching from
Stable Diffusion to FLUX or Qwen-Image changes one string.

- Construction never touches the network; downloading is a separate, explicit step.
- Settings are validated before any weight is loaded, and unknown ones raise an error.
- The same seed gives the same image as the reference implementation.

</div>
<div markdown>

```python
from imagehub import pipeline

with pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5") as pipe:
    result = pipe("a photograph of an astronaut riding a horse", seed=42)

result.images[0].save("astronaut.png")
print(result.metadata)
```

</div>
</div>

## Implementation properties

<div class="ih-features" markdown>
<div markdown>
### Recorded parity
Each model is accepted only after tensors load strictly and a full generation from the same noise
matches the reference. The differences are recorded, not asserted.
</div>
<div markdown>
### Readable architectures
Every model is plain PyTorch in this repository: no hidden dependency on diffusers or
transformers at inference time.
</div>
<div markdown>
### Pinned and verified weights
Catalog entries pin a commit and file list. Downloads check sizes and hashes and write a
manifest of what you got.
</div>
<div markdown>
### Validation scope
Parity is claimed for CPU and float32. Other devices and precisions are smoke-tested and labelled
as such.
</div>
</div>

## The models

<!-- ih:model-grid -->

## Same seed, same image

Drag the slider: the left side is the reference implementation, the right side is imagehub.

<div class="ih-gallery" markdown>
<!-- ih:compare:pixart-sigma-xl-1024 -->
<!-- ih:compare:kolors -->
<!-- ih:compare:flux1-krea-dev -->
</div>

[Open the full gallery](parity/gallery.md){ .md-button }
[How parity is measured](parity/index.md){ .md-button }
