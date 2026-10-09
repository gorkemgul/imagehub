---
title: Quickstart
---

# Quickstart

From nothing to an image in four steps. You need Python 3.11 or newer; PyTorch is the only
required package.

## 1. Install

=== "pip"

    ```bash
    pip install imagehub            # PyTorch only
    pip install "imagehub[image]"   # adds Pillow, for Image.to_pil()
    ```

=== "From source"

    ```bash
    git clone https://github.com/gorkemgul/imagehub
    cd imagehub
    pip install -e ".[image]"
    ```

## 2. Download a model

Downloads happen only when you run this command. It fetches the pinned, validated revision and
verifies every file.

```bash
python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5
```

!!! note "Gated and large models"
    Some repositories (for example `krea2-turbo`) are gated: accept the terms on Hugging Face and
    set `HF_TOKEN`. Weight licenses differ per model; check the
    [catalog](catalog.md) before you use one commercially.

## 3. Generate

```python
from imagehub import pipeline

with pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5") as pipe:
    result = pipe("a photograph of an astronaut riding a horse", seed=42)

result.images[0].save("astronaut.png")
print(result.metadata)   # seed, steps, guidance, size, revision, device, dtype
```

The `with` block loads the weights on entry and releases them on exit. Calling `pipe(...)`
outside a block also works; the model loads on first use.

## 4. Change one thing

=== "Another model"

    ```python
    with pipeline("t2i", backend="flux1-schnell", model="checkpoints/flux1-schnell") as pipe:
        result = pipe("a lighthouse at dusk", seed=7)
    ```

=== "Another device"

    ```python
    pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5", device="mps")
    ```

=== "Generation settings"

    ```python
    pipe("a cat, oil painting", seed=42, steps=30, guidance=6.0, negative_prompt="blurry")
    ```

Settings differ a little between architectures (a distilled model has no guidance, for example).
Each [model page](models/index.md) lists its own, and unknown settings raise an error instead
of being ignored.

## Where next

<div class="ih-features" markdown>
<div markdown>
### Guides
[Devices and precision](guides/devices.md), [image input](guides/image-input.md) and
[reproducibility](guides/reproducibility.md).
</div>
<div markdown>
### Models
[{{ ih.models }} checkpoints](models/index.md) with their settings, licenses and notes.
</div>
<div markdown>
### Parity
[What was measured](parity/index.md), and the [gallery](parity/gallery.md) to look at.
</div>
</div>
