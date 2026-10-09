# Reproducibility

## Same seed, same image

Pass `seed` and the initial noise is drawn on the **CPU** with `torch.Generator("cpu").manual_seed(seed)`,
the way the reference does it (shown for Stable Diffusion; each [model page](../models/index.md)
notes where an architecture differs), whatever device runs the model.

```python
a = pipe("a cat", seed=1).images[0]
b = pipe("a cat", seed=1).images[0]
assert a.tensor.equal(b.tensor)
```

Leave `seed` out and a random one is chosen and recorded in `result.metadata`, so an image can be
repeated afterwards. Pass either `seed` or `latents` (your own noise tensor), not both.

## Same image as the reference

For the recorded checkpoints, on CPU in float32, with diffusers {{ ih.diffusers }} and
transformers {{ ih.transformers }}, imagehub produces the **same pixels** as the reference. That
claim comes with its evidence:

- [Results](../parity/results.md): the recorded differences, one row per model.
- [Gallery](../parity/gallery.md): the reference and imagehub images side by side.
- [Validation record](../validation.md): the method, the tolerances and the bugs found on the way.

## What can still differ

- A different device or precision: see [Devices and precision](devices.md).
- A different library version of the reference: the record names the versions it was measured with.
- Different weights: the catalog pins a revision; another revision is not covered.

!!! info "Alignment matters"
    Safetensors files are memory-mapped so weights are read at their published alignment. CPU BLAS can
    pick a different summation order for differently aligned data, which was one of the real sources
    of a mismatch found during validation.
