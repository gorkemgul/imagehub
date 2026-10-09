# Devices and precision

```python
pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5", device="cpu")   # default
pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5", device="mps")   # Apple silicon
pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5", device="cuda")  # NVIDIA
```

Asking for a device that is not available raises an error that lists what is.

## Precision

`dtype` accepts `torch.float32` (default), `torch.float16` or `torch.bfloat16` where a model
supports them:

```python
import torch
pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5", device="cuda", dtype=torch.float16)
```

## What is claimed where

| Device | Precision | Claim |
| --- | --- | --- |
| CPU | float32 | **Parity**: bit-identical to the reference for the recorded checkpoints |
| MPS | float32 | Smoke test: finite output, right size, difference from CPU recorded |
| CUDA | float32 | Not tested yet |
| any | float16 / bfloat16 | No parity claim |

!!! warning "Fast is not the same as identical"
    Different hardware sums floating-point numbers in a different order, so GPU output can differ
    from CPU output by a few pixel values. imagehub records that difference where it was measured and
    does not call it parity.

The full numbers are in the [validation record](../validation.md#device-smoke-tests-no-parity-claim).
