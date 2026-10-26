---
title: Parity
---

# Parity

imagehub adds a model only after it reproduces the original implementation. This section shows
what was measured, how, and where the claim stops.

<div class="ih-stats" markdown>
<div class="ih-stat" markdown><strong>{{ ih.validated }} / {{ ih.models }}</strong><span>checkpoints with recorded end-to-end evidence</span></div>
<div class="ih-stat" markdown><strong>{{ ih.max-pixel-diff }}</strong><span>largest recorded pixel difference</span></div>
<div class="ih-stat" markdown><strong>CPU · fp32</strong><span>where parity is claimed</span></div>
<div class="ih-stat" markdown><strong>{{ ih.gallery-count }}</strong><span>models in the side-by-side gallery</span></div>
</div>

## The method

For each checkpoint, in this order:

1. **Strict load.** Every published tensor is loaded; any missing, extra or mismatched name or
   shape fails the model.
2. **Components.** The tokenizer, text encoders, transformer or U-Net, VAE and scheduler are each
   compared with the reference on the same inputs.
3. **End to end.** A full generation from the same initial noise is compared with the reference:
   final latents and final pixels.
4. **Tolerance.** The accepted difference is written down before the run; for most models it is exactly
   zero.

The recorded reference is the model's implementation in diffusers {{ ih.diffusers }} and
transformers {{ ih.transformers }}, or its official implementation where those libraries do not
cover it. Measurements use PyTorch {{ ih.torch }}.

## Reading the results

| Page | Contents |
| --- | --- |
| [Gallery](gallery.md) | The reference image and the imagehub image side by side, with a slider |
| [Results](results.md) | One row per checkpoint with steps, size and the largest recorded difference |
| [Full validation record](../validation.md) | Method, tolerances, device tests, limits and the bugs found |

## Where the claim stops

!!! warning "Parity has a scope"
    - It holds on **CPU in float32** against the stated library versions.
    - MPS and CUDA runs are smoke tests, not parity.
    - There is no float16 or bfloat16 parity claim.
    - It is not a quality benchmark: no FID or CLIP score was run.
    - There is no safety checker or invisible watermark; the reference was run without one.

The complete list is in the [limits](../validation.md#limits).
