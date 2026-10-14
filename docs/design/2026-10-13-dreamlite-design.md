# DreamLite design

- Date: 2026-10-13
- Issue: #125

## Scope

`dreamlite-base` and `dreamlite-mobile` are author-published diffusers snapshots.  They use
the same bfloat16 tensor layout: Qwen3-VL 2B conditioning, a DreamLite mobile U-Net, TinyVAE,
and FlowMatch Euler sampling.  The base revision is
`a9a0f151ffd99d3c37f3fd0472f5e8f1b31215aa`; the mobile revision is
`a48c656c291fe98f74a0405961c9e4e3eee5d6c8`.  Both are CC-BY-NC-4.0, which prohibits commercial use.

## Reference and behavior

The reference is diffusers 0.40.0 `DreamLitePipeline` and `DreamLiteMobilePipeline`.
Both accept text generation and image-guided editing.  They concatenate target and condition
latents horizontally, use a flow-matching Euler schedule with dynamic shift, and decode through
`AutoencoderTiny`.  Base defaults are 30 steps, text guidance 3.5 and image guidance 1.5.
Mobile defaults are 4 steps and do not apply classifier-free guidance.

## Components and validation

The existing Qwen3-VL text implementation, FlowMatch Euler scheduler, and tokenizer assets can
be reused after their checkpoint layouts are verified.  The DreamLite U-Net requires its own
native implementation: depthwise-separable residual convolutions, grouped-query attention,
RMS-normalized Q/K, and the no-self-attention blocks are not represented by the existing U-Net.
TinyVAE also needs a native module.  Required validation is random-weight component parity,
tiny end-to-end parity for each pipeline mode, then real-checkpoint header, component, end-to-end,
and seed checks on the bfloat16 snapshots.

## Memory and exclusions

The Qwen3-VL conditioning model dominates memory; component-by-component CPU validation should
free the reference before loading the native component.  Safety checking, watermarks, batching
beyond the reference's forced batch size of one, and image editing API design beyond the reference
pipeline are out of scope.
