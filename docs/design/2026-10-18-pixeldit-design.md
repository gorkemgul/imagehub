# PixelDiT design

- Date: 2026-10-18
- Issue: #137
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

One author weight set, `pixeldit`, for text-to-image generation in pixel space. The PixelDiT
weights are `nvidia/PixelDiT-1300M-1024px` revision
`7c63b99a7a399918a1d6478b095698a65f664847` (NSCLv1), and the Gemma-2 text encoder is
`Efficient-Large-Model/gemma-2-2b-it` revision
`569d9809d0c8b6722d4d31b5a77a2ec7a400650a` (Gemma license). Both are the repositories selected
by the official inference code. The official implementation is `NVlabs/PixelDiT` revision
`f7d7c35f71893d81c6e71d2576ea80601a1e8529`.

Out of scope: training, class-conditional PixelDiT, FlowEdit, CUDA-only acceleration, and
reduced-precision parity.

## Reference

The official `PixDiTTrainer` wrapping `PixDiT_T2I`, followed by `DPMS.sample` with the FLOW
schedule, second-order multistep DPM-Solver++, uniform-flow steps, and flow shift 4. Prompt
conditioning uses the official CHI instruction, right-padded Gemma-2-2B-IT final hidden states,
and the first token plus the final 299 tokens.

## Components

| Part | Status |
| --- | --- |
| Gemma tokenizer | reuse `GemmaTokenizer` |
| Gemma-2 text encoder | reuse `Gemma2Model` |
| dual-level PixelDiT transformer | new |
| FLOW DPM-Solver++ sampling loop | new backend-local implementation |
| VAE | none; generated pixels are mapped directly from `[-1, 1]` to uint8 |

## Pipeline

Defaults are 50 steps, guidance 2.75, 1024x1024, seed 0, the published negative prompt, and
guidance over the full open interval. Noise is sampled as `(1, 3, height, width)`. Height and
width must be divisible by the 16-pixel patch size. The denoiser operates directly on RGB pixels;
the final tensor is clamped after `(sample + 1) / 2` and converted to uint8.

## Memory

The transformer and text encoder are about 1.3B and 2.6B parameters respectively. Component
parity frees the reference before loading the native component. Real end-to-end tests use a
small divisible image size and few sampling steps to keep CPU execution bounded.

## Validation

Published-layout checks cover the PixelDiT pickle state dictionary and both Gemma safetensor
shards. Exact random-weight tests compare the new blocks, full tiny transformer, flow solver,
and tiny end-to-end generation with the pinned official implementation. Real-checkpoint tests
separately cover tokenizer, text encoder, a denoiser step, seed-0 end to end, and a seed-42 public
pipeline run. The real end-to-end cases use two steps and 16x16 pixels.

