# HART design

- Date: 2026-10-18
- Issue: #139
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

One author weight set, `hart`, for text-to-image generation at 1024×1024. The HART weights are
`mit-han-lab/hart-0.7b-1024px` revision `b1521ff1449d6817c685b08614c554e7a5b5b8eb`
(MIT), and the text encoder is `mit-han-lab/Qwen2-VL-1.5B-Instruct` revision
`6e5b5399a390528a3b2c9f5109fb795553d3db71` (Apache-2.0). Both repositories are maintained by
the authors. The official implementation is `mit-han-lab/hart` revision
`e28a41fe15025baf3d48ab004f5d3727c88d8fc7`.

Out of scope: training, the safety checker, CUDA-only fused kernels, and non-EMA transformer
weights. The catalog uses the EMA transformer selected by the official inference script.

## Reference

The official `HARTForT2I.autoregressive_infer_cfg` method at the pinned code revision, including
its multi-scale autoregressive loop, final-stage MaskGIT pass, eight-step improved-DDPM residual
sampler, and `HARTAutoEncoder.fhat_to_img` decoder. Prompt embeddings come from the text-only path
of Qwen2-VL-1.5B-Instruct using the official system prompt.

## Components

| Part | Status |
| --- | --- |
| Qwen2 byte-level BPE tokenizer | reuse `Qwen2Tokenizer` |
| Qwen2-VL 1.5B text-only decoder | new native Qwen2 text module if its published layout differs from existing Qwen modules |
| HART multi-scale transformer with 1D/2D rotary attention | new |
| HART residual diffusion MLP and improved-DDPM sampler | new |
| HART hybrid quantizer and autoencoder decoder | new |

## Pipeline

For one prompt, apply the official system prompt, tokenize to 300 tokens, take the Qwen2 final
hidden states and attention mask, and run classifier-free HART generation. Defaults are guidance
4.5, seed 1, the published fourteen-stage patch schedule ending at 64×64, one final MaskGIT pass,
and eight residual-diffusion steps. Decode the 32-channel latent to a 1024×1024 RGB image and use
the official multiply-by-255 then uint8 conversion.

## Memory

The HART generator is about 0.7B parameters, the Qwen2 encoder about 1.5B, and the tokenizer
autoencoder about 0.1B. CPU float32 component comparisons free the reference before loading the
native component; end-to-end validation uses one implementation at a time.

## Validation

Published-layout checks cover both HART pickle state dictionaries and Qwen2 safetensors. Random
weight tests compare each new module and a tiny whole generation loop with the pinned official
implementation exactly. Real-checkpoint tests separately cover tokenizer, text encoder,
transformer step, decoder, seed-0 end to end, and seed-42 end to end through `pipeline(...)`.
The real end-to-end tests use a shortened patch schedule and diffusion run to keep CPU execution
bounded while retaining every pipeline phase.

