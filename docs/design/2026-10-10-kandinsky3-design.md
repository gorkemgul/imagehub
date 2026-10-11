# Kandinsky 3 design

- Date: 2026-10-10
- Issue: #90
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

One weight set of a new family `kandinsky3`, text-to-image, at `full` validation status, catalog id
`kandinsky-3`, by the authors, not gated, Apache-2.0 (model card and the awesome catalog):

| Model id | Repository | Revision | Parts |
| --- | --- | --- | --- |
| `kandinsky-3` | `kandinsky-community/kandinsky-3` | `bf79e6c219da8a94abb50235fdc4567eb8fb4632` | T5 encoder 8.7B, U-Net 3.06B, MoVQ 0.27B |

The repository publishes float16 weights only (`*.fp16.safetensors`, a float16 text encoder index
with five shards); no float32 files exist. The catalog lists those files, as for Kolors; the loaders
upcast to float32 and the reference is loaded with `variant="fp16"` and `torch_dtype=torch.float32`.
The `assets/` images are not listed.

Out of scope: Kandinsky 3.1, Flash, inpainting and image-prompted variants, LoRA, other schedulers,
precomputed prompt embeddings, several prompts per call, reduced-precision parity.

## Reference

diffusers 0.40.0 `Kandinsky3Pipeline`, `Kandinsky3UNet`, `VQModel`, `DDPMScheduler`; transformers
5.18.0 `T5EncoderModel` and `T5Tokenizer`.

## Components

| Part | Status |
| --- | --- |
| T5 tokenizer from `spiece.model` (legacy, 100 extra ids) | exists |
| T5 encoder: Flan-UL2 widths (head size 256 not equal to width / heads) and the gated SiLU feed-forward | exists, extended for `gated-silu` |
| `Kandinsky3UNet` (bottleneck residual blocks, conditional group norm, attention pooling of the text into the time embedding) | **new** |
| MoVQ decoder (`VQModel`, spatial norm, four levels, `force_not_quantize`) | exists, no change |
| DDPM scheduler with `squaredcos_cap_v2` betas | exists, extended |

## Pipeline

As `Kandinsky3Pipeline.__call__` for one prompt: tokenization to 128 tokens (padded, truncated);
encoder hidden states with the padded positions zeroed and the context cut to the longest prompt
plus one position, multiplied by the mask. With guidance (> 1) and no negative prompt the
unconditional context is zeros with an all-false mask; with a negative prompt (an empty string
counts) it is encoded and cut to the conditional length. Mask to the U-Net as `(1 - m) * -10000`.
Latents `randn(B, 4, H', W')` where the pixel size is rounded up to a multiple of 64 and divided by
8; guidance `(g + 1) * text - g * uncond`; DDPM steps (leading spacing, offset 1, clipping,
`fixed_small` variance) draw noise from the same generator as the initial latents; the image is
`movq.decode(latents, force_not_quantize=True)`, scaled to [0, 1] and rounded to uint8. Defaults: 25
steps, guidance 3.0, 1024×1024.

## Memory

fp32: text encoder 34.9 GB, U-Net 12.2 GB, MoVQ 1.1 GB. The backend keeps all three loaded. In the
real-checkpoint tests the reference conditioning is computed, saved and freed before ours is
loaded, and the reference denoiser runs without a text encoder.

## Validation

Header checks of all three components against the published headers; random-weight parity for the
silu T5, the U-Net (tiny and published widths) and MoVQ (published config); DDPM schedule and
trajectory parity for the published scheduler config; tiny end to end against the pipeline;
real-checkpoint tokenizer, conditioning, one denoiser step at three timesteps, MoVQ decode, end to
end with shortened steps, and the seed test.

## Task breakdown

1. Fixtures; `gated-silu` T5 and squared-cosine DDPM, with parity.
2. `Kandinsky3UNet` with header test and random-weight parity.
3. Backend, catalog entry, tiny snapshot unit tests and end to end.
4. Real-checkpoint tests and docs; the pull request closes #90.
