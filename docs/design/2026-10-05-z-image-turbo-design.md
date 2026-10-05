# Z-Image-Turbo design

- Date: 2026-10-05
- Roadmap step: 4 in `.ai/GOAL.md`
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`z-image-turbo` (`Tongyi-MAI/Z-Image-Turbo`, Apache-2.0), text-to-image only, at `full` validation
status. A new family `z-image`.

Out of scope: the Omni / edit / controlnet / inpaint / img2img variants (SigLIP branch, noise masks,
`patchify_and_embed_omni`), LoRA, the base (non-turbo) Z-Image checkpoint, fp16/bf16 parity.

## Reference

diffusers 0.40.0 `ZImagePipeline`, `ZImageTransformer2DModel`, `AutoencoderKL`,
`FlowMatchEulerDiscreteScheduler`; transformers 5.18.0 `Qwen3Model` and the `Qwen2Tokenizer`
loaded from the snapshot. Every numeric detail follows the installed source of these versions.

## Published checkpoint (revision `f332072aa78be7aecdf3ee76d5c247082da564a6`)

| Component | Class | Notes |
| --- | --- | --- |
| tokenizer | Qwen2Tokenizer | byte-level BPE, `tokenizer.json` + vocab/merges, chat template |
| text_encoder | Qwen3 (36 layers, hidden 2560, GQA 32/8 heads, head_dim 128, q/k RMSNorm, RoPE θ=1e6) | bf16 weights, 3 shards (8 GB) |
| transformer | ZImageTransformer2DModel (dim 3840, 30 layers, 2 noise + 2 context refiner layers, 30 heads, 3-axis RoPE 32/48/48, θ=256, patch 2) | fp32 weights, 3 shards (24.6 GB) |
| vae | AutoencoderKL, 16 latent channels, no quant convs, scaling 0.3611, shift 0.1159 (FLUX VAE) | |
| scheduler | FlowMatchEulerDiscreteScheduler, shift 3.0, no dynamic shifting | |

## Design

### Tokenizer (`tokenization_qwen.py`)

Native byte-level BPE matching transformers' Qwen2 tokenizer as loaded from this snapshot:
NFC normalization, the Qwen2 pre-tokenizer pattern (implemented without the `regex` package, as
the CLIP tokenizer already does for `\p{L}`/`\p{N}`), added/special tokens extracted before BPE,
merges by rank. The pipeline's chat template is rendered natively for the single case the pipeline
uses (one user message, `add_generation_prompt=True`, `enable_thinking=True`); its exact output
string is pinned by a parity test against `apply_chat_template`. Padding to `max_sequence_length`
(512) with `<|endoftext|>`, attention mask, truncation. Parity: fixed prompts plus a fuzz test
(Unicode, digits, whitespace runs, newlines, special-token literals), 0 mismatches.

### Text encoder (`models/qwen3.py`)

Qwen3 decoder stack without the LM head: embeddings, RMSNorm, GQA attention with per-head q/k
RMSNorm, rotary embeddings (default rope), SwiGLU MLP, causal plus padding attention mask. The
pipeline uses `hidden_states[-2]` (output of layer 35 before the final norm), then keeps only the
unmasked positions per prompt (variable length list). Weights are bf16 on disk; loading casts to
the requested dtype like transformers does with `torch_dtype=float32`.

### Transformer (`models/z_image_transformer.py`)

Single-stream DiT as in `ZImageTransformer2DModel` (text-to-image path only):
timestep embedder (`t * t_scale`, sinusoidal 256 → MLP, mid 1024, `min(dim, 256)` out), caption
embedder (RMSNorm + Linear 2560→3840), patchify (2×2, frame patch 1) with padding of each stream to
a multiple of 32 using learned `x_pad_token` / `cap_pad_token`, 3-axis position ids and complex
RoPE, two noise refiner blocks (with adaLN) and two context refiner blocks (without adaLN), 30 main
blocks over the concatenated unified sequence with adaLN modulation (scale/gate, tanh gates,
sandwich RMSNorms), final layer, unpatchify. Inputs and outputs are lists of per-sample tensors as
in the reference; batching pads to the longest sequence with an attention mask.

### Scheduler (`schedulers/flow_match_euler.py`)

`FlowMatchEulerDiscreteScheduler` for this config: custom sigmas
`linspace(1, 1/steps, steps)` from the pipeline, static shift 3.0
(`shift*s / (1 + (shift-1)*s)`), terminal 0 appended, timesteps `sigmas * 1000`, Euler step
`x + (sigma_next - sigma) * v`. The pipeline's `mu` is computed but unused with static shifting;
reproduce what the installed code does.

### VAE

Extend `AutoencoderKL` for `latent_channels` 16, `use_quant_conv` / `use_post_quant_conv` false and
`shift_factor`: decode input is `latents / scaling_factor + shift_factor`.

### Backend (`backends/z_image.py`)

Components: tokenizer, text_encoder, transformer, vae, scheduler. Flow as in the reference:
latents `randn(B, 16, H/8, W/8)` float32 from the seeded CPU generator; per step
`timestep = (1000 - t) / 1000`; the transformer's output is negated; CFG (when guidance > 0) is
`pos + g * (pos - neg)` with optional `cfg_normalization` and `cfg_truncation` (pipeline defaults:
off / 1.0). Defaults from the model card: 9 steps, guidance 0.0, 1024×1024 (the reference pipeline
defaults are 50 steps and guidance 5.0; the card's settings are what the checkpoint is for, as with
SDXL Turbo). Height/width must be multiples of 16.

## Memory

fp32 transformer 24.6 GB and text encoder 16 GB; holding our model and the reference together for
both components (≈ 82 GB) is too close to the 96 GB machine. The real-checkpoint tests therefore
load one component pair at a time and free it, and the end-to-end test runs the reference pipeline
first, stores its latents and image, frees it, then runs ours from the same initial noise.

## Validation

Header check (sharded safetensors index), random-weight structural parity for the text encoder,
transformer and VAE (tiny configs and the full configs on the meta device for layout), tokenizer
and chat-template parity plus fuzz, scheduler parity, a tiny-snapshot end-to-end comparison with
`ZImagePipeline`, then real-checkpoint component parity and end-to-end parity at 1024×1024 on CPU
float32 (9 steps, guidance 0.0) plus a guided run (guidance > 0, negative prompt) and the seed test.

## Task breakdown

1. Qwen2 tokenizer + chat template + fuzz parity.
2. Qwen3 text encoder + random-weight parity (and `hidden_states[-2]` selection).
3. FlowMatch Euler scheduler + VAE extensions (16 channels, shift factor, no quant convs) + parity.
4. Z-Image transformer + random-weight parity (tiny config exact; full-config layout on meta).
5. Backend, catalog entry, family registration, tiny-snapshot unit and reference tests.
6. Real-checkpoint download, parity run, validation records and docs.
