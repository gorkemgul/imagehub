# Qwen-Image-Edit design

## Scope

Add the Apache-2.0 `Qwen/Qwen-Image-Edit` and `Qwen/Qwen-Image-Edit-2509` checkpoints to the
existing `qwen-image` family. Both are pinned to immutable Hub revisions. The official code
repository is pinned to `6b5e1f5cec987d404be5ac6657db3b9aacb56a89`; numerical reference is
diffusers 0.40.0 with transformers 5.18.0.

## Components

The existing Qwen-Image transformer, causal VAE, Qwen2 tokenizer and dynamic-shift flow-match
Euler scheduler are reused. The only new model component is the full Qwen2.5-VL conditioner: its
vision patch embedder, window/full attention blocks, patch merger and image-token injection extend
the existing native Qwen2.5-VL text module while retaining the published tensor names.

## Pipeline

The public `image=` option accepts one image for the original edit checkpoint and a list for the
2509 plus pipeline. Images are independently prepared for CLIP-normalized visual conditioning and
for deterministic posterior-mode VAE conditioning. Packed source latents follow the target noise
tokens through every denoiser call; only target predictions are stepped. True CFG uses the same
image with the negative instruction and norm rescales the combined prediction.

## Validation and limits

Random weights cover the vision tower and full multimodal conditioner exactly. Tiny pipeline tests
compare final uint8 images for guidance on/off, non-square output and two seeds. Real-checkpoint
tests use subprocess isolation for the reference pipeline and two 32×32 steps. Transformer LoRA
adapters load as on the other Qwen-Image releases. Training, video and reduced-precision parity are
out of scope.
