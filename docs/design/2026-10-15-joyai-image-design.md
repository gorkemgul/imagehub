# JoyAI-Image design

- Scope: native support for JoyAI-Image Edit and Edit Plus instruction-editing checkpoints.
- Weights: `jdopensource/JoyAI-Image-Edit-Diffusers` at `4b41fb25d961f37668750178ccbb380da326201c` and
  `jdopensource/JoyAI-Image-Edit-Plus-Diffusers` at `c2686460c7b64d8aa11bc4d0da423fb316b33f9e`.
  Both publish bfloat16 safetensors only.
- Reference: diffusers 0.40.0 `JoyImageEditPipeline` and `JoyImageEditPlusPipeline`.
- Components: native Qwen3-VL vision patch encoder, merger, and language model conditioning;
  flow-match Euler scheduler; a native single-image WAN causal VAE; and the new dual-stream JoyAI
  3D transformer. The existing Qwen tokenizer is reused.
- Pipeline: the base model allows text-only generation or one reference image; Plus requires an
  image. Input images are RGB-converted, bucketed, resized bilinearly, center-cropped, then VAE encoded.
- Memory: the published 40-layer, 4096-wide transformer and Qwen3-VL encoder require bfloat16
  inference; real-checkpoint CPU float32 validation may need staged reference/native loading.
- Validation: strict published-header tests and exact random-weight tests cover the base and Plus
  transformers, Qwen3-VL vision encoder and merger, and inline image-token conditioning. Snapshot-gated
  tests cover the tokenizer and processor, conditioning, transformer, VAE, and two fixed-image
  end-to-end seeds per checkpoint; the real-checkpoint measurements remain pending snapshots.
- Out of scope: safety checkers, callbacks, batched prompts, and reference pipeline multi-image
  features beyond the required single `image` input.
