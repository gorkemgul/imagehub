# Qwen-Image design

## Scope

Add the original `qwen-image` checkpoint from `Qwen/Qwen-Image` at revision
`75e0b4be04f60ec59a75f475837eced720f823b6`. The repository is author-published,
Apache-2.0, and provides bfloat16 safetensors only: four text-encoder shards, nine
transformer shards, and one VAE file. CPU float32 loading upcasts these published
weights for parity with diffusers 0.40.0.

## Components

The release is not layout-compatible with Qwen-Image 2.1. It uses the Qwen2.5-VL
text model (28 layers, 3584 hidden size), a 60-layer dual-stream
`QwenImageTransformer2DModel` with 64-channel patch-2 input and 16-channel output,
and the original 16-channel causal Qwen VAE. The tokenizer lives in `tokenizer/`,
not the 2.1 `processor/` directory. Dynamic-shift FlowMatch Euler sampling is shared.

## Pipeline and validation

The native pipeline follows `QwenImagePipeline`: prompt templating, last text-layer
embeddings, patch packing, dynamic-shift Euler sampling, latent denormalization and
uint8 conversion. Validation covers safetensor headers, tokenizer and Qwen2.5-VL
outputs, transformer and VAE random/reference parity, tiny pipeline parity for
guidance, non-square inputs and two seeds, then separate real-checkpoint seed-zero
and seed-42 public-pipeline runs. The 20B transformer reference image is persisted
before native weights are loaded.

## Out of scope

Image editing, safety checking, vision inputs, LoRA, and reduced-precision parity.
