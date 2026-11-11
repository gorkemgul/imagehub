# Nucleus-Image design

- Date: 2026-10-13
- Scope: text-to-image support for `nucleus-image`, pinned to NucleusAI/Nucleus-Image revision `5e963db4fd0a65c7e4faf53ca2d4eca567c4dcfa`.
- Reference: diffusers 0.40.0 `NucleusMoEImagePipeline` and `NucleusMoEImageTransformer2DModel`.
- Components: the Qwen3-VL text model, Qwen tokenizer, Qwen-Image VAE, and flow-match Euler scheduler are reused. The Nucleus mixture-of-experts transformer is native and new.
- Pipeline: Qwen chat formatting, layer -8 text states, packed 2x2 latents, flow-match Euler denoising, normalized CFG, and Qwen-Image latent normalization.
- Weights: all published checkpoint shards are bfloat16; CPU inference upcasts them when `dtype=torch.float32` is selected.
- Expert products: the published config sets `use_grouped_mm`, so routed experts use `torch.nn.functional.grouped_mm` on CPU and CUDA, as diffusers does; one product per expert differs in the last bit with several CPU threads. MPS uses one product per expert.
- Validation: tensor headers, random-weight transformer parity (including the published widths at 4,096 image tokens), tiny pipeline parity, and real-checkpoint text encoder, VAE and 1024×1024 end-to-end tests for two seeds, all exact.
- Out of scope: batching, callbacks, prompt embeddings, custom sigmas, and reduced-precision parity.
