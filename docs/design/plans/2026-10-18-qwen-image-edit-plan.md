# Qwen-Image-Edit implementation plan

1. Pin both official Hub repositories and the Qwen-Image code repository; save published configs,
   weight indexes and safetensor headers.
2. Extend the native Qwen2.5-VL module with the vision tower and multimodal image-token path, then
   prove exact random-weight parity against transformers 5.18.0.
3. Reuse the Qwen-Image transformer, VAE, tokenizer and scheduler in an edit component loader and
   reproduce `QwenImageEditPipeline` / `QwenImageEditPlusPipeline` preprocessing and denoising.
4. Add tiny snapshot unit tests, exact tiny end-to-end reference tests, and an opt-in real snapshot
   suite using subprocess isolation for heavyweight end-to-end comparisons.
5. Add both pinned checkpoints to the catalog and document usage, limitations and validation.
