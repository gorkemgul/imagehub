# Kandinsky 2.1 implementation plan

1. Pin both published repositories; store model, scheduler, tokenizer and processor configs plus
   safetensors header shapes for every weight-bearing component.
2. Implement the XLM-RoBERTa tokenizer, encoder and MultilingualCLIP projection with exact
   transformers parity.
3. Extend the shared U-Net with `text_image` addition embeddings and `text_image_proj` encoder
   conditioning; retain exact behavior for existing configurations.
4. Add the combined prior/decoder backend, catalog entry and registry mapping, validating every
   option before model loading.
5. Add meta-device header checks, random-weight component parity, DDIM parity, tiny combined-pipeline
   parity, and a snapshot-gated real-checkpoint suite.
6. Update the README, catalog and model/validation documentation, then run formatting, static checks,
   the full fast suite in both environments, and all Kandinsky 2.1 reference tests.
