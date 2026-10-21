# Show-o design

- Date: 2026-10-20
- Issue: #169
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Add `show-o-512`, the author-published `showlab/show-o-512x512` checkpoint at revision
`dbe757d951099968fb102d249d35a764528df2dd`, for text-to-image generation. The Apache-2.0 model
uses the `showlab/magvitv2` image tokenizer at revision
`5c3fa78f8b3523347c5cd1a4c97f3c4e96f33d5d` and the `microsoft/phi-1_5` tokenizer at revision
`77aa61eeac94fbf33d492b9f2744c98b42d5b5eb`.

Out of scope: multimodal understanding, inpainting, extrapolation, CLIP-ViT variants and 256-pixel
checkpoints.

## Reference

Official Show-o repository revision `45a5a2de01d1ebd10cd5864d29310a76476cdf23`, specifically
`Showo`, `MAGVITv2`, `UniversalPrompting`, `create_attention_mask_predict_next`, and
`Showo.t2i_generate`.

## Components and pipeline

- A native Phi-1.5 decoder with the published `showo.*` tensor names and Show-o's four-dimensional
  omni-attention mask.
- A native MAGVIT-v2 lookup-free quantizer and encoder/decoder with the published tensor names.
- A CodeGen/GPT-2 byte-level BPE tokenizer plus Show-o's ten appended prompt tokens.
- The official 129-token left-padded text prefix, image boundary tokens, 1,024 masked image tokens,
  cosine remasking, multinomial token draws, classifier-free guidance, and MAGVIT-v2 decoding.
- Defaults: 512 by 512, 18 sampling steps, guidance 10, temperature 1.

The full model is about 5.8 GB in float32 and the VQ model about 0.38 GB. Real-checkpoint reference
runs compute and release the official result before loading the native model.

## Validation

Published header checks cover all model and VQ tensors. Random-weight tests compare Phi attention,
the MAGVIT-v2 decoder, prompt construction, sampling, and a tiny end-to-end run exactly against the
pinned official implementation. Opt-in real tests cover tokenizer output, logits, VQ decoding, a
short two-step 512-pixel end-to-end run at seeds 0 and 42 through the public pipeline API.
