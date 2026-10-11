# GLM-Image design

- Date: 2026-10-10
- Issue: #87
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

One weight set of a new family `glm-image`, text-to-image, catalog id `glm-image`, by the authors, not gated,
MIT (model card):

| Model id | Repository | Revision | Size |
| --- | --- | --- | --- |
| `glm-image` | `zai-org/GLM-Image` | `2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4` | autoregressive 20 GB, transformer 13.9 GB, ByT5 0.9 GB, VAE 0.8 GB |

The transformer, the autoregressive model and the VAE are stored in bfloat16 (there is no float32 copy); the ByT5
encoder is float32. Both the reference and the native code widen them to float32. About 16B parameters in all.

Out of scope: the optional condition images (the sibling catalog lists the model as `editing`). They need the
40-block vision tower, the VQ encoder, the VAE encoder and the transformer's per-layer key/value cache, which is
substantially more than text-to-image; the vision and VQ tensors in the autoregressive files are ignored when
loading, and `ModelSpec.image_input` stays unset. Also out of scope: batches, precomputed prompt embeddings or
prior tokens, custom timesteps, reduced-precision parity.

## Reference

diffusers 0.40.0 `GlmImagePipeline`, `GlmImageTransformer2DModel`, `FlowMatchEulerDiscreteScheduler`,
`AutoencoderKL`; transformers 5.18.0 `GlmImageForConditionalGeneration` (text tower, `generate`),
`GlmImageProcessor`, `T5EncoderModel`, `ByT5Tokenizer`.

## Components

| Part | Status |
| --- | --- |
| Autoregressive text tower: GLM-4 layers with four norms per layer, three-axis rotary positions (sections 8/12/12, half-rotary), GQA, vision LM head, KV-cache decoding with temperature / top-k / top-p sampling | **new** (`glm_image_ar.py`; norm and MLP from `glm.py`) |
| Prompt tokenizer (GLM-4 BPE with the GLM-Image added tokens, target-grid suffix) | **new** thin subclass of `GlmTokenizer` |
| ByT5 tokenizer (bytes + 3, added tokens with strip flags) | **new** |
| ByT5 encoder (T5 encoder, gated GELU, 12 layers, width 1472) | existing `T5EncoderModel` |
| Diffusion transformer | **new** (`glm_image_transformer.py`; blocks, modulation and final norm from `cogview4_transformer.py`) |
| VAE (16 latent channels, no attention in the mid block, no quant convs) | existing `AutoencoderKL` |
| Flow-match Euler, linear dynamic shifting, custom sigmas | existing; the pipeline's sigma list is a new helper |

## Pipeline details that matter for parity

- The prompt is followed by `<sop>H W<eop><sop>h w<eop>` and the image-start token, where H, W are the pixel sizes
  divided by 32 and h, w a 16-token preview grid with the same aspect ratio. The model generates h*w preview
  tokens, then H*W tokens, then stops; the preview is discarded and the full grid is upsampled 2x (nearest).
- Rotary positions: the prompt uses plain positions on all three axes; generated tokens use the grid positions
  (time fixed, height and width per row and column), preview first.
- Sampling uses the vision-head logits as float32, then temperature, top-k and top-p from `generation_config.json`,
  softmax and `torch.multinomial` on the global CPU stream seeded with the generator's seed. The native code
  uses a CPU generator with the same seed, which yields the same stream.
- Glyph texts are the quoted passages of the prompt; when the number of passages is even every row gets a
  leading pad token (the reference computes the parity from the row count). The embeddings of all passages are
  concatenated into one sequence. Guidance is on above 1; the unconditional pass uses the empty-string glyph
  embedding and zeroed prior-token embeddings.
- Timesteps are `int(linspace(1000, 1, steps + 1)[:-1])` in float32 over 1000, shifted linearly with
  `mu = sqrt(seq_len / 256) * 0.75 + 0.25`; the transformer receives `timestep - 1`.
- Latents are unnormalized with the VAE `latents_mean` / `latents_std` before decoding.

## Memory

fp32: autoregressive text tower about 36 GB, transformer 28 GB, ByT5 0.9 GB, VAE 1.6 GB. The backend loads the
stages one after the other and frees each, so the peak is about 36 GB plus loading overhead. The real-checkpoint
tests save each reference stage's result before loading the native one.

## Validation

Random-weight parity for the transformer and the sampling loop; sigma, ByT5 and tokenizer parity; a tiny end to
end against the complete diffusers pipeline; real-checkpoint component and end-to-end tests with a shortened
diffusion stage (4 steps at 1024x1024; the autoregressive stage runs in full) and a second seed at 512x512.
