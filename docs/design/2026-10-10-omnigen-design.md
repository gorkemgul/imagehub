# OmniGen design

- Date: 2026-10-10
- Issue: #91
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

One weight set of a new family `omnigen`, text-to-image with an optional input image, catalog id `omnigen`, by the
authors, not gated, MIT (model card and the awesome catalog):

| Model id | Repository | Revision | Transformer |
| --- | --- | --- | --- |
| `omnigen-v1` | `Shitao/OmniGen-v1-diffusers` | `016e2f61d12a98303f6bbdf122687694d7984268` | 3.8B, one bfloat16 file (7.75 GB) |

The repository publishes the transformer only in bfloat16 (no float32 copy); the autoencoder is float32. Both
implementations convert the transformer to float32 on load, which is the parity target. The model id is `omnigen-v1`
(not `omnigen`) so that the family selector `omnigen` stays distinct from the model alias.

Out of scope: several input images, `num_images_per_prompt` above 1, batches of prompts, a custom negative prompt,
caller-supplied latents, callbacks, reduced-precision parity, the original (non-diffusers) OmniGen repository, OmniGen2.

## Reference

diffusers 0.40.0 `OmniGenPipeline` (`pipelines/omnigen/pipeline_omnigen.py`), its `OmniGenMultiModalProcessor` and
`OmniGenCollator` (`processor_omnigen.py`), `OmniGenTransformer2DModel`, `AutoencoderKL`, `FlowMatchEulerDiscreteScheduler`;
transformers 5.18.0 `LlamaTokenizer` (loaded from `tokenizer.json`).

## Components

| Part | Status |
| --- | --- |
| `OmniGenTransformer2DModel` (Phi-3 style blocks with fused gate/up MLP, "su" rotary embedding, two patch projections, stored position table, time token, AdaLN output) | **new** |
| Llama-style BPE tokenizer (byte fallback, prefix space on the first piece only, `rstrip` added tokens) | **new**, a subclass of the Gemma BPE tokenizer |
| Multimodal prompt processor: instruction wrapper, placeholder splitting, image preprocessing, padded ids, positions, block attention masks | **new** |
| `AutoencoderKL` (4 channels, quant convs) | exists; gains `encode_moments` (mean and log-variance) because the pipeline samples the posterior |
| Flow-match Euler with `invert_sigmas` | scheduler exists; the inverted schedule is added behind an explicit opt-in (`allow_inverted_sigmas`), the default constructor still rejects it |

## Pipeline details that matter for parity

- Prompt: `<|user|>\nGenerate an image according to the following instructions\n{prompt}<|end|>\n<|assistant|>\n<|diffusion|>`.
  The batch is `[prompt, negative prompt]`, plus an image-only prompt (just the placeholders) when an image is given.
  The fixed negative prompt is the processor's default. Sequences are left padded with token id 2.
- Tokenization: `<|assistant|>` and `<|placeholder*|>`/`<|image|>`/`</s>` carry `rstrip`, so the newline after
  `<|assistant|>` disappears; only the piece that starts the text gets a prefix `▁`; every placeholder-split chunk is
  tokenized separately and loses its leading `<s>` except the first.
- Placeholder convention: the prompt must contain the exact text `<img><|image_1|></img>` (the pipeline's own check); the
  processor then replaces `<|image_1|>` by one zero id per 16×16 input pixels and the transformer overwrites those
  embeddings with the patch embeddings of the VAE latents. Only image number 1 is accepted; the placeholder may occur
  several times (the same image is then encoded once per occurrence).
- Image preprocessing: box-halve while both sides are at least twice `max_input_image_size`, bicubic to the maximum, bicubic
  up to 16 on the short side, centre crop to multiples of 16, `ToTensor`, normalize to [-1, 1]. Non-RGB images are converted
  to RGB first (the reference fails on them).
- Input-image latents are `posterior.sample() * scaling_factor`. The reference samples from the global torch generator
  before the initial noise is drawn; a dedicated CPU generator with the request seed gives the same stream as
  `torch.manual_seed(seed)`, which is how the reference runs are seeded in the tests.
- Position ids and masks come from the collator: causal over text plus the time token, bidirectional inside each input
  image span, noisy-image tokens attend to everything. The reference rotary module builds cos and sin from the first
  batch item only and switches to the long factors when the largest position exceeds 4096; both are reproduced.
- Sampling: `sigmas = np.linspace(1, 0, steps + 1)[:steps]` (float64, then float32), scheduler `shift` 1, `invert_sigmas`,
  `num_train_timesteps` 1, so the transformer sees times `0, 1/n, ...`. Guidance is always applied:
  `uncond + g (cond - uncond)` without an image, `uncond + image_g (img_cond - uncond) + g (cond - img_cond)` with one.
  Defaults: 50 steps, guidance 2.5, image guidance 1.6, 1024×1024, `max_input_image_size` 1024.
- Decode: `latents / scaling_factor`, `vae.decode`, `(x / 2 + 0.5).clamp(0, 1)`, uint8 as diffusers.
- Image input is wired through `ModelSpec.image_input="optional"` and the `image` option of `t2i`, as for FLUX.1 Kontext
  (`2026-10-08-image-input-and-flux-kontext-design.md`): the backend accepts `image`, rejects it with the other
  unknown options elsewhere, and records `image_input`, `input_height` and `input_width` in the result metadata.

## Memory

fp32: transformer about 15.5 GB (stored as 7.75 GB of bfloat16), autoencoder 0.3 GB. The reference and the native model fit
together on a 96 GB host; the end-to-end tests still compute and free the reference result before ours is loaded.

## Validation

Tokenizer parity with a fuzz test; random-weight transformer parity (tiny, long-rope, published config with two layers,
with and without image tokens); processor parity (ids, masks, positions, pixels); scheduler parity at several step
counts; tiny end to end against `OmniGenPipeline` (defaults, guidance, non-square, seeds, image input variants); then
real-checkpoint header, tokenizer, transformer step (text only and with an image), VAE, end-to-end (text-to-image and
image input, shortened steps because a full 1024×1024 run takes hours on CPU) and seed tests.

## Task breakdown

1. Fixtures, transformer, tokenizer, processor, scheduler option and `encode_moments`, with parity.
2. Backend, catalog entry, registry, unit tests, tiny end to end.
3. Real-checkpoint test file and docs; the pull request closes #91.
