# Janus-Pro design

- Date: 2026-10-05
- Roadmap step: 5 in `.ai/GOAL.md`
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`janus-pro-1b`, then `janus-pro-7b` (code MIT, weights under the DeepSeek Model License), text-to-image only, at `full` validation status. A new
family `janus`: the first autoregressive template (VQ image tokenizer, language-model backbone,
classifier-free guidance over logits). LlamaGen reuses the VQ decoder and the sampling loop later.

Out of scope: image understanding (the SigLIP vision encoder and the aligner are not built, and
their tensors are skipped when loading), text generation, the original `janus` package format,
fp16/bf16 parity.

## Reference

transformers 5.18.0 `JanusForConditionalGeneration.generate(generation_mode="image")`,
`decode_image_tokens`, `JanusProcessor` (chat template, `generation_mode="image"` appends the
begin-of-image token) and `JanusImageProcessor.postprocess`. Every numeric detail follows the
installed source.

## Published checkpoint

The transformers-format conversion `deepseek-community/Janus-Pro-1B`
(revision `1655280bb75959cc1cb85529a2a8b26e7016072e`): one `model.safetensors` (bf16, 4.2 GB),
`config.json`, `generation_config.json` (guidance 5, boi token 100003), tokenizer files and
`chat_template.jinja`. The authors' repository `deepseek-ai/Janus-Pro-1B` publishes the same
weights as `pytorch_model.bin` for the original `janus` package, which transformers cannot load.
The catalog therefore uses the conversion with a new provenance value `community-conversion`, and
the validation adds a conversion check: every converted tensor equals the corresponding tensor of
the authors' `pytorch_model.bin` (names mapped, values compared exactly).

| Part | Config |
| --- | --- |
| language model | Llama, 24 layers, hidden 2048, 16 heads (no GQA), head_dim 128, SwiGLU 5632, RMSNorm 1e-6, RoPE θ=1e4, vocab 102400 |
| generation embeddings / aligner / head | `Embedding(16384, 8)` → MLP to 2048; head MLP 2048 → 16384 logits |
| VQ decoder | LlamaGen VQ-16: embed_dim 8, 16384 codes, 256 latent channels, channel multipliers 1,1,2,2,4, 24×24 tokens → 384×384 |

## Design

### Tokenizer (`tokenization_llama.py` or a shared byte-level BPE module)

The DeepSeek LLM tokenizer (`tokenizer.json`, byte-level BPE with its own pre-tokenizer pattern
and added tokens). Reuse the byte-level BPE core written for Qwen where the rules are the same;
the pre-tokenizer and normalizer follow `tokenizer.json`. The prompt is the processor's chat
template for one user message plus `<begin_of_image>`; it is rendered natively and pinned by a
parity test. Fuzz parity as for the other tokenizers.

### Language model (`models/llama.py`)

Llama decoder stack without the LM head, with a KV cache. The reference runs image generation
with a static cache whose length is `max(max_length, 576 + prompt length)` and an attention mask
over it; attention sums over that padded length may change the BLAS summation order, so the
native cache reproduces the reference's cache length and masking unless a test shows the result
is identical without it.

### Generation head and loop (`backends/janus.py`)

Exactly as `generate(generation_mode="image")`: batch doubled for CFG (unconditional rows keep
only BOS and begin-of-image, everything else replaced by the pad token), 576 steps, logits from
`generation_head`, `ClassifierFreeGuidanceLogitsProcessor` (guidance > 1), then the default
logits processors for the generation config (temperature / top-k / top-p only if set), sampling
with `softmax` + `multinomial` (or argmax when sampling is off), next-token embeddings through
`generation_embeddings` + `generation_aligner`. The reference samples from the global torch RNG;
we sample from a dedicated `torch.Generator` seeded with `seed` and test that it draws the same
stream as `torch.manual_seed(seed)` on the reference side.

### VQ decoder (`models/vq_decoder.py`)

`JanusVQVAE` decode path only: codebook lookup (`quantize.get_codebook_entry`), `post_quant_conv`,
decoder (conv in, mid block with attention, up blocks with nearest upsampling, norm, conv out).
Encoder tensors are skipped when loading. Postprocessing as `JanusImageProcessor.postprocess`
(unnormalize with mean/std 0.5, clip, to uint8).

### Backend options

`steps` is not applicable; options: `guidance` (5.0, from `generation_config.json`),
`temperature` (1.0), `num_images` (1), `seed`, `do_sample` (True). Output size is fixed at
384×384; any other `width`/`height` raises `ValueError`.

## Memory and runtime

1B parameters in fp32 ≈ 8 GB per copy; ours and the reference together fit easily. 576
autoregressive steps with a KV cache take minutes on CPU. 7B (fp32 ≈ 28 GB per copy) still fits
one pair at a time.

## Validation

Header check (with the skipped understanding-only tensors listed explicitly), the conversion check
against `deepseek-ai/Janus-Pro-1B`, random-weight structural parity (Llama stack with cache, VQ
decoder, generation head/aligner), tokenizer and chat-template parity plus fuzz, sampling-RNG
equivalence, a tiny-checkpoint end-to-end comparison against
`JanusForConditionalGeneration.generate`, then real-checkpoint component parity (language-model
logits at the first steps, VQ decode) and end-to-end parity of the 576 image tokens and the decoded
pixels with sampling on (seeded) and guidance 5, plus the seed test through `pipeline()`.

## Task breakdown

1. DeepSeek tokenizer + chat prompt + fuzz parity.
2. Llama language model with the reference's cache behaviour + random-weight parity.
3. VQ decoder + generation embeddings/aligner/head + random-weight parity.
4. Backend, sampling loop, catalog entry (`community-conversion` provenance), family registration,
   tiny-checkpoint unit and reference tests.
5. Real-checkpoint download (1B and the authors' `.bin` for the conversion check), parity run,
   validation records and docs. Then 7B as a follow-up.
