# LlamaGen design

- Date: 2026-10-09
- Issue: #14
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

Two text-to-image weight sets of a new family `llamagen`, catalog id `llamagen`, at `full` validation status:

| Model id | GPT checkpoint | Image size | Tokens |
| --- | --- | --- | --- |
| `llamagen-t2i-xl-256` | `t2i_XL_stage1_256.pt` | 256×256 | 16×16 |
| `llamagen-t2i-xl-512` | `t2i_XL_stage2_512.pt` | 512×512 | 32×32 |

Weights: `peizesun/llamagen_t2i` (revision `276f5c5a3d915b922899a03f1912605531574747`, by an author of the paper, MIT)
with `vq_ds16_t2i.pt`, and the text encoder `google/flan-t5-xl` (revision `7d6315df2c2fb742f0f5b556879d730926ca9001`,
Apache-2.0) in the subfolder `text_encoder/` (the existing `subfolders` mechanism). The GPT and VQ checkpoints are
PyTorch zip pickles; they are loaded with `torch.load(weights_only=True)` and the documented key layout.

Out of scope: the class-conditional (ImageNet) checkpoints, the caption cleaning with `ftfy` and BeautifulSoup that
`T5Embedder` applies by default (imagehub lower-cases and strips the prompt, which is what `T5Embedder` does with
`use_text_preprocessing=False`; plain prompts give the same tokens), `torch.compile`, bf16/fp16 parity.

## Reference

No diffusers or transformers pipeline exists. The reference is the official repository `FoundationVision/LlamaGen` at
commit `ce98ec41803a74a90ce68c40ababa9eaeffeb4ec`, imported from a local checkout (not installed) next to transformers
5.18.0 for the T5 encoder: `autoregressive/models/gpt.py` (`GPT_XL`), `autoregressive/models/generate.py`
(`generate`, prefill and per-token decode with a KV cache, classifier-free guidance, temperature/top-k/top-p
sampling with `torch.multinomial`), `tokenizer/tokenizer_image/vq_model.py` (`VQ_16`), `language/t5.py`
(`T5Embedder`), and the driver logic of `autoregressive/sample/sample_t2i.py` (left padding by flipping the masks,
`caption_embs * masks`, `vq_model.decode_code`). The reference driver in the tests reproduces `sample_t2i.py` without
its CUDA-only settings (`set_float32_matmul_precision`, compilation). Sampling draws from the global CPU generator
after `torch.manual_seed(seed)`; imagehub uses a generator seeded the same way and must consume random numbers in the
same order.

## Components

| Part | Status |
| --- | --- |
| Flan-T5-XL encoder and SentencePiece tokenizer (120 tokens) | encoder exists (`models/t5.py`, gated-GELU); tokenizer exists (`tokenization_t5.py`) — checked against this checkpoint |
| LlamaGen GPT-XL (caption embedder, 2D RoPE with the caption prefix, KV cache) | **new** |
| VQ-16 image tokenizer decoder | compare with `models/janus_vq.py` (Janus-Pro uses a LlamaGen-derived VQ); reuse if identical, else **new** |
| Autoregressive sampler (CFG, top-k/top-p, temperature) | **new** |

Defaults from `sample_t2i.py`: `cfg_scale` 7.5, `temperature` 1.0, `top_k` 1000, `top_p` 1.0, 120 text tokens.

## Memory

fp32: GPT-XL 3.1 GB, Flan-T5-XL encoder about 4.9 GB (encoder only), VQ 0.3 GB.

## Validation

Random-weight parity for the GPT (tiny, and the published XL configuration with small token grids) and the VQ
decoder; sampler parity with fixed logits and seeds; tokenizer and T5 encoder parity; a tiny end to end against the
reference driver; then real-checkpoint component and end-to-end parity for both sizes (identical token ids and
pixels), plus the seed test.

## Task breakdown

1. GPT model and VQ decoder with header checks and random-weight parity.
2. Sampler, T5 path, backend, catalog entries, tiny end to end, real tests, docs; the pull request closes #14.
