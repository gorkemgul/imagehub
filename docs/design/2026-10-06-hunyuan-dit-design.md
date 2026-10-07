# HunyuanDiT design

- Date: 2026-10-06
- Issue: #34
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`hunyuan-dit-v1.1-distilled` (`Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled`, revision
`527cf2e`, Tencent Hunyuan Community License, not gated), text-to-image only, at `full` validation
status, as a new family `hunyuan-dit`. Other HunyuanDiT releases (v1.0, v1.2, non-distilled) can
follow with the same code. Out of scope: ControlNet, the PAG and differential-image pipelines,
LoRA, fp16/bf16 parity.

## Reference

diffusers 0.40.0 `HunyuanDiTPipeline` (`pipelines/hunyuandit/pipeline_hunyuandit.py`),
`HunyuanDiT2DModel`, `AutoencoderKL`, `DDPMScheduler`; transformers 5.18.0 `BertModel`,
`BertTokenizer`, `T5EncoderModel`, `T5Tokenizer` (loaded from `spiece.model`; the reference
environment needs `sentencepiece` and `protobuf`, which join the `reference` extra; the runtime stays
torch-only).

## Components

| Part | Status |
| --- | --- |
| Bilingual BERT tokenizer (WordPiece, `vocab.txt`, lower case, Chinese characters split), 77 tokens | **new** |
| BERT-large encoder (24 layers, 1024 wide, 47020 vocabulary), last hidden state | **new** |
| mT5 tokenizer read natively from `spiece.model` (SentencePiece protobuf: pieces, scores, precompiled normalizer), 256 tokens | **new** reader; Unigram and the precompiled normalizer exist in `tokenization_t5` |
| mT5-XL encoder (24 layers, 2048 wide, gated GELU) | exists (`models/t5.py`) |
| `HunyuanDiT2DModel`: 40 blocks, 1408 wide, 16 heads × 88, long skip connections, 2D RoPE, text-embedding mask, image-meta-size and style embeddings, learned sigma | **new** |
| 4-channel VAE, scaling factor 0.13025 | exists |
| DDPM scheduler (v-prediction, scaled-linear betas, leading spacing, `fixed_small` variance, generator noise each step) | **new** |

## Pipeline

As `HunyuanDiTPipeline.__call__` for text-to-image: resolution binning to the supported shapes, BERT
and mT5 embeddings with attention masks, classifier-free guidance with an empty (or given) negative
prompt for both encoders, latents from a seeded CPU generator, 2D rotary embeddings over the
resize-crop grid region, `image_meta_size` from original size, target size and crop, style 0, the
first half of the learned-sigma output as the noise prediction, optional guidance rescale, DDPM
steps that draw variance noise from the same generator, VAE decode. Defaults from the model card:
25 steps, guidance 5.0, 1024×1024.

## Memory

fp32: transformer ≈ 6.1 GB, mT5-XL ≈ 6.7 GB, BERT ≈ 1.4 GB, VAE ≈ 0.3 GB. Reference and native fit
together.

## Validation

Random-weight parity for BERT and the transformer (tiny configs exact; full-config layouts on the
meta device), tokenizer parity for both tokenizers (fixed prompts and a fuzz set including Chinese
text), DDPM scheduler parity including generator noise, a tiny-snapshot end to end against
`HunyuanDiTPipeline`, then real-checkpoint component parity and end-to-end parity at 1024×1024 with
the defaults, plus the seed test.

## Task breakdown

1. Tokenizers: BERT WordPiece and the `spiece.model` reader for the mT5 tokenizer, with parity tests.
2. BERT encoder and DDPM scheduler, with parity tests.
3. HunyuanDiT transformer, with random-weight parity.
4. Backend (`hunyuan-dit` family), catalog entry, tiny-snapshot tests, real-checkpoint tests.
5. Real-checkpoint parity, docs; PR closes #34.
