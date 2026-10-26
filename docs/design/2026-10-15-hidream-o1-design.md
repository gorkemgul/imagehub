# HiDream-O1 design

- Date: 2026-10-15
- Issue: #135
- Workflow: `.ai/skills/add-or-validate-image-model/SKILL.md`

## Scope

`hidream-o1-image`, the undistilled `HiDream-ai/HiDream-O1-Image` checkpoint at revision
`0b0901d99f200389e138c61946af1185f5f49a13`, as a new `hidream-o1` family for text-to-image.
The author repository is ungated and both its model card and the official source declare MIT.
The same checkpoint can condition generation on images, but this implementation exposes only the
text-to-image path; the catalog declares `image_input=None`.

## Weights

The checkpoint contains eight float32 safetensors shards: 759 tensors and 35,219,551,168 bytes.
It is a Qwen3-VL-8B layout plus a pixel-generation input projection, timestep embedder and output
projection. Images are represented directly as non-overlapping 32x32 RGB patches, so no VAE is
loaded. The tokenizer and Qwen3-VL processor configuration are stored at the repository root.

## Reference

No released diffusers pipeline contains this model. The reference is the official
`HiDream-ai/HiDream-O1-Image` repository at commit
`2c2d29ff729e48f33e41f49edfdbd81d5ac103b4`: `models/qwen3_vl_transformers.py` for the model,
`models/pipeline.py` for prompt construction and sampling, and `models/fm_solvers_unipc.py` for
the default scheduler. Reference tests import this pinned checkout directly; optional acceleration
packages are not required by the non-flash random-weight path.

## Components

| Part | Status |
| --- | --- |
| Qwen tokenizer and chat template | reuse `tokenization_qwen.py` |
| Qwen3-VL text decoder (36 layers, 4096 wide, 32/8 heads) | reuse `models/qwen3_vl_text.py` |
| Qwen3-VL vision encoder for optional image input | published layout represented; execution out of scope |
| 32x32 pixel patch input projection, timestep embedding and pixel output head | **new** |
| Mixed causal/full attention generation forward | **new** |
| Flow-UniPC sampling loop | **new** |

## Pipeline details

The prompt uses the checkpoint chat template followed by `<|boi_token|><|tms_token|>`. The
timestep token embedding is computed from `t * 1000`. Target pixels start from CPU noise seeded
with `seed + 1` and scaled by 8.0. Generation tokens attend to the complete joint sequence while
text tokens remain causal. The model predicts clean pixel patches; the pipeline converts these to
velocity with `(x0 - x) / sigma`, applies classifier-free guidance (default 5.0), and uses 50
Flow-UniPC steps with shift 3.0. Final patches are rearranged without a decoder and rounded to
uint8 after mapping from `[-1, 1]`.

## Memory

The published float32 files total about 32.8 GiB. Component parity uses tiny random models; real
checkpoint validation must compute and free the official result before loading the native model.

## Validation

Inventory includes every config, index, and shard header. Random-weight tests compare the new
patch and timestep modules, joint generation forward, scheduler and complete sampling loop exactly
with the pinned official implementation. The real-checkpoint test skips without a local snapshot
and records separate `e2e_hidream-o1-image` and seed-42 `seed_e2e_hidream-o1-image` results through
the public pipeline.

## Out of scope for this increment

Image conditioning and editing, subject personalization, layout and skeleton conditioning, the
prompt-refiner agent, distributed inference, flash attention, the distilled Dev releases, and
reduced precision.

