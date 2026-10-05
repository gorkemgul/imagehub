# Model catalog — imagehub 0.1

**4 architecture families, 12 published checkpoints.** A model is a separately published set of weights;
fp16 copies, non-EMA weights and single-file `.ckpt` releases do not count as separate models.

`pipeline("t2i", backend=model_id, model="local/folder")` selects the family by catalog id; the
`model` folder determines which weights are used.

**Validation levels:** `full` — tensors loaded strictly and parity with diffusers proven on CPU in
fp32. `header_only` — tensor names and shapes matched, generation not tested. `experimental` — not
tested with the original weights. In this release every model is `full`.

| Model id | Family | Source | Revision | Origin | Validation | Weight license | Awesome catalog |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sd-v1-1` | `stable-diffusion` | [CompVis/stable-diffusion-v1-1](https://huggingface.co/CompVis/stable-diffusion-v1-1) | `574555a` | author | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-2` | `stable-diffusion` | [CompVis/stable-diffusion-v1-2](https://huggingface.co/CompVis/stable-diffusion-v1-2) | `f519eeb` | author | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-3` | `stable-diffusion` | [CompVis/stable-diffusion-v1-3](https://huggingface.co/CompVis/stable-diffusion-v1-3) | `159f26f` | author | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-4` | `stable-diffusion` | [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | `133a221` | author | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-5` | `stable-diffusion` | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | `451f4fe` | community mirror | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sdxl-base-1.0` | `stable-diffusion-xl` | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | `4621659` | author | full | CreativeML Open RAIL++-M | `sdxl` |
| `sdxl-turbo` | `stable-diffusion-xl` | [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) | `7115331` | author | full | sai-nc-community (non-commercial) | `sdxl-turbo` |
| `ssd-1b` | `stable-diffusion-xl` | [segmind/SSD-1B](https://huggingface.co/segmind/SSD-1B) | `60987f3` | author | full | Apache-2.0 | `ssd-1b` |
| `playground-v2.5` | `stable-diffusion-xl` | [playgroundai/playground-v2.5-1024px-aesthetic](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic) | `1e032f1` | author | full | playground-v2dot5-community | `playground-v2` |
| `z-image-turbo` | `z-image` | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | `f332072` | author | full | Apache-2.0 | `z-image` |
| `janus-pro-1b` | `janus` | [deepseek-community/Janus-Pro-1B](https://huggingface.co/deepseek-community/Janus-Pro-1B) | `1655280` | community conversion | full | DeepSeek Model License | `janus-pro` |
| `janus-pro-7b` | `janus` | [deepseek-community/Janus-Pro-7B](https://huggingface.co/deepseek-community/Janus-Pro-7B) | `8f23139` | community conversion | full | DeepSeek Model License | `janus-pro` |

The original repository of `sd-v1-5` (`runwayml/stable-diffusion-v1-5`) has been removed; the
repository used is a community mirror. The v1.1–v1.3 fp32 weights are published only as `.bin`
files, so they are loaded with `weights_only=True`, containing tensors only.

The catalog uses the `deepseek-community/Janus-Pro-1B` and `deepseek-community/Janus-Pro-7B`
transformers conversions with `community-conversion` provenance because the authors' repositories
ship weights for the original `janus` package rather than the transformers layout.
