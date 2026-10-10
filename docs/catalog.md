# Model catalog — imagehub 0.1

**23 architecture families, 52 published checkpoints.** A model is a separately published set of weights;
fp16 copies, non-EMA weights and single-file `.ckpt` releases do not count as separate models.

`pipeline("t2i", backend=model_id, model="local/folder")` selects the family by catalog id; the
`model` folder determines which weights are used.

**Validation levels:** `full` — tensors loaded strictly and parity with the recorded reference proven on CPU in
fp32. `header_only` — tensor names and shapes matched, generation not tested. `experimental` — not
tested with the original weights. In this release every model is `full`.

| Model id | Family | Source | Revision | Origin | Validation | Weight license | Awesome catalog |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sd-v1-1` | `stable-diffusion` | [CompVis/stable-diffusion-v1-1](https://huggingface.co/CompVis/stable-diffusion-v1-1) | `574555a` | author | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-2` | `stable-diffusion` | [CompVis/stable-diffusion-v1-2](https://huggingface.co/CompVis/stable-diffusion-v1-2) | `f519eeb` | author | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-3` | `stable-diffusion` | [CompVis/stable-diffusion-v1-3](https://huggingface.co/CompVis/stable-diffusion-v1-3) | `159f26f` | author | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-4` | `stable-diffusion` | [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | `133a221` | author | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-5` | `stable-diffusion` | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | `451f4fe` | community mirror | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v2-1` | `stable-diffusion` | [sd2-community/stable-diffusion-2-1](https://huggingface.co/sd2-community/stable-diffusion-2-1) | `bb21548` | community mirror | full | CreativeML-OpenRAIL++-M | `stable-diffusion-2` |
| `sd-turbo` | `stable-diffusion` | [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) | `b261bac` | author | full | sai-nc-community (non-commercial) | `sdxl-turbo` (SD Turbo) |
| `lcm-dreamshaper-v7` | `stable-diffusion` | [SimianLuo/LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) | `a85df6a` | author | full | MIT | `lcm` |
| `sdxl-base-1.0` | `stable-diffusion-xl` | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | `4621659` | author | full | CreativeML Open RAIL++-M | `sdxl` |
| `sdxl-lightning-2step` | `stable-diffusion-xl` | [ByteDance/SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) | `c9a24f4` | author | full | CreativeML Open RAIL++-M | `sdxl-lightning` |
| `sdxl-lightning-4step` | `stable-diffusion-xl` | [ByteDance/SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) | `c9a24f4` | author | full | CreativeML Open RAIL++-M | `sdxl-lightning` |
| `sdxl-lightning-8step` | `stable-diffusion-xl` | [ByteDance/SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) | `c9a24f4` | author | full | CreativeML Open RAIL++-M | `sdxl-lightning` |
| `hyper-sdxl-1step` | `stable-diffusion-xl` | [ByteDance/Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD) | `bc08d97` | author | full | Unspecified | `hyper-sd` |
| `dmd2-sdxl-4step` | `stable-diffusion-xl` | [tianweiy/DMD2](https://huggingface.co/tianweiy/DMD2) | `be22767` | author | full | CC-BY-NC-4.0 | `dmd2` |
| `dmd2-sdxl-1step` | `stable-diffusion-xl` | [tianweiy/DMD2](https://huggingface.co/tianweiy/DMD2) | `be22767` | author | full | CC-BY-NC-4.0 | `dmd2` |
| `sdxl-turbo` | `stable-diffusion-xl` | [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) | `7115331` | author | full | sai-nc-community (non-commercial) | `sdxl-turbo` |
| `ssd-1b` | `stable-diffusion-xl` | [segmind/SSD-1B](https://huggingface.co/segmind/SSD-1B) | `60987f3` | author | full | Apache-2.0 | `ssd-1b` |
| `playground-v2.5` | `stable-diffusion-xl` | [playgroundai/playground-v2.5-1024px-aesthetic](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic) | `1e032f1` | author | full | playground-v2dot5-community | `playground-v2` |
| `sd3-medium` | `stable-diffusion-3` | [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) | `ea42f8c` | author | full | Stability AI Community License | `stable-diffusion-3` |
| `sd3.5-medium` | `stable-diffusion-3` | [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | `b940f67` | author | full | Stability AI Community License | `stable-diffusion-3` |
| `sd3.5-large` | `stable-diffusion-3` | [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) | `ceddf0a` | author | full | Stability AI Community License | `stable-diffusion-3` |
| `sd3.5-large-turbo` | `stable-diffusion-3` | [stabilityai/stable-diffusion-3.5-large-turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo) | `ec07796` | author | full | Stability AI Community License | `stable-diffusion-3` |
| `flux1-schnell` | `flux` | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | `741f7c3` | author | full | Apache-2.0 | `flux-1` |
| `flux1-dev` | `flux` | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | `3de623f` | author | full | FLUX.1-dev Non-Commercial License | `flux-1` |
| `flux1-krea-dev` | `flux` | [black-forest-labs/FLUX.1-Krea-dev](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) | `8162a9c` | author | full | FLUX.1-dev Non-Commercial License | `flux-1` |
| `flux1-kontext-dev` | `flux` | [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | `24e9ded` | author | full | FLUX.1-dev Non-Commercial License | `flux-1-kontext` |
| `flux2-klein-4b` | `flux2` | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | `e7b7dc2` | author | full | Apache-2.0 | `flux-2` |
| `flux2-klein-9b` | `flux2` | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | `92196c8` | author | full | FLUX Non-Commercial License | `flux-2` |
| `hunyuan-dit-v1.1-distilled` | `hunyuan-dit` | [Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled) | `527cf2e` | author | full | Tencent Hunyuan Community License | `hunyuan-dit` |
| `kandinsky-2.2` | `kandinsky22` | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) | `9ae140d` | author | full | Apache-2.0 | `kandinsky-2` |
| `pixart-sigma-xl-1024` | `pixart` | [PixArt-alpha/PixArt-Sigma-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS) | `e102b35` | author | full | CreativeML-OpenRAIL++-M | `pixart-sigma` |
| `pixart-alpha-xl-1024` | `pixart` | [PixArt-alpha/PixArt-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) | `b89adad` | author | full | CreativeML-OpenRAIL++-M | `pixart-alpha` |
| `sana-1600m-1024` | `sana` | [Efficient-Large-Model/Sana_1600M_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers) | `ac0da2f` | author | full | Apache-2.0 | `sana` |
| `sana-600m-1024` | `sana` | [Efficient-Large-Model/Sana_600M_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers) | `28f3af7` | author | full | Apache-2.0 | `sana` |
| `kolors` | `kolors` | [Kwai-Kolors/Kolors-diffusers](https://huggingface.co/Kwai-Kolors/Kolors-diffusers) | `7e091c7` | author | full | Apache-2.0 | `kolors` |
| `lumina-image-2.0` | `lumina2` | [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) | `53504ab` | author | full | Apache-2.0 | `lumina-image-2` |
| `amused-256` | `amused` | [amused/amused-256](https://huggingface.co/amused/amused-256) | `09b6259` | author | full | OpenRAIL++ | `amused` |
| `amused-512` | `amused` | [amused/amused-512](https://huggingface.co/amused/amused-512) | `6f3f309` | author | full | OpenRAIL++ | `amused` |
| `auraflow-v0.3` | `auraflow` | [fal/AuraFlow-v0.3](https://huggingface.co/fal/AuraFlow-v0.3) | `2cd8588` | author | full | Apache-2.0 | `auraflow` |
| `krea2-turbo` | `krea2` | [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) | `98e0fe1` | author | full | Krea 2 Community License | `krea-2` |
| `longcat-image` | `longcat-image` | [meituan-longcat/LongCat-Image](https://huggingface.co/meituan-longcat/LongCat-Image) | `d2ea50b` | author | full | Apache-2.0 | `longcat-image` |
| `cogview3-plus-3b` | `cogview3` | [zai-org/CogView3-Plus-3B](https://huggingface.co/zai-org/CogView3-Plus-3B) | `5d70e40` | author | full | Apache-2.0 | `cogview3-plus` |
| `cogview4-6b` | `cogview4` | [zai-org/CogView4-6B](https://huggingface.co/zai-org/CogView4-6B) | `63a52b7` | author | full | Apache-2.0 | `cogview4` |
| `ideogram4-fp8` | `ideogram4` | [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8) | `ee79a72` | author | full | Ideogram 4 Non-Commercial License | `ideogram-4` |
| `qwen-image-2.1` | `qwen-image` | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | `d26bb61` | author | full | Qwen Research License (research use; see the license) | `qwen-image-2` |
| `z-image` | `z-image` | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) | `04cc4ab` | author | full | Apache-2.0 | `z-image` |
| `z-image-turbo` | `z-image` | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | `f332072` | author | full | Apache-2.0 | `z-image` |
| `janus-pro-1b` | `janus` | [deepseek-community/Janus-Pro-1B](https://huggingface.co/deepseek-community/Janus-Pro-1B) | `1655280` | community conversion | full | DeepSeek Model License | `janus-pro` |
| `janus-pro-7b` | `janus` | [deepseek-community/Janus-Pro-7B](https://huggingface.co/deepseek-community/Janus-Pro-7B) | `8f23139` | community conversion | full | DeepSeek Model License | `janus-pro` |
| `stable-cascade` | `stable_cascade` | [stabilityai/stable-cascade](https://huggingface.co/stabilityai/stable-cascade) | `a89f66d` | author | full | stable-cascade-nc-community (non-commercial) | `wuerstchen` |
| `stable-cascade-lite` | `stable_cascade` | [stabilityai/stable-cascade](https://huggingface.co/stabilityai/stable-cascade) | `a89f66d` | author | full | stable-cascade-nc-community (non-commercial) | `wuerstchen` |
| `wuerstchen-v2` | `wuerstchen` | [warp-ai/wuerstchen](https://huggingface.co/warp-ai/wuerstchen) | `c3da414` | author | full | mit | `wuerstchen` |

The original repository of `sd-v1-5` (`runwayml/stable-diffusion-v1-5`) has been removed; the
repository used is a community mirror. The v1.1–v1.3 fp32 weights are published only as `.bin`
files, so they are loaded with `weights_only=True`, containing tensors only.

Composite checkpoints can name a base model, omit base files with `base_exclude`, and overlay their
own files. They can also use `subfolders` to place files from a second repository in a named
snapshot subfolder. DMD2 excludes the base U-Net safetensors because its replacement U-Net is a
PyTorch pickle under a different name. Kandinsky 2.2 places the prior repository at revision
`9fc51ad` in the snapshot's `prior/` subfolder while the decoder remains at the snapshot root.
Stable Cascade does the same with `stabilityai/stable-cascade-prior` at revision `7ca32c2`, and
Würstchen v2 with `warp-ai/wuerstchen-prior` at revision `c41be63`.
`python -m imagehub.pretrained` downloads both sources, each pinned and verified, and the snapshot
manifest records provenance for every file.

The `hyper-sdxl-1step` model card has no license metadata. The `ByteDance/Hyper-SD` repository's
license file covers only the FLUX-related files, so the Hyper-SDXL weights are recorded as
`Unspecified`.

The `dmd2-sdxl-4step` and `dmd2-sdxl-1step` weights are licensed under CC-BY-NC-4.0. The license
prohibits commercial use; see the license before downloading or using the weights.

The catalog uses the `deepseek-community/Janus-Pro-1B` and `deepseek-community/Janus-Pro-7B`
transformers conversions with `community-conversion` provenance because the authors' repositories
ship weights for the original `janus` package rather than the transformers layout.

All four Stable Diffusion 3 and 3.5 repositories are gated. Accept their terms on Hugging Face,
then set `HF_TOKEN` when downloading with `python -m imagehub.pretrained`.

Both FLUX.1 repositories are gated. Accept their terms on Hugging Face, then set `HF_TOKEN` when
downloading with `python -m imagehub.pretrained`. The schnell weights use Apache-2.0; the dev
weights use the FLUX.1-dev Non-Commercial License.

The `flux2-klein-9b` repository is gated. Accept its terms on Hugging Face, then set `HF_TOKEN`
when downloading with `python -m imagehub.pretrained`.

The HunyuanDiT weights use the Tencent Hunyuan Community License, which excludes use in the
European Union, the United Kingdom and South Korea; see the license.

The `krea2-turbo` repository is gated. Accept its terms on Hugging Face, then set `HF_TOKEN` when
downloading with `python -m imagehub.pretrained`. The weights use the Krea 2 Community License;
see the license.

The `ideogram4-fp8` repository is gated. Accept its terms on Hugging Face, then set `HF_TOKEN` when
downloading with `python -m imagehub.pretrained`. The weights use the Ideogram 4 Non-Commercial
License; see the license.

Kolors publishes float16 weights only (`*.fp16.safetensors`), so its file list uses those names. Parity is
measured on their float32 upcast, which is also what diffusers computes with
`from_pretrained(..., variant="fp16", torch_dtype=torch.float32)`.

The `stable-cascade` and `stable-cascade-lite` weights use the Stable Cascade Non-Commercial
Community License (`stable-cascade-nc-community`), which prohibits commercial use; see the license
before downloading or using the weights. Both models share one decoder repository and one prior
repository and differ only in the stage C and stage B files they select (`prior` and `decoder`, or
`prior_lite` and `decoder_lite`).
