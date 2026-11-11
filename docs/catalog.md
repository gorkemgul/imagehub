# Model catalog — imagehub 0.1

**{{ ih.families }} architecture families, {{ ih.models }} published checkpoints.** A model is a separately published set of weights;
fp16 copies, non-EMA weights and single-file `.ckpt` releases do not count as separate models.

`pipeline("t2i", backend=model_id, model="local/folder")` selects the family by catalog id; the
`model` folder determines which weights are used.

**Validation levels:** `full` — tensors loaded strictly and parity with the recorded reference proven on CPU in
fp32. `header_only` — tensor names and shapes matched, generation not tested. `experimental` — not
tested with the original weights. Models are `full` unless noted otherwise.

`Weight license` governs the separately downloaded checkpoint. `Code license` and `Code commercial
use` govern this repository's implementation for that row; they default to Apache-2.0 and `yes`.
`Gated` means the publisher requires access approval before download.

| Model id | Family | Source | Revision | Origin | Validation | Gated | Weight license | Code license | Code commercial use | Awesome catalog |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ldm-text2im-large-256` | `ldm` | [CompVis/ldm-text2im-large-256](https://huggingface.co/CompVis/ldm-text2im-large-256) | `30de525` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `ldm` |
| `sd-v1-1` | `stable-diffusion` | [CompVis/stable-diffusion-v1-1](https://huggingface.co/CompVis/stable-diffusion-v1-1) | `574555a` | author | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `stable-diffusion-1` |
| `sd-v1-2` | `stable-diffusion` | [CompVis/stable-diffusion-v1-2](https://huggingface.co/CompVis/stable-diffusion-v1-2) | `f519eeb` | author | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `stable-diffusion-1` |
| `sd-v1-3` | `stable-diffusion` | [CompVis/stable-diffusion-v1-3](https://huggingface.co/CompVis/stable-diffusion-v1-3) | `159f26f` | author | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `stable-diffusion-1` |
| `sd-v1-4` | `stable-diffusion` | [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | `133a221` | author | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `stable-diffusion-1` |
| `sd-v1-5` | `stable-diffusion` | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | `451f4fe` | community mirror | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `stable-diffusion-1` |
| `dreamshaper-8` | `stable-diffusion` | [Lykon/dreamshaper-8](https://huggingface.co/Lykon/dreamshaper-8) | `a7e52b9` | author | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `stable-diffusion-1` |
| `segmind-small-sd` | `stable-diffusion` | [segmind/small-sd](https://huggingface.co/segmind/small-sd) | `f8b087e` | author | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `bk-sdm` |
| `segmind-tiny-sd` | `stable-diffusion` | [segmind/tiny-sd](https://huggingface.co/segmind/tiny-sd) | `cad0bd7` | author | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `bk-sdm` |
| `bk-sdm-base` | `stable-diffusion` | [nota-ai/bk-sdm-base](https://huggingface.co/nota-ai/bk-sdm-base) | `0c03b7a` | author | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `bk-sdm` |
| `bk-sdm-small` | `stable-diffusion` | [nota-ai/bk-sdm-small](https://huggingface.co/nota-ai/bk-sdm-small) | `572238d` | author | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `bk-sdm` |
| `bk-sdm-tiny` | `stable-diffusion` | [nota-ai/bk-sdm-tiny](https://huggingface.co/nota-ai/bk-sdm-tiny) | `0364108` | author | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `bk-sdm` |
| `bk-sdm-v2-base` | `stable-diffusion` | [nota-ai/bk-sdm-v2-base](https://huggingface.co/nota-ai/bk-sdm-v2-base) | `21915be` | author | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `bk-sdm` |
| `bk-sdm-v2-small` | `stable-diffusion` | [nota-ai/bk-sdm-v2-small](https://huggingface.co/nota-ai/bk-sdm-v2-small) | `ad82c51` | author | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `bk-sdm` |
| `bk-sdm-v2-tiny` | `stable-diffusion` | [nota-ai/bk-sdm-v2-tiny](https://huggingface.co/nota-ai/bk-sdm-v2-tiny) | `68277af` | author | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `bk-sdm` |
| `sd-v2-1` | `stable-diffusion` | [sd2-community/stable-diffusion-2-1](https://huggingface.co/sd2-community/stable-diffusion-2-1) | `bb21548` | community mirror | full | no | CreativeML-OpenRAIL++-M | Apache-2.0 | yes | `stable-diffusion-2` |
| `sd-turbo` | `stable-diffusion` | [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) | `b261bac` | author | full | no | sai-nc-community | Apache-2.0 | yes | `sdxl-turbo` (SD Turbo) |
| `sdxs-512-0.9` | `stable-diffusion` | [IDKiro/sdxs-512-0.9](https://huggingface.co/IDKiro/sdxs-512-0.9) | `c332f05` | author | full | no | CreativeML-OpenRAIL++-M | Apache-2.0 | yes | `sdxs` |
| `lcm-dreamshaper-v7` | `stable-diffusion` | [SimianLuo/LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) | `a85df6a` | author | full | no | MIT | Apache-2.0 | yes | `lcm` |
| `dreamlite-base` | `dreamlite` | [carlofkl/DreamLite-base](https://huggingface.co/carlofkl/DreamLite-base) | `a9a0f15` | author | full | no | CC-BY-NC-4.0 | Apache-2.0 | yes | `dreamlite` |
| `dreamlite-mobile` | `dreamlite` | [carlofkl/DreamLite-mobile](https://huggingface.co/carlofkl/DreamLite-mobile) | `a48c656` | author | full | no | CC-BY-NC-4.0 | Apache-2.0 | yes | `dreamlite` |
| `sdxl-base-1.0` | `stable-diffusion-xl` | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | `4621659` | author | full | no | CreativeML-OpenRAIL++-M | Apache-2.0 | yes | `sdxl` |
| `juggernaut-xl-v9` | `stable-diffusion-xl` | [RunDiffusion/Juggernaut-XL-v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) | `cf41923` | author | full | no | CreativeML-OpenRAIL-M | Apache-2.0 | yes | `sdxl` |
| `animagine-xl-4.0` | `stable-diffusion-xl` | [cagliostrolab/animagine-xl-4.0](https://huggingface.co/cagliostrolab/animagine-xl-4.0) | `2b7c1b3` | author | full | no | CreativeML-OpenRAIL++-M | Apache-2.0 | yes | `sdxl` |
| `dreamshaper-xl-1.0` | `stable-diffusion-xl` | [Lykon/dreamshaper-xl-1-0](https://huggingface.co/Lykon/dreamshaper-xl-1-0) | `41e6644` | author | full | no | CreativeML-OpenRAIL++-M | Apache-2.0 | yes | `sdxl` |
| `sdxl-refiner-1.0` | `stable-diffusion-xl` | [stabilityai/stable-diffusion-xl-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) | `5d4cfe8` | author | full | no | CreativeML-OpenRAIL++-M | Apache-2.0 | yes | `sdxl` |
| `sdxl-lightning-2step` | `stable-diffusion-xl` | [ByteDance/SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) | `c9a24f4` | author | full | no | CreativeML-OpenRAIL++-M | Apache-2.0 | yes | `sdxl-lightning` |
| `sdxl-lightning-4step` | `stable-diffusion-xl` | [ByteDance/SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) | `c9a24f4` | author | full | no | CreativeML-OpenRAIL++-M | Apache-2.0 | yes | `sdxl-lightning` |
| `sdxl-lightning-8step` | `stable-diffusion-xl` | [ByteDance/SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) | `c9a24f4` | author | full | no | CreativeML-OpenRAIL++-M | Apache-2.0 | yes | `sdxl-lightning` |
| `hyper-sdxl-1step` | `stable-diffusion-xl` | [ByteDance/Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD) | `bc08d97` | author | full | no | Unspecified | Apache-2.0 | yes | `hyper-sd` |
| `dmd2-sdxl-4step` | `stable-diffusion-xl` | [tianweiy/DMD2](https://huggingface.co/tianweiy/DMD2) | `be22767` | author | full | no | CC-BY-NC-4.0 | Apache-2.0 | yes | `dmd2` |
| `dmd2-sdxl-1step` | `stable-diffusion-xl` | [tianweiy/DMD2](https://huggingface.co/tianweiy/DMD2) | `be22767` | author | full | no | CC-BY-NC-4.0 | Apache-2.0 | yes | `dmd2` |
| `sdxl-turbo` | `stable-diffusion-xl` | [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) | `7115331` | author | full | no | sai-nc-community | Apache-2.0 | yes | `sdxl-turbo` |
| `ssd-1b` | `stable-diffusion-xl` | [segmind/SSD-1B](https://huggingface.co/segmind/SSD-1B) | `60987f3` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `ssd-1b` |
| `segmind-vega` | `stable-diffusion-xl` | [segmind/Segmind-Vega](https://huggingface.co/segmind/Segmind-Vega) | `7714c43` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `segmind-vega` |
| `koala-700m` | `stable-diffusion-xl` | [etri-vilab/koala-700m](https://huggingface.co/etri-vilab/koala-700m) | `5e73a31` | author | full | no | Unspecified | Apache-2.0 | yes | `koala` |
| `koala-1b` | `stable-diffusion-xl` | [etri-vilab/koala-1b](https://huggingface.co/etri-vilab/koala-1b) | `8ef26f3` | author | full | no | Unspecified | Apache-2.0 | yes | `koala` |
| `koala-lightning-700m` | `stable-diffusion-xl` | [etri-vilab/koala-lightning-700m](https://huggingface.co/etri-vilab/koala-lightning-700m) | `358e0e3` | author | full | no | Unspecified | Apache-2.0 | yes | `koala` |
| `koala-lightning-1b` | `stable-diffusion-xl` | [etri-vilab/koala-lightning-1b](https://huggingface.co/etri-vilab/koala-lightning-1b) | `b4114f7` | author | full | no | Unspecified | Apache-2.0 | yes | `koala` |
| `playground-v2` | `stable-diffusion-xl` | [playgroundai/playground-v2-1024px-aesthetic](https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic) | `737a24e` | author | full | no | playground-v2-community | Apache-2.0 | yes | `playground-v2` |
| `playground-v2.5` | `stable-diffusion-xl` | [playgroundai/playground-v2.5-1024px-aesthetic](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic) | `1e032f1` | author | full | no | playground-v2dot5-community | Apache-2.0 | yes | `playground-v2` |
| `sd3-medium` | `stable-diffusion-3` | [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) | `ea42f8c` | author | full | yes | Stability AI Non-Commercial Research Community License | Apache-2.0 | yes | `stable-diffusion-3` |
| `sd3.5-medium` | `stable-diffusion-3` | [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | `b940f67` | author | full | yes | Stability AI Community License | Apache-2.0 | yes | `stable-diffusion-3` |
| `sd3.5-large` | `stable-diffusion-3` | [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) | `ceddf0a` | author | full | yes | Stability AI Community License | Apache-2.0 | yes | `stable-diffusion-3` |
| `sd3.5-large-turbo` | `stable-diffusion-3` | [stabilityai/stable-diffusion-3.5-large-turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo) | `ec07796` | author | full | yes | Stability AI Community License | Apache-2.0 | yes | `stable-diffusion-3` |
| `flux1-schnell` | `flux` | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | `741f7c3` | author | full | yes | Apache-2.0 | Apache-2.0 | yes | `flux-1` |
| `flux1-dev` | `flux` | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | `3de623f` | author | full | yes | FLUX.1-dev Non-Commercial License | Apache-2.0 | yes | `flux-1` |
| `flux1-krea-dev` | `flux` | [black-forest-labs/FLUX.1-Krea-dev](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) | `8162a9c` | author | full | yes | FLUX.1-dev Non-Commercial License | Apache-2.0 | yes | `flux-1` |
| `flux-1-lite-8b` | `flux` | [Freepik/flux.1-lite-8B](https://huggingface.co/Freepik/flux.1-lite-8B) | `16b9373` | author | full | no | FLUX.1-dev Non-Commercial License | Apache-2.0 | yes | `flux-1` |
| `flux1-kontext-dev` | `flux` | [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | `24e9ded` | author | full | yes | FLUX.1-dev Non-Commercial License | Apache-2.0 | yes | `flux-1-kontext` |
| `visualcloze-384` | `visualcloze` | [VisualCloze/VisualClozePipeline-384](https://huggingface.co/VisualCloze/VisualClozePipeline-384) | `59c469d` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `visualcloze` |
| `joyai-image-edit` | `joyai-image` | [jdopensource/JoyAI-Image-Edit-Diffusers](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Diffusers) | `4b41fb2` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `joyai-image` |
| `joyai-image-edit-plus` | `joyai-image` | [jdopensource/JoyAI-Image-Edit-Plus-Diffusers](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Plus-Diffusers) | `c268646` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `joyai-image` |
| `flux2-klein-4b` | `flux2` | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | `e7b7dc2` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `flux-2` |
| `flux2-klein-9b` | `flux2` | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | `92196c8` | author | full | yes | FLUX Non-Commercial License | Apache-2.0 | yes | `flux-2` |
| `hunyuan-dit-v1.1-distilled` | `hunyuan-dit` | [Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled) | `527cf2e` | author | full | no | Tencent Hunyuan Community License | Apache-2.0 | yes | `hunyuan-dit` |
| `hunyuan-dit-v1.2` | `hunyuan-dit` | [Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers) | `5e96094` | author | full | no | Tencent Hunyuan Community License | Apache-2.0 | yes | `hunyuan-dit` |
| `hunyuan-dit-v1.2-distilled` | `hunyuan-dit` | [Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled) | `ba991d1` | author | full | no | Tencent Hunyuan Community License | Apache-2.0 | yes | `hunyuan-dit` |
| `deepfloyd-if-i-xl` | `deepfloyd-if` | [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0) | `c03d510` | author | full | yes | DeepFloyd IF License (non-commercial) | Apache-2.0 | yes | `deepfloyd-if` |
| `kandinsky-2-1` | `kandinsky21` | [kandinsky-community/kandinsky-2-1](https://huggingface.co/kandinsky-community/kandinsky-2-1) | `04b3666` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `kandinsky-2` |
| `kandinsky-2.2` | `kandinsky22` | [kandinsky-community/kandinsky-2-2-decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder) | `9ae140d` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `kandinsky-2` |
| `kandinsky-3` | `kandinsky3` | [kandinsky-community/kandinsky-3](https://huggingface.co/kandinsky-community/kandinsky-3) | `bf79e6c` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `kandinsky-3` |
| `kandinsky-5-t2i-lite` | `kandinsky5` | [kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers) | `25da1e8` | author | full | no | MIT | Apache-2.0 | yes | `kandinsky-5` |
| `pixart-sigma-xl-1024` | `pixart` | [PixArt-alpha/PixArt-Sigma-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS) | `e102b35` | author | full | no | CreativeML-OpenRAIL++-M | Apache-2.0 | yes | `pixart-sigma` |
| `pixart-alpha-xl-1024` | `pixart` | [PixArt-alpha/PixArt-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) | `b89adad` | author | full | no | CreativeML-OpenRAIL++-M | Apache-2.0 | yes | `pixart-alpha` |
| `pixart-alpha-xl-512` | `pixart` | [PixArt-alpha/PixArt-XL-2-512x512](https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512) | `50f7021` | author | full | no | CreativeML-OpenRAIL++-M | Apache-2.0 | yes | `pixart-alpha` |
| `pixart-sigma-xl-512` | `pixart` | [PixArt-alpha/PixArt-Sigma-XL-2-512-MS](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS) | `76fb7eb` | author | full | no | CreativeML-OpenRAIL++-M | Apache-2.0 | yes | `pixart-sigma` |
| `pixart-sigma-xl-256` | `pixart` | [PixArt-alpha/PixArt-Sigma-XL-2-256x256](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-256x256) | `495cb1b` | author | full | no | OpenRAIL (exact terms unspecified) | Apache-2.0 | yes | `pixart-sigma` |
| `pixart-sigma-xl-2k` | `pixart` | [PixArt-alpha/PixArt-Sigma-XL-2-2K-MS](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-2K-MS) | `cd274f4` | author | full | no | OpenRAIL (exact terms unspecified) | Apache-2.0 | yes | `pixart-sigma` |
| `pixart-lcm-xl-1024` | `pixart` | [PixArt-alpha/PixArt-LCM-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-LCM-XL-2-1024-MS) | `e153060` | author | header_only | no | CreativeML-OpenRAIL++-M | Apache-2.0 | yes | `pixart-delta` |
| `sana-1600m-1024` | `sana` | [Efficient-Large-Model/Sana_1600M_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers) | `ac0da2f` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `sana` |
| `sana-600m-1024` | `sana` | [Efficient-Large-Model/Sana_600M_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers) | `28f3af7` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `sana` |
| `sana-sprint-1.6b-1024` | `sana` | [Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers) | `19683c5` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `sana` |
| `sana-1.5-4.8b-1024` | `sana` | [Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers) | `9468102` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `sana` |
| `kolors` | `kolors` | [Kwai-Kolors/Kolors-diffusers](https://huggingface.co/Kwai-Kolors/Kolors-diffusers) | `7e091c7` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `kolors` |
| `lumina-image-2.0` | `lumina2` | [Alpha-VLLM/Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) | `53504ab` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `lumina-image-2` |
| `lumina-next` | `lumina-next` | [Alpha-VLLM/Lumina-Next-SFT-diffusers](https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers) | `0ee5ec9` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `lumina-next` |
| `lumina-dimoo` | `lumina-dimoo` | [Alpha-VLLM/Lumina-DiMOO](https://huggingface.co/Alpha-VLLM/Lumina-DiMOO) | `f014549` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `lumina-dimoo` |
| `omnigen-v1` | `omnigen` | [Shitao/OmniGen-v1-diffusers](https://huggingface.co/Shitao/OmniGen-v1-diffusers) | `016e2f6` | author | full | no | MIT | Apache-2.0 | yes | `omnigen` |
| `chroma1-hd` | `chroma` | [lodestones/Chroma1-HD](https://huggingface.co/lodestones/Chroma1-HD) | `0e0c60e` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `chroma1-hd` |
| `chroma1-flash` | `chroma` | [lodestones/Chroma1-Flash](https://huggingface.co/lodestones/Chroma1-Flash) | `093c4f6` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `chroma1-flash` |
| `chroma1-base` | `chroma` | [lodestones/Chroma1-Base](https://huggingface.co/lodestones/Chroma1-Base) | `50ebe12` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `chroma1-base` |
| `amused-256` | `amused` | [amused/amused-256](https://huggingface.co/amused/amused-256) | `09b6259` | author | full | no | OpenRAIL++ | Apache-2.0 | yes | `amused` |
| `amused-512` | `amused` | [amused/amused-512](https://huggingface.co/amused/amused-512) | `6f3f309` | author | full | no | OpenRAIL++ | Apache-2.0 | yes | `amused` |
| `meissonic` | `meissonic` | [MeissonFlow/Meissonic](https://huggingface.co/MeissonFlow/Meissonic) | `7ee5068` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `meissonic` |
| `auraflow-v0.3` | `auraflow` | [fal/AuraFlow-v0.3](https://huggingface.co/fal/AuraFlow-v0.3) | `2cd8588` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `auraflow` |
| `prx-1024-t2i-beta` | `prx` | [Photoroom/prx-1024-t2i-beta](https://huggingface.co/Photoroom/prx-1024-t2i-beta) | `0370504` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `prx` |
| `prx-512-t2i-sft` | `prx` | [Photoroom/prx-512-t2i-sft](https://huggingface.co/Photoroom/prx-512-t2i-sft) | `2996423` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `prx` |
| `llamagen-t2i-xl-256` | `llamagen` | [peizesun/llamagen_t2i](https://huggingface.co/peizesun/llamagen_t2i) | `276f5c5` | author | full | no | MIT | Apache-2.0 | yes | `llamagen` |
| `llamagen-t2i-xl-512` | `llamagen` | [peizesun/llamagen_t2i](https://huggingface.co/peizesun/llamagen_t2i) | `276f5c5` | author | full | no | MIT | Apache-2.0 | yes | `llamagen` |
| `infinity-2b` | `infinity` | [FoundationVision/Infinity](https://huggingface.co/FoundationVision/Infinity) | `e710892` | author | full | no | MIT | Apache-2.0 | yes | `infinity` |
| `hart` | `hart` | [mit-han-lab/hart-0.7b-1024px](https://huggingface.co/mit-han-lab/hart-0.7b-1024px) | `b1521ff` | author | full | no | MIT | Apache-2.0 | yes | `hart` |
| `show-o-512` | `show-o` | [showlab/show-o-512x512](https://huggingface.co/showlab/show-o-512x512) | `dbe757d` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `show-o` |
| `emu3-gen` | `emu3` | [BAAI/Emu3-Gen](https://huggingface.co/BAAI/Emu3-Gen) | `40b3bb8` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `emu3` |
| `pixeldit` | `pixeldit` | [nvidia/PixelDiT-1300M-1024px](https://huggingface.co/nvidia/PixelDiT-1300M-1024px) | `7c63b99` | author | full | no | NVIDIA License (NSCLv1; non-commercial research or evaluation) | NSCLv1 | no | `pixeldit` |
| `krea2-turbo` | `krea2` | [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) | `98e0fe1` | author | full | yes | Krea 2 Community License | Apache-2.0 | yes | `krea-2` |
| `ernie-image` | `ernie-image` | [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) | `5346b31` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `ernie-image` |
| `longcat-image` | `longcat-image` | [meituan-longcat/LongCat-Image](https://huggingface.co/meituan-longcat/LongCat-Image) | `d2ea50b` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `longcat-image` |
| `cogview3-plus-3b` | `cogview3` | [zai-org/CogView3-Plus-3B](https://huggingface.co/zai-org/CogView3-Plus-3B) | `5d70e40` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `cogview3-plus` |
| `cogview4-6b` | `cogview4` | [zai-org/CogView4-6B](https://huggingface.co/zai-org/CogView4-6B) | `63a52b7` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `cogview4` |
| `ideogram4-fp8` | `ideogram4` | [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8) | `ee79a72` | author | full | yes | Ideogram 4 Non-Commercial License | Apache-2.0 | yes | `ideogram-4` |
| `qwen-image` | `qwen-image` | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | `75e0b4b` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `qwen-image` |
| `qwen-image-2.1` | `qwen-image` | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | `d26bb61` | author | full | no | Qwen Research License | Apache-2.0 | yes | `qwen-image-2` |
| `qwen-image-edit` | `qwen-image` | [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) | `ac7f931` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `qwen-image-edit` |
| `qwen-image-edit-2509` | `qwen-image` | [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) | `d3968ef` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `qwen-image-edit` |
| `ovis-image-7b` | `ovis-image` | [ATH-MaaS/Ovis-Image-7B](https://huggingface.co/ATH-MaaS/Ovis-Image-7B) | `41be1c5` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `ovis-image` |
| `z-image` | `z-image` | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) | `04cc4ab` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `z-image` |
| `z-image-turbo` | `z-image` | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | `f332072` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `z-image` |
| `janus-pro-1b` | `janus` | [deepseek-community/Janus-Pro-1B](https://huggingface.co/deepseek-community/Janus-Pro-1B) | `1655280` | community conversion | full | no | DeepSeek Model License | Apache-2.0 | yes | `janus-pro` |
| `janus-pro-7b` | `janus` | [deepseek-community/Janus-Pro-7B](https://huggingface.co/deepseek-community/Janus-Pro-7B) | `8f23139` | community conversion | full | no | DeepSeek Model License | Apache-2.0 | yes | `janus-pro` |
| `bagel-7b-mot` | `bagel` | [ByteDance-Seed/BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) | `5019f57` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `bagel` |
| `stable-cascade` | `stable-cascade` | [stabilityai/stable-cascade](https://huggingface.co/stabilityai/stable-cascade) | `a89f66d` | author | full | no | stable-cascade-nc-community | Apache-2.0 | yes | `wuerstchen` |
| `stable-cascade-lite` | `stable-cascade` | [stabilityai/stable-cascade](https://huggingface.co/stabilityai/stable-cascade) | `a89f66d` | author | full | no | stable-cascade-nc-community | Apache-2.0 | yes | `wuerstchen` |
| `wuerstchen-v2` | `wuerstchen` | [warp-ai/wuerstchen](https://huggingface.co/warp-ai/wuerstchen) | `c3da414` | author | full | no | mit | Apache-2.0 | yes | `wuerstchen` |
| `hidream-i1-fast` | `hidream` | [HiDream-ai/HiDream-I1-Fast](https://huggingface.co/HiDream-ai/HiDream-I1-Fast) | `4856a5d` | author | full | yes | MIT | Apache-2.0 | yes | `hidream-i1` |
| `hidream-i1-dev` | `hidream` | [HiDream-ai/HiDream-I1-Dev](https://huggingface.co/HiDream-ai/HiDream-I1-Dev) | `0fad2ea` | author | full | yes | MIT | Apache-2.0 | yes | `hidream-i1` |
| `hidream-i1-full` | `hidream` | [HiDream-ai/HiDream-I1-Full](https://huggingface.co/HiDream-ai/HiDream-I1-Full) | `8ccbbfb` | author | full | yes | MIT | Apache-2.0 | yes | `hidream-i1` |
| `hidream-o1-image` | `hidream-o1` | [HiDream-ai/HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) | `0b0901d` | author | full | no | MIT | Apache-2.0 | yes | `hidream-o1-image` |
| `glm-image` | `glm-image` | [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image) | `2c433cc` | author | full | no | MIT | Apache-2.0 | yes | `glm-image` |
| `cosmos-predict2-2b-t2i` | `cosmos2` | [nvidia/Cosmos-Predict2-2B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image) | `acdb5fd` | author | full | yes | NVIDIA Open Model License | Apache-2.0 | yes | `cosmos-predict2-2b-t2i` |
| `nextstep-1-large` | `nextstep` | [stepfun-ai/NextStep-1-Large](https://huggingface.co/stepfun-ai/NextStep-1-Large) | `97dd0b9` | author | full | no | Apache-2.0 | Apache-2.0 | yes | `nextstep-1` |

The original repository of `sd-v1-5` (`runwayml/stable-diffusion-v1-5`) has been removed; the
repository used is a community mirror. The v1.1–v1.3 fp32 weights are published only as `.bin`
files, so they are loaded with `weights_only=True`, containing tensors only.
Segmind Small-SD and Tiny-SD likewise publish tensor-only `.bin` component files. Small-SD's
tensors are float32; Tiny-SD has only float16 published tensors, which are upcast for CPU float32
validation.
Juggernaut XL v9 publishes only float16 component files. Animagine XL 4.0 likewise publishes
float16 component tensors, although its component filenames do not carry an `fp16` suffix. Both
are upcast for CPU float32 validation. DreamShaper XL 1.0 publishes float32 component tensors.
Cosmos-Predict2 is gated and requires accepting the NVIDIA Open Model License before download.
Its transformer and VAE are float32; its two T5 encoder shards are bfloat16 and are upcast for
CPU float32 inference.

Composite checkpoints can name a base model, omit base files with `base_exclude`, and overlay their
own files. They can also use `subfolders` to place files from a second repository in a named
snapshot subfolder. DMD2 excludes the base U-Net safetensors because its replacement U-Net is a
PyTorch pickle under a different name. Kandinsky 2.1 and 2.2 place their pinned prior repositories
in the snapshot's `prior/` subfolder while each decoder remains at the snapshot root.
The PixArt-Sigma 256, 512 and 2K repositories publish only a transformer, so those models use
`pixart-sigma-xl-1024` as their base and overlay their transformer files with their own.
Stable Cascade does the same with `stabilityai/stable-cascade-prior` at revision `7ca32c2`, and
Würstchen v2 with `warp-ai/wuerstchen-prior` at revision `c41be63`.
`python -m imagehub.pretrained` downloads both sources, each pinned and verified, and the snapshot
manifest records provenance for every file.

LlamaGen uses the same mechanism for its text encoder: `google/flan-t5-xl` at revision `7d6315d` goes
into the snapshot's `text_encoder/` subfolder, while the GPT and VQ-16 PyTorch checkpoints of
`peizesun/llamagen_t2i` (`format` `pytorch`, loaded with `weights_only=True`) stay at the root.
The Flan-T5 shards also hold the decoder, which imagehub never reads.

Infinity-2B also places `google/flan-t5-xl` at revision `7d6315d` in `text_encoder/`.
`infinity_2b_reg.pth` and `infinity_vae_d32reg.pth` remain at the snapshot root and are loaded as
weights-only PyTorch archives; training-only VAE perceptual tensors are verified but not loaded.

The `hyper-sdxl-1step` model card has no license metadata. The `ByteDance/Hyper-SD` repository's
license file covers only the FLUX-related files, so the Hyper-SDXL weights are recorded as
`Unspecified`.

The `dmd2-sdxl-4step` and `dmd2-sdxl-1step` weights are licensed under CC-BY-NC-4.0. The license
prohibits commercial use; see the license before downloading or using the weights.

The `dreamlite-base` and `dreamlite-mobile` weights are licensed under CC-BY-NC-4.0. The license
prohibits commercial use; see the license before downloading or using the weights.

The catalog uses the `deepseek-community/Janus-Pro-1B` and `deepseek-community/Janus-Pro-7B`
transformers conversions with `community-conversion` provenance because the authors' repositories
ship weights for the original `janus` package rather than the transformers layout.

All four Stable Diffusion 3 and 3.5 repositories are gated. Accept their terms on Hugging Face,
then set `HF_TOKEN` when downloading with `python -m imagehub.pretrained`. SD3 Medium uses the
non-commercial research license. The SD3.5 Community License allows registered commercial use below
USD 1 million annual revenue and requires an enterprise license at or above that threshold.

The Black Forest Labs FLUX.1 repositories are gated. Accept their terms on Hugging Face, then set
`HF_TOKEN` when downloading with `python -m imagehub.pretrained`. Flux.1-Lite is ungated. The
schnell weights use Apache-2.0; the dev, Krea, Lite and Kontext weights use the FLUX.1-dev
Non-Commercial License.

The `flux2-klein-9b` repository is gated. Accept its terms on Hugging Face, then set `HF_TOKEN`
when downloading with `python -m imagehub.pretrained`.

The HunyuanDiT weights use the Tencent Hunyuan Community License, which excludes use in the
European Union, the United Kingdom and South Korea and requires a separate license above 100 million
monthly active users; see the license.

The `krea2-turbo` repository is gated. Accept its terms on Hugging Face, then set `HF_TOKEN` when
downloading with `python -m imagehub.pretrained`. The weights use the Krea 2 Community License;
commercial use is conditional below USD 1 million annual revenue and requires an enterprise license
at or above that threshold.

`krea2-turbo` is the distilled inference checkpoint. Krea 2 LoRA training uses the separate,
gated `krea/Krea-2-Raw` checkpoint at `6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603`; accept the
same license before downloading it. Raw-trained adapters run on Turbo and remain bound by the Krea
2 Community License.

The `ideogram4-fp8` repository is gated. Accept its terms on Hugging Face, then set `HF_TOKEN` when
downloading with `python -m imagehub.pretrained`. The weights use the Ideogram 4 Non-Commercial
License; see the license.

Every official Ideogram 4 checkpoint is quantized, so `ideogram4-fp8` remains inference-only.

Kandinsky 5 publishes bfloat16 transformer and text-encoder weights; the catalog lists them as
released and the native loaders upcast them to float32 for parity testing.

Kolors publishes float16 weights only (`*.fp16.safetensors`), so its file list uses those names. Parity is
measured on their float32 upcast, which is also what diffusers computes with
`from_pretrained(..., variant="fp16", torch_dtype=torch.float32)`.

The `hidream-i1-fast`, `hidream-i1-dev` and `hidream-i1-full` entries combine their respective
HiDream-I1 repositories (MIT) with `meta-llama/Llama-3.1-8B-Instruct` at revision `0e9e39f`, placed in the
snapshot's `text_encoder_4/` subfolder together with its tokenizer. The Llama repository is gated:
accept its terms (Llama 3.1 Community License) on Hugging Face, then set `HF_TOKEN` when downloading
with `python -m imagehub.pretrained`. The HiDream repository does not ship that encoder, and the
pipeline loads it separately.

The `stable-cascade` and `stable-cascade-lite` weights use the Stable Cascade Non-Commercial
Community License (`stable-cascade-nc-community`), which prohibits commercial use; see the license
before downloading or using the weights. Both models share one decoder repository and one prior
repository and differ only in the stage C and stage B files they select (`prior` and `decoder`, or
`prior_lite` and `decoder_lite`).
