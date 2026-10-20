# Third-party architecture and license notices

imagehub implements native PyTorch inference for the model families listed below. These are new
implementations written with reference to upstream architecture definitions and checkpoint layouts;
they do not invoke the upstream inference packages at runtime. Layer structure, parameter names,
tokenizer rules and sampling order follow those projects so that published weights load unchanged
and outputs can be compared with them.

| Project | Attribution | Relevant imagehub components | License |
| --- | --- | --- | --- |
| [diffusers](https://github.com/huggingface/diffusers) v0.40.0 (plus the Qwen-Image 2.1 pipeline from commit `7997e4e`) | The HuggingFace Team and contributors | U-Nets and diffusion transformers of the supported families including PRX, prior and Würstchen/Cascade networks, KL/VQ/Paella/DC autoencoders, DDPM/DDIM/PNDM/Euler/Euler-ancestral/DPM-Solver/DEIS/EDM/flow-matching/LCM/UnCLIP/aMUSEd schedulers, classifier-free guidance, pipeline sampling order and checkpoint layout | [Apache 2.0](licenses/Diffusers-Apache-2.0.txt) |
| [transformers](https://github.com/huggingface/transformers) v5.18.0 | The HuggingFace Team, OpenAI and contributors | CLIP, T5/UMT5, T5Gemma, Gemma-2, Llama, Qwen, Qwen2.5-VL/Qwen3-VL, GLM, ChatGLM, BERT and Janus encoders; CLIP, Qwen, Gemma, ChatGLM, BERT, T5 and DeepSeek tokenizers | [Apache 2.0](licenses/Transformers-Apache-2.0.txt) |
| [CLIP](https://github.com/openai/CLIP) | 2021 OpenAI | Byte-level BPE vocabulary conventions | [MIT](licenses/OpenAI-CLIP-MIT.txt) |
| [SSD-1B](https://github.com/segmind/SSD-1B) at `2bfc35f` | Segmind, Hugging Face and contributors | Segmind-Vega published architecture, checkpoint configuration and SDXL loading recipe | [Apache 2.0](licenses/Segmind-SSD-1B-Apache-2.0.txt) |
| [sdxl-koala](https://github.com/youngwanLEE/sdxl-koala) at `df9dac7` | Electronics and Telecommunications Research Institute and contributors | KOALA and KOALA-Lightning published U-Net layouts, checkpoint configurations and SDXL loading recipes | [dual license](licenses/ETRI-KOALA-Dual-License.txt) |
| [distill-sd](https://github.com/segmind/distill-sd) at `c1e97a7` | Segmind and contributors | Small-SD and Tiny-SD pruned U-Net layouts, checkpoint configurations and loading recipe | [CreativeML OpenRAIL-M](licenses/Segmind-distill-sd-CreativeML-OpenRAIL-M.txt) |
| [BK-SDM](https://github.com/Nota-NetsPresso/BK-SDM) at `e13670c` | Nota Inc. and contributors | BK-SDM and BK-SDM v2 published U-Net pruning layouts and checkpoint configurations | [CreativeML OpenRAIL-M](licenses/BK-SDM-CreativeML-OpenRAIL-M.txt) |
| [LlamaGen](https://github.com/FoundationVision/LlamaGen) at `ce98ec4` | FoundationVision and contributors | GPT-XL caption embedder and autoregressive sampler, VQ-16 image tokenizer, 2D RoPE layout and KV-cache generation order | [MIT](licenses/LlamaGen-MIT.txt) |
| [HART](https://github.com/mit-han-lab/hart) at `e28a41f` | MIT HAN Lab and contributors | Multi-scale autoregressive transformer, hybrid tokenizer decoder, residual diffusion head and sampling order | [MIT](licenses/HART-MIT.txt) |
| [PixelDiT](https://github.com/NVlabs/PixelDiT) at `f7d7c35` | NVIDIA Corporation and contributors | Dual-level pixel diffusion transformer, pixel-wise modulation and FLOW DPM-Solver sampling order | [NVIDIA NSCLv1](licenses/PixelDiT-NSCLv1.txt) |
| [ideogram4](https://github.com/ideogram-oss/ideogram4) at `990fe1c` | Ideogram AI and contributors | Ideogram 4 transformer and text-model hidden-state selection, fp8 linear dequantization, sampler presets and prompt format | [Apache 2.0](licenses/Ideogram4-Apache-2.0.txt) |
| [HiDream-O1-Image](https://github.com/HiDream-ai/HiDream-O1-Image) at `2c2d29f` | HiDream.ai and contributors | Pixel patch projections, timestep conditioning, mixed causal/generation attention mask and sampling order | [MIT](licenses/HiDream-O1-MIT.txt) |
| [Meissonic](https://github.com/viiika/Meissonic) at `a9de06c` | The MeissonFlow Team and contributors | Masked-token transformer, rotary position layout, micro-conditioning and sampling order | [Apache 2.0](licenses/Meissonic-Apache-2.0.txt) |

Model weights are not distributed with imagehub; each checkpoint is downloaded by the user from the
repository listed in the [model catalog](docs/catalog.md), and its weight license — recorded there —
binds whoever downloads it. Several catalog licenses are non-commercial (`sai-nc-community`,
`CC-BY-NC-4.0`, `FLUX.1-dev Non-Commercial License`, `FLUX Non-Commercial License`,
`Ideogram 4 Non-Commercial License`, `NVIDIA NSCLv1` and `stable-cascade-nc-community`); read the license before
downloading.
