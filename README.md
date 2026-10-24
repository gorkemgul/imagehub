<h2 align="center">Open-weight text-to-image architectures in native PyTorch.</h2>

imagehub provides one API for open-weight text-to-image architectures using native PyTorch reimplementations; PyTorch is the only required runtime dependency. CPU float32 outputs are bit-identical to the reference implementations; see [validation](docs/validation.md). It supports Python 3.11 and later.

## Install

```bash
pip install imagehub            # torch only
pip install "imagehub[image]"   # adds Pillow for Image.to_pil()
```

## Quickstart

```bash
python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5
```

```python
from imagehub import pipeline

with pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5") as pipe:
    result = pipe("a photograph of an astronaut riding a horse", seed=42)
result.images[0].save("astronaut.png")
print(result.metadata)
```

Most models accept text alone. Some also accept an optional `image=` input for image-conditioned generation or editing.

```python
from PIL import Image
with pipeline("t2i", backend="flux1-kontext-dev", model="checkpoints/flux1-kontext-dev") as edit:
    result = edit("Add a red hat to the scene", image=Image.open("photo.png"), seed=0)
result.images[0].save("edited.png")
```

## Models

The [model catalog](docs/catalog.md) lists every supported checkpoint with its source, pinned revision, and license.

<details>
<summary>All model families</summary>

| Family | Checkpoints | Task | Input | Validation |
| --- | --- | --- | --- | --- |
| Latent Diffusion Model | `ldm-text2im-large-256` | `t2i` | T | CPU fp32 parity |
| Stable Diffusion | `sd-v1-1`, `sd-v1-2`, `sd-v1-3`, `sd-v1-4`, `sd-v1-5`, `dreamshaper-8`, `segmind-small-sd`, `segmind-tiny-sd`, `bk-sdm-base`, `bk-sdm-small`, `bk-sdm-tiny`, `bk-sdm-v2-base`, `bk-sdm-v2-small`, `bk-sdm-v2-tiny`, `sd-v2-1`, `sd-turbo`, `sdxs-512-0.9`, `lcm-dreamshaper-v7` | `t2i` | T | CPU fp32 parity (`segmind-tiny-sd` on the upcast float16 weights) |
| Stable Diffusion XL | `sdxl-base-1.0`, `sdxl-lightning-2step`, `sdxl-lightning-4step`, `sdxl-lightning-8step`, `hyper-sdxl-1step`, `dmd2-sdxl-4step`, `dmd2-sdxl-1step`, `sdxl-turbo`, `ssd-1b`, `segmind-vega`, `koala-700m`, `koala-1b`, `koala-lightning-700m`, `koala-lightning-1b`, `playground-v2`, `playground-v2.5`, `juggernaut-xl-v9`, `animagine-xl-4.0`, `dreamshaper-xl-1.0` | `t2i` | T | CPU fp32 parity (Juggernaut and Animagine on upcast float16 weights) |
| Stable Diffusion XL refiner | `sdxl-refiner-1.0` | `t2i` | T, I | CPU fp32 parity |
| Stable Diffusion 3 and 3.5 | `sd3-medium`, `sd3.5-medium`, `sd3.5-large`, `sd3.5-large-turbo` | `t2i` | T | CPU fp32 parity |
| DreamLite | `dreamlite-base`, `dreamlite-mobile` | `t2i` | T, I | CPU fp32 parity on the upcast bfloat16 weights |
| JoyAI-Image | `joyai-image-edit`, `joyai-image-edit-plus` | `t2i` | T, I | CPU fp32 parity on the upcast bfloat16 weights |
| Cosmos-Predict2 | `cosmos-predict2-2b-t2i` | `t2i` | T | Exact random-weight and tiny-pipeline parity |
| FLUX.1 | `flux1-schnell`, `flux1-dev`, `flux1-krea-dev`, `flux-1-lite-8b` | `t2i` | T | CPU fp32 parity on the upcast bfloat16 weights |
| FLUX.1 Kontext | `flux1-kontext-dev` | `t2i` | T, I | CPU fp32 parity |
| FLUX.2 | `flux2-klein-4b`, `flux2-klein-9b` | `t2i` | T | CPU fp32 parity |
| HunyuanDiT | `hunyuan-dit-v1.1-distilled`, `hunyuan-dit-v1.2`, `hunyuan-dit-v1.2-distilled` | `t2i` | T | CPU fp32 parity |
| DeepFloyd IF | `deepfloyd-if-i-xl` | `t2i` | T | CPU fp32 parity |
| Kandinsky 2.1 | `kandinsky-2-1` | `t2i` | T | CPU fp32 parity |
| Kandinsky 2.2 | `kandinsky-2.2` | `t2i` | T | CPU fp32 parity |
| Kandinsky 3 | `kandinsky-3` | `t2i` | T | CPU fp32 parity on the upcast float16 weights |
| Kandinsky 5 | `kandinsky-5-t2i-lite` | `t2i` | T | CPU fp32 parity |
| Stable Cascade | `stable-cascade`, `stable-cascade-lite` | `t2i` | T | CPU fp32 parity |
| Würstchen v2 | `wuerstchen-v2` | `t2i` | T | CPU fp32 parity |
| PixArt | `pixart-sigma-xl-1024`, `pixart-sigma-xl-512`, `pixart-sigma-xl-256`, `pixart-sigma-xl-2k`, `pixart-alpha-xl-1024`, `pixart-alpha-xl-512`, `pixart-lcm-xl-1024` | `t2i` | T | CPU fp32 parity |
| Sana | `sana-1600m-1024`, `sana-600m-1024`, `sana-sprint-1.6b-1024`, `sana-1.5-4.8b-1024` | `t2i` | T | CPU fp32 parity |
| Kolors | `kolors` | `t2i` | T | CPU fp32 parity on the upcast float16 weights |
| Lumina-Image 2.0 | `lumina-image-2.0` | `t2i` | T | CPU fp32 parity |
| Lumina-Next | `lumina-next` | `t2i` | T | CPU fp32 parity |
| Lumina-DiMOO | `lumina-dimoo` | `t2i` | T | CPU fp32 parity on the upcast bfloat16 weights |
| OmniGen | `omnigen-v1` | `t2i` | T, I | CPU fp32 parity on the upcast bfloat16 transformer |
| Ovis-Image | `ovis-image-7b` | `t2i` | T | CPU fp32 parity on the upcast bfloat16 VAE weights |
| aMUSEd | `amused-256`, `amused-512` | `t2i` | T | CPU fp32 parity |
| Meissonic | `meissonic` | `t2i` | T | CPU fp32 parity |
| AuraFlow | `auraflow-v0.3` | `t2i` | T | CPU fp32 parity |
| PRX | `prx-1024-t2i-beta`, `prx-512-t2i-sft` | `t2i` | T | CPU fp32 parity on the upcast bfloat16 text encoder |
| Chroma | `chroma1-hd`, `chroma1-flash`, `chroma1-base` | `t2i` | T | CPU fp32 parity on the upcast bfloat16 weights |
| LlamaGen | `llamagen-t2i-xl-256`, `llamagen-t2i-xl-512` | `t2i` | T | CPU fp32 parity |
| HART | `hart` | `t2i` | T | CPU fp32 parity on the upcast bfloat16 text encoder |
| Show-o | `show-o-512` | `t2i` | T | CPU fp32 parity |
| PixelDiT | `pixeldit` | `t2i` | T | CPU fp32 parity on the upcast bfloat16 text encoder |
| Krea 2 | `krea2-turbo` | `t2i` | T | CPU fp32 parity |
| ERNIE-Image | `ernie-image` | `t2i` | T | CPU fp32 parity without the prompt enhancer |
| LongCat-Image | `longcat-image` | `t2i` | T | CPU fp32 parity |
| CogView3-Plus | `cogview3-plus-3b` | `t2i` | T | CPU fp32 parity on the upcast bfloat16 transformer and VAE |
| CogView4 | `cogview4-6b` | `t2i` | T | CPU fp32 parity |
| GLM-Image | `glm-image` | `t2i` | T | CPU fp32 parity |
| Ideogram 4 | `ideogram4-fp8` | `t2i` | T | CPU fp32 parity with the official implementation |
| Qwen-Image | `qwen-image`, `qwen-image-2.1` | `t2i` | T | CPU fp32 parity on the upcast bfloat16 weights |
| Z-Image | `z-image`, `z-image-turbo` | `t2i` | T | CPU fp32 parity |
| Janus-Pro | `janus-pro-1b`, `janus-pro-7b` | `t2i` | T | CPU fp32 parity |
| HiDream-I1 | `hidream-i1-dev`, `hidream-i1-fast`, `hidream-i1-full` | `t2i` | T | CPU fp32 parity on the upcast bfloat16 and float16 weights |
| HiDream-O1 | `hidream-o1-image` | `t2i` | T | CPU fp32 parity |

</details>

## Validation

Every listed checkpoint is validated against its reference implementation, with CPU float32 image outputs required to match bit-for-bit. The [validation documentation](docs/validation.md) describes the coverage and recorded results.

## Documentation

[Model catalog](docs/catalog.md) · [Model architecture notes](docs/models.md) · [Validation](docs/validation.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

## Development

```bash
uv venv -p 3.13 .venv && uv pip install --python .venv -e ".[dev,image]"
.venv/bin/python -m pytest
```

Reference comparisons are optional; install `.[dev,image,reference]` and run `IMAGEHUB_REFERENCE_TESTS=1 .venv/bin/python -m pytest tests/reference` with local checkpoint paths configured.

## License

The library is licensed under Apache-2.0. Each checkpoint has its own license; see the [model catalog](docs/catalog.md).
