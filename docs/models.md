# Stable Diffusion 1.x

Catalog: [`stable-diffusion-1`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `sd-v1-1` … `sd-v1-5` · Task: `t2i`

**Architecture.** An f=8 KL autoencoder (512×512×3 image ↔ 64×64×4 latent), a 860M-parameter U-Net
conditioned through cross-attention, and a frozen CLIP ViT-L/14 text encoder (last layer, unpooled
output). The scheduler is the PNDM published with the checkpoints (`skip_prk_steps=True`, i.e. PLMS):
50 steps perform 51 U-Net evaluations.

**Usage.**

```bash
python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42, negative_prompt="blurry")
    result.images[0].save("cat.png")
```

**Settings.** `steps` (50), `guidance` (7.5; 1 and below turns guidance off), `width`/`height`
(512, multiples of 8), `negative_prompt`, `num_images` (1), `seed`, `latents`. Unrecognized settings
raise an error.

**Seed compatibility.** On CPU, the same seed gives the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS has a smoke
test (at most 1 pixel of difference from CPU, PSNR 72.8 dB); CUDA has not been tested yet. There is
no claim of exact equality on these devices.

**Out of scope.** There is no safety checker or invisible watermark. img2img, other schedulers and
fp16 parity are not in this release.

**License.** The weights are licensed under CreativeML OpenRAIL-M; the license's use restrictions
bind whoever downloads and uses the weights. imagehub does not distribute weights.


# Latent Consistency Model (LCM)

Catalog: [`lcm`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `lcm-dreamshaper-v7` · Task: `t2i`

**Architecture.** A Stable Diffusion 1.x architecture distilled with latent consistency. Guidance
enters the U-Net as an embedding instead of classifier-free guidance. The checkpoint uses an LCM
scheduler and has a native image size of 768×768.

**Usage.**

```bash
python -m imagehub.pretrained lcm-dreamshaper-v7 checkpoints/lcm-dreamshaper-v7
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="lcm-dreamshaper-v7", model="checkpoints/lcm-dreamshaper-v7", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** `steps` defaults to 4, `guidance` defaults to 8.0 and `width`/`height` default to
768. `negative_prompt`, `num_images`, `seed` and `latents` are also supported. Unrecognized
settings raise an error.

**Seed compatibility.** On CPU, the same seed gives the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** There is no safety checker or invisible watermark. img2img, other schedulers and
fp16 parity are not in this release.

**License.** The weights are licensed under MIT. imagehub does not distribute weights.


# SD-Turbo

Catalog: [`sdxl-turbo`](https://github.com/gorkemgul/awesome-text-to-image-architectures) (variant
"SD Turbo") · Models: `sd-turbo` · Task: `t2i`

**Architecture.** A distilled Stable Diffusion 2.1-based model with a frozen OpenCLIP-H text
encoder, U-Net and f=8 KL autoencoder. The checkpoint publishes an Euler scheduler with
`timestep_spacing="trailing"`. Its native image size is 512×512.

**Usage.**

```bash
python -m imagehub.pretrained sd-turbo checkpoints/sd-turbo
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="sd-turbo", model="checkpoints/sd-turbo", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** `steps` defaults to 1, `guidance` defaults to 0.0 and `width`/`height` default to
512. `negative_prompt`, `num_images`, `seed` and `latents` are also supported. Unrecognized
settings raise an error.

**Seed compatibility.** On CPU, the same seed gives the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** There is no safety checker or invisible watermark. img2img, other schedulers and
fp16 parity are not in this release.

**License.** The weights use the `sai-nc-community` license; see the license for its restrictions.
imagehub does not distribute weights.


# Stable Diffusion XL

Catalog: [`sdxl`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `sdxl-base-1.0`, `sdxl-lightning-2step`, `sdxl-lightning-4step`,
`sdxl-lightning-8step`, `hyper-sdxl-1step`, `dmd2-sdxl-4step`, `dmd2-sdxl-1step` · Task: `t2i`

**Architecture.** Two frozen text encoders: CLIP ViT-L/14 (penultimate layer output) and OpenCLIP
bigG (penultimate layer output and pooled output). The concatenated outputs condition a
2.6B-parameter U-Net with `text_time` micro-conditioning: the original size, the crop position and
the target size of the image, encoded as a 6-element vector per sample. The VAE is an f=8 KL
autoencoder with a scaling factor of 0.13025 (1024×1024×3 image ↔ 128×128×4 latent). The
scheduler is the Euler published with the checkpoint (`timestep_spacing="leading"`,
`steps_offset=1`): 50 steps perform 50 U-Net evaluations.

**SDXL-Lightning.** These three models place a distilled U-Net on the SDXL base. Their defaults are
respectively 2, 4 and 8 steps, guidance 0.0, and Euler scheduling with
`timestep_spacing="trailing"`.

**Hyper-SD.** `hyper-sdxl-1step` places the published Hyper-SDXL U-Net on the SDXL base. It uses an
LCM scheduler built from the SDXL scheduler configuration and defaults to 1 step, guidance 0.0,
explicit timesteps `[800]` and 1024×1024.

**DMD2.** `dmd2-sdxl-4step` and `dmd2-sdxl-1step` place the published full float32 DMD2 U-Net on
the SDXL base. Both use an LCM scheduler built from the SDXL scheduler configuration, guidance
0.0, and 1024×1024. Their defaults are respectively 4 steps with explicit timesteps
`[999, 749, 499, 249]` and 1 step with `[399]`.

**Usage.**

```bash
python -m imagehub.pretrained sdxl-base-1.0 checkpoints/sdxl-base-1.0
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="sdxl-base-1.0", model="checkpoints/sdxl-base-1.0", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42, negative_prompt="blurry")
    result.images[0].save("cat.png")
```

Hyper-SD follows the model card's explicit-timestep recipe:

```bash
python -m imagehub.pretrained hyper-sdxl-1step checkpoints/hyper-sdxl-1step
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="hyper-sdxl-1step", model="checkpoints/hyper-sdxl-1step") as pipe:
    result = pipe("a cat, oil painting", steps=1, guidance=0.0, timesteps=[800], seed=42)
    result.images[0].save("cat.png")
```

DMD2 follows the model card's explicit-timestep recipes:

```bash
python -m imagehub.pretrained dmd2-sdxl-4step checkpoints/dmd2-sdxl-4step
python -m imagehub.pretrained dmd2-sdxl-1step checkpoints/dmd2-sdxl-1step
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="dmd2-sdxl-4step", model="checkpoints/dmd2-sdxl-4step") as pipe:
    result = pipe(
        "a cat, oil painting",
        steps=4,
        guidance=0.0,
        timesteps=[999, 749, 499, 249],
        seed=42,
    )
    result.images[0].save("cat.png")
```

**Settings.** `steps` (50, at most 1000), `guidance` (5.0; 1 and below turns guidance off),
`width`/`height` (the U-Net config's native size, 1024 here, and multiples of 8), `negative_prompt`
(None; with an empty negative prompt, zero embeddings are used), `num_images` (1), `seed`,
`latents`, and the micro-conditioning inputs `original_size`, `crops_coords_top_left` and
`target_size` (default to the generated size and `(0, 0)`). The SDXL backend also supports the LCM
scheduler and an explicit `timesteps` sequence; supplying `timesteps` sets the step count.
Unrecognized settings raise an error.

**Seed compatibility.** On CPU, the same seed gives the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). As with SD 1.x,
MPS and CUDA runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** There is no safety checker, and the optional image watermark is off. The SDXL
refiner model, img2img, other schedulers and fp16 parity are not in this release.

**License.** The SDXL base and SDXL-Lightning weights are licensed under CreativeML Open RAIL++-M;
the license's use restrictions bind whoever downloads and uses the weights. The Hyper-SD model
card has no license metadata, and the repository's license file covers only the FLUX-related files,
so the Hyper-SDXL weights are recorded as `Unspecified`. The DMD2 weights are licensed under
CC-BY-NC-4.0, which prohibits commercial use. imagehub does not distribute weights.


# SDXL Turbo

Catalog: [`sdxl-turbo`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `sdxl-turbo` · Task: `t2i`

**Architecture.** A distilled SDXL model with the same two frozen text encoders, `text_time`
micro-conditioning and f=8 KL autoencoder design as SDXL. Its native image size is 512×512. The
checkpoint publishes an Euler ancestral scheduler with `timestep_spacing="trailing"` and
`steps_offset=1`; its ancestral noise is drawn from the same seeded CPU generator as the initial
noise.

**Usage.**

```bash
python -m imagehub.pretrained sdxl-turbo checkpoints/sdxl-turbo
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="sdxl-turbo", model="checkpoints/sdxl-turbo", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** `steps` defaults to 1, `guidance` defaults to 0.0 and `width`/`height` default to
512. Other supported settings match SDXL. Other SDXL models retain defaults of 50 steps and 5.0
guidance and take their default size from the U-Net config. Unrecognized settings raise an error.

**Seed compatibility.** On CPU, the same seed gives the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)), including the
ancestral noise at every scheduler step. MPS and CUDA runs are smoke tests only; there is no claim
of exact equality on these devices.

**Out of scope.** There is no safety checker, and the optional image watermark is off. The SDXL
refiner model, img2img, other schedulers and fp16 parity are not in this release.

**License.** The weights use the `sai-nc-community` license and are limited to non-commercial use.
The license's use restrictions bind whoever downloads and uses the weights. imagehub does not
distribute weights.


# SSD-1B

Catalog: [`ssd-1b`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `ssd-1b` · Task: `t2i`

**Architecture.** A distilled SDXL model with the same frozen CLIP ViT-L/14 and OpenCLIP bigG text
encoders, f=8 KL autoencoder and Euler scheduler as SDXL. Its 1.3B-parameter U-Net uses per-block
transformer depths and a mid block without attention while retaining SDXL's `text_time`
micro-conditioning.

**Usage.**

```bash
python -m imagehub.pretrained ssd-1b checkpoints/ssd-1b
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="ssd-1b", model="checkpoints/ssd-1b", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42, negative_prompt="blurry")
    result.images[0].save("cat.png")
```

**Settings.** The defaults and supported settings are the same as for SDXL: `steps` (50, at most
1000), `guidance` (5.0), `width`/`height` (the U-Net config's native size, 1024 here, and multiples
of 8), `negative_prompt` (None), `num_images` (1), `seed`, `latents`, `original_size`,
`crops_coords_top_left` and `target_size`. Unrecognized settings raise an error.

**Seed compatibility.** On CPU, the same seed gives the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** There is no safety checker, and the optional image watermark is off. The SDXL
refiner model, img2img, other schedulers and fp16 parity are not in this release.

**License.** The weights are licensed under Apache-2.0. imagehub does not distribute weights.


# Stable Diffusion 3 and 3.5

Catalog: [`stable-diffusion-3`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `sd3-medium`, `sd3.5-medium`, `sd3.5-large`, `sd3.5-large-turbo` · Task: `t2i`

**Architecture.** Three frozen text encoders—CLIP-L, CLIP-G and T5-XXL—produce a joint sequence
of 77 CLIP tokens and 256 T5 tokens plus a 2048-wide pooled CLIP embedding. The SD 3.0 medium
checkpoint uses a 24-block MMDiT without q/k normalization. The SD 3.5 medium checkpoint uses
MMDiT-X, with dual attention in its first 13 blocks; the large and large-turbo checkpoints use the
same 38-block MMDiT architecture. Large-turbo is distilled for 4 steps without guidance. A
16-channel VAE with a shift factor decodes the latents. The flow-match Euler scheduler uses shift 3.

**Usage.** All four repositories are gated. Accept the repository terms and provide `HF_TOKEN` when
downloading:

```bash
HF_TOKEN=... python -m imagehub.pretrained sd3-medium checkpoints/sd3-medium
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="sd3-medium", model="checkpoints/sd3-medium", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42, negative_prompt="blurry")
    result.images[0].save("cat.png")
```

**Settings.** SD 3.0 medium defaults to 28 steps and guidance 7.0; SD 3.5 medium defaults to 40
steps and guidance 4.5; large defaults to 28 steps and guidance 3.5; large-turbo defaults to 4
steps and guidance 0.0. All four default to 1024×1024, and `width`/`height` must be multiples of
16. The other supported options are `negative_prompt`, `num_images`, `seed`, `latents` and
`max_sequence_length` (256). Unrecognized settings raise an error.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** There is no safety checker or invisible watermark. Img2img, other schedulers and
fp16/bf16 parity are not in this release.

**License.** The weights use the Stability AI Community License; see the license. imagehub does
not distribute weights.


# FLUX.1

Catalog: [`flux-1`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `flux1-schnell`, `flux1-dev` · Task: `t2i`

**Architecture.** CLIP-L supplies a pooled embedding and T5-XXL supplies sequence embeddings to a
12B rectified-flow transformer with 19 double-stream blocks, 38 single-stream blocks, 24 attention
heads of width 128 and 3-axis RoPE. The dev checkpoint adds a guidance embedding. A 16-channel VAE
with a shift factor decodes latents packed in 2×2 patches. The flow-match Euler scheduler uses a
static shift of 1.0 for schnell and dynamic shifting for dev. The native image size is 1024×1024.

**Usage.** Both repositories are gated. Accept the repository terms and provide `HF_TOKEN` when
downloading:

```bash
HF_TOKEN=... python -m imagehub.pretrained flux1-schnell checkpoints/flux1-schnell
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="flux1-schnell", model="checkpoints/flux1-schnell", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** Schnell defaults to 4 steps, guidance 0.0 and a maximum sequence length of 256. Dev
defaults to 50 steps, guidance 3.5 and a maximum sequence length of 512. Both default to 1024×1024,
and `width`/`height` must be multiples of 16. The other supported options are `num_images`, `seed`,
`latents`, `negative_prompt` and `true_cfg_scale`. Unrecognized settings raise an error.

**Validation.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** Kontext, Fill, Redux, Depth and Canny variants, ControlNet, LoRA and fp16/bf16
parity are not supported.

**License.** The schnell weights use the Apache-2.0 license. The dev weights use the FLUX.1-dev
Non-Commercial License; see the license. imagehub does not distribute weights.


# FLUX.2 klein

Catalog: [`flux-2`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `flux2-klein-4b`, `flux2-klein-9b` · Task: `t2i`

**Architecture.** The 4B checkpoint uses a Qwen3-4B text encoder, and its FLUX.2 transformer uses 5
double-stream blocks followed by 20 single-stream blocks. The 9B checkpoint uses a Qwen3-8B text
encoder, and its FLUX.2 transformer uses 8 double-stream blocks followed by 24 single-stream blocks.
Both text encoders supply hidden states from layers 9, 18 and 27, which are stacked for prompt
conditioning. The FLUX.2 VAE has 32 latent channels; its latents are patchified 2×2 and normalized
with batch-norm statistics. The flow-match Euler scheduler uses an empirical shift. The native image
size is 1024×1024.

**Usage.**

```bash
python -m imagehub.pretrained flux2-klein-4b checkpoints/flux2-klein-4b
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="flux2-klein-4b", model="checkpoints/flux2-klein-4b", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** `steps` defaults to 4, `guidance` defaults to 1.0 and `width`/`height` default to
1024. The other supported options are `num_images`, `seed` and `latents`. Unrecognized settings
raise an error.

**Seed compatibility.** On CPU, the same seed and settings give the same image as the pinned
diffusers reference with `generator=torch.Generator("cpu").manual_seed(seed)`
([validation](validation.md)). MPS and CUDA runs are smoke tests only; there is no claim of exact
equality on these devices.

**Out of scope.** Image editing, condition images and fp16/bf16 parity are not supported.

**License.** The 4B weights use the Apache-2.0 license. The 9B weights use the
FLUX Non-Commercial License; see the license. imagehub does not distribute weights.


# HunyuanDiT

Catalog: [`hunyuan-dit`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `hunyuan-dit-v1.1-distilled` · Task: `t2i`

**Architecture.** Bilingual Chinese/English BERT-large and mT5-XL text encoders condition a 1.5B
diffusion transformer with 40 blocks, width 1408, 16 attention heads of width 88, long skip
connections and 2D RoPE. It also uses timestep, image-size, style and pooled-text conditioning and
predicts a learned sigma. A 4-channel VAE with scaling factor 0.13025 decodes the latents. The DDPM
scheduler uses v-prediction, and requested resolutions are binned to supported shapes.

**Usage.**

```bash
python -m imagehub.pretrained hunyuan-dit-v1.1-distilled checkpoints/hunyuan-dit-v1.1-distilled
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="hunyuan-dit-v1.1-distilled", model="checkpoints/hunyuan-dit-v1.1-distilled", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** `steps` defaults to 25, `guidance` defaults to 5.0 and `width`/`height` default to
1024. The other supported options are `seed`, `latents`, `negative_prompt`, `guidance_rescale` and
`use_resolution_binning`. Unrecognized settings raise an error.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** Other HunyuanDiT releases (v1.0, v1.2 and non-distilled), ControlNet, PAG, LoRA
and fp16/bf16 parity are not supported.

**License.** The weights use the Tencent Hunyuan Community License, which excludes use in the
European Union, the United Kingdom and South Korea; see the license. imagehub does not distribute
weights.


# Kandinsky 2.2

Catalog: [`kandinsky-2`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `kandinsky-2.2` · Task: `t2i`

**Architecture.** A CLIP text model with projection conditions a PriorTransformer, while a CLIP
vision model with projection supplies the zero image embedding for the negative branch. The prior
uses an UnCLIP scheduler with sample prediction and `fixed_small_log` variance. Its image embedding
conditions a U-Net with `ResnetDownsampleBlock2D`, `SimpleCrossAttn*` blocks and `scale_shift` time
modulation. A MoVQ VQ model with spatially conditioned normalization decodes the latents. The
decoder uses a DDPM scheduler with linear betas and clipping.

**Usage.**

```bash
python -m imagehub.pretrained kandinsky-2.2 checkpoints/kandinsky-2.2
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="kandinsky-2.2", model="checkpoints/kandinsky-2.2", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** The prior defaults to 25 steps and guidance 4.0; the decoder defaults to 100 steps,
guidance 4.0 and 512×512 output. The supported options are `steps`, `guidance`, `prior_steps`,
`prior_guidance` and `negative_prompt`. Unrecognized settings raise an error.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** Img2img, inpainting, ControlNet, prior interpolation and fp16/bf16 parity are not
supported.

**License.** The decoder and prior weights use the Apache-2.0 license. imagehub does not distribute
weights.


# PixArt

Catalog: [`pixart-sigma`](https://github.com/gorkemgul/awesome-text-to-image-architectures),
[`pixart-alpha`](https://github.com/gorkemgul/awesome-text-to-image-architectures) · Models:
`pixart-sigma-xl-1024`, `pixart-alpha-xl-1024` · Task: `t2i`

**Architecture.** A T5-v1.1-XXL text encoder supplies up to 300 tokens for Sigma and 120 for Alpha
to a 0.6B diffusion transformer with 28 blocks, width 1152 and 16 attention heads of width 72. The
transformer uses shared `ada_norm_single` timestep modulation, caption projection, a 2D sine-cosine
position embedding and learned sigma prediction. Sigma uses interpolation scale 2 and the SDXL VAE
with scaling factor 0.13025. Alpha adds resolution and aspect-ratio embeddings and uses the SD VAE
with scaling factor 0.18215. Both use the order-2 midpoint DPM-Solver++ multistep scheduler and bin
requested resolutions to their supported aspect-ratio tables.

**Usage.**

```bash
python -m imagehub.pretrained pixart-sigma-xl-1024 checkpoints/pixart-sigma-xl-1024
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="pixart-sigma-xl-1024", model="checkpoints/pixart-sigma-xl-1024", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42, negative_prompt="blurry")
    result.images[0].save("cat.png")
```

**Settings.** Both models default to 20 steps, guidance 4.5 and 1024×1024. Sigma defaults to a
maximum sequence length of 300 and Alpha to 120. The other supported options are `seed`, `latents`,
`negative_prompt`, `use_resolution_binning` and `max_sequence_length`. Unrecognized settings raise
an error.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** The 256×256, 512×512 and 2K releases, PixArt-LCM, PixArt-DMD, ControlNet,
fp16/bf16 parity and caption cleaning are not supported. Caption cleaning requires `bs4` and `ftfy`;
parity is against `clean_caption=False`, which diffusers uses when those packages are unavailable.

**License.** Both models use the CreativeML-OpenRAIL++-M license. imagehub does not distribute
weights.


# Krea 2

Catalog: [`krea-2`](https://github.com/gorkemgul/awesome-text-to-image-architectures) · Models:
`krea2-turbo` · Task: `t2i`

**Architecture.** A Qwen3-VL text model with 36 layers and width 2560 supplies the hidden states
of 12 selected layers, stacked for a roughly 13B diffusion transformer with 28 blocks, 48 query
heads and 12 key-value heads of width 128. The transformer uses 3-axis RoPE plus layerwise and
refiner text blocks. A Qwen-Image VAE uses 16 latent channels, causal 3D convolutions and
per-channel latent mean and standard deviation. The flow-match Euler scheduler uses exponential
dynamic shifting. The native image size is 1024×1024.

**Usage.** The repository is gated. Accept its terms and provide `HF_TOKEN` when downloading:

```bash
HF_TOKEN=... python -m imagehub.pretrained krea2-turbo checkpoints/krea2-turbo
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="krea2-turbo", model="checkpoints/krea2-turbo", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** The model defaults to 8 steps, guidance 0.0 and 1024×1024; `width` and `height` must
be multiples of 16. The other supported options are `negative_prompt`, `seed`, `latents` and
`max_sequence_length`. Unrecognized settings raise an error.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices. The published
bfloat16 weights are upcast for float32 parity.

**Out of scope.** Krea 2 Raw, the single-file `turbo.safetensors` in the official codebase layout,
LoRA and fp16/bf16 parity are not supported.

**License.** The weights use the Krea 2 Community License; see the license. imagehub does not
distribute weights.


# LongCat-Image

Catalog: [`longcat-image`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `longcat-image` · Task: `t2i`

**Architecture.** A Qwen2.5-VL 7B text model with 28 layers, width 3584, 28 query heads, 4
key-value heads and 16/24/24 multimodal RoPE sections first rewrites the prompt deterministically
with the published generation configuration and up to 512 new tokens, then encodes it. A FLUX-style
transformer has 10 double-stream blocks, 20 single-stream blocks and 24 attention heads of width
128, with no guidance embedding. The 16-channel FLUX VAE decodes the latents. The flow-match Euler
scheduler uses dynamic shifting, and classifier-free guidance is renormalized by default.

**Usage.**

```bash
python -m imagehub.pretrained longcat-image checkpoints/longcat-image
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="longcat-image", model="checkpoints/longcat-image", device="mps") as pipe:
    result = pipe(
        "a cat, oil painting",
        seed=42,
        enable_prompt_rewrite=True,
        enable_cfg_renorm=True,
    )
    result.images[0].save("cat.png")
```

**Settings.** `steps` defaults to 50, `guidance` defaults to 4.5, `enable_prompt_rewrite` and
`enable_cfg_renorm` default to true, and `width`/`height` default to 1024. The other supported
options are `seed`, `latents`, `negative_prompt` and `cfg_renorm_min`. Unrecognized settings raise
an error.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** LongCat-Image-Edit and fp16/bf16 parity are not supported. The published weights
are bfloat16; validated parity is in float32 after upcasting.

**License.** The weights use the Apache-2.0 license. imagehub does not distribute weights.


# CogView4

Catalog: [`cogview4`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `cogview4-6b` · Task: `t2i`

**Architecture.** A GLM-4-9B text model with 40 layers, width 4096, 32 query heads, 2 key-value
heads and partial rotary factor 0.5 supplies its second-to-last hidden state. A 6B transformer uses
28 blocks, 32 attention heads of width 128, 2D RoPE, and timestep, original-size, target-size and
crop conditioning. A 16-channel VAE with scaling factor 1.0 and shift factor 0.0 decodes the
latents. The flow-match Euler scheduler uses linear dynamic shifting. The native image size is
1024×1024.

**Usage.**

```bash
python -m imagehub.pretrained cogview4-6b checkpoints/cogview4-6b
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="cogview4-6b", model="checkpoints/cogview4-6b", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** `steps` defaults to 50, `guidance` defaults to 5.0, `width`/`height` default to 1024
and `max_sequence_length` defaults to 1024. The other supported options are `seed`, `latents` and
`negative_prompt`; `width` and `height` must be multiples of 16. Unrecognized settings raise an
error.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** CogView4 control pipelines and fp16/bf16 parity are not supported. The published
weights are bfloat16; CPU parity was measured in float32 after upcasting.

**License.** The weights use the Apache-2.0 license. imagehub does not distribute weights.


# Ideogram 4

Catalog: [`ideogram-4`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `ideogram4-fp8` · Task: `t2i`

**Architecture.** A Qwen3-VL 8B text model uses fp8 linears and supplies 53,248-wide features
formed by stacking selected hidden states. Two separate 9.3B single-stream diffusion transformers
provide the conditional and unconditional classifier-free-guidance passes. Each transformer has 34
layers, 18 attention heads of width 256, adaLN and 24/20/20 multimodal RoPE. The official
32-channel autoencoder applies latent normalization. The official sampler supplies the
`V4_QUALITY_48` (default), `V4_DEFAULT_20` and `V4_TURBO_12` presets, including each preset's
per-step guidance schedule. The native image size is 1024×1024.

The published checkpoint contains only quantized weights: large linear weights are float8 e4m3
with per-row float32 scales. imagehub keeps those weights in fp8 and dequantizes each forward pass
as `weight.to(dtype) * weight_scale.to(dtype).unsqueeze(1)`, exactly as the official
implementation does. The result is bit-identical when the compute dtype is float32.

**Usage.** The repository is gated. Accept its terms and provide `HF_TOKEN` when downloading:

```bash
HF_TOKEN=... python -m imagehub.pretrained ideogram4-fp8 checkpoints/ideogram4-fp8
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="ideogram4-fp8", model="checkpoints/ideogram4-fp8", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42, preset="V4_QUALITY_48")
    result.images[0].save("cat.png")
```

**Settings.** `preset` defaults to `V4_QUALITY_48`, and `width`/`height` default to 1024 and must
be multiples of 16. `V4_DEFAULT_20` and `V4_TURBO_12` select the official 20-step and 12-step
sampler configurations. The other supported options are `steps`, `guidance`, `guidance_schedule`,
`seed` and `latents`. Unrecognized settings raise an error.

**Seed compatibility.** On CPU in float32, the same seed and settings give the same image as the
official `ideogram-oss/ideogram4` implementation at commit `990fe1c`; this validation is not
against diffusers ([validation](validation.md)). MPS and CUDA runs are smoke tests only; there is
no claim of exact equality on these devices.

**Out of scope.** The CUDA-only nf4 releases, magic prompt, safety moderation, LoRA and fp16/bf16
parity are not supported. Magic prompt and safety moderation call external paid APIs.

**License.** The weights use the Ideogram 4 Non-Commercial License; see the license. imagehub does
not distribute weights.


# Qwen-Image 2.1

Catalog: [`qwen-image-2`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `qwen-image-2.1` · Task: `t2i`

**Architecture.** A Qwen3-VL-8B text encoder renders the prompt through the pipeline's prompt
template and supplies the last decoder-layer output before the final norm. A 7B transformer uses
block-causal joint attention and reuses a text KV cache across sampling steps. A causal 3D VAE has
64 latent channels. The flow-match Euler scheduler uses dynamic shifting. The native image size is
1024×1024.

**Usage.**

```bash
python -m imagehub.pretrained qwen-image-2.1 checkpoints/qwen-image-2.1
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="qwen-image-2.1", model="checkpoints/qwen-image-2.1", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42, negative_prompt="blurry", true_cfg=4.0)
    result.images[0].save("cat.png")
```

**Settings.** `steps` defaults to 40, `true_cfg` defaults to 1.0 and `width`/`height` default to
1024 and must be multiples of 32. The other supported options are `negative_prompt`, `num_images`,
`seed` and `latents`. Unrecognized settings raise an error.

**Seed compatibility.** On CPU, the same seed and settings give the same image as the pinned
diffusers reference with `generator=torch.Generator("cpu").manual_seed(seed)`
([validation](validation.md)). MPS and CUDA runs are smoke tests only; there is no claim of exact
equality on these devices.

**Out of scope.** Image editing, condition images, the Qwen3-VL vision encoder and fp16/bf16 parity
are not supported.

**License.** The weights use the Qwen Research License and are for research use; see the license.
imagehub does not distribute weights.


# Janus-Pro

Catalog: [`janus-pro`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `janus-pro-1b`, `janus-pro-7b` · Task: `t2i`

**Architecture.** The DeepSeek LLM tokenizer renders the prompt through a chat template ending in
the image-start token. The 1B checkpoint uses a 24-layer Llama with hidden size 2048; the 7B
checkpoint uses 30 layers with hidden size 4096. It autoregressively generates 576 image tokens,
applying classifier-free guidance over the logits. Model-specific special-token ids, which differ
between the two checkpoints, are read from each checkpoint's config. A LlamaGen VQ-16 decoder
produces the fixed 384×384 image.

**Usage.**

```bash
python -m imagehub.pretrained janus-pro-1b checkpoints/janus-pro-1b
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="janus-pro-1b", model="checkpoints/janus-pro-1b", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** The supported options are `guidance` (5), `temperature` (1.0), `do_sample` (True),
`num_images` and `seed`. The output size is fixed at 384×384. Unrecognized settings raise an error.

**Seed compatibility.** On CPU, seed 42 produces the same image tokens and pixels as transformers
5.18.0 `JanusForConditionalGeneration.generate(generation_mode="image")`
([validation](validation.md)). MPS and CUDA runs are smoke tests only; there is no claim of exact
equality on these devices.

**Out of scope.** Image understanding and variable output sizes are not supported.

**License.** The original Janus code is licensed under MIT. The weights are under the DeepSeek
Model License; see the license. imagehub does not distribute weights.


# Playground v2.5

Catalog: [`playground-v2`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `playground-v2.5` · Task: `t2i`

**Architecture.** An SDXL-family model with the same two frozen text encoders and `text_time`
micro-conditioning. Its EDM DPM-Solver++ scheduler is a second-order multistep scheduler with
Karras sigmas and EDM preconditioning. The f=8 KL autoencoder uses a scaling factor of 0.5, with
per-channel `latents_mean` and `latents_std` applied before decoding. Its native image size is
1024×1024.

**Usage.**

```bash
python -m imagehub.pretrained playground-v2.5 checkpoints/playground-v2.5
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="playground-v2.5", model="checkpoints/playground-v2.5", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** `steps` defaults to 50, `guidance` defaults to 3.0 and `width`/`height` default to
1024. Other supported settings match SDXL. Unrecognized settings raise an error.

**Seed compatibility.** On CPU, the same seed gives the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** There is no safety checker, and the optional image watermark is off. The SDXL
refiner model, img2img, other schedulers and fp16 parity are not in this release.

**License.** The weights use the `playground-v2dot5-community` license, which restricts commercial
use under its terms; see the license. imagehub does not distribute weights.


# Z-Image

Catalog: [`z-image`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `z-image`, `z-image-turbo` · Task: `t2i`

**Architecture.** A Qwen3-4B text encoder renders the prompt through its chat template and supplies
penultimate hidden states. A 6B single-stream DiT uses 3-axis RoPE and refiner blocks. The
FLUX-style VAE has 16 latent channels and a shift factor. `z-image` is the non-distilled base model
with the same architecture. The flow-match Euler scheduler uses shift 6.0 for `z-image` and shift 3
for `z-image-turbo`. The native image size is 1024×1024.

**Usage.**

```bash
python -m imagehub.pretrained z-image-turbo checkpoints/z-image-turbo
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="z-image", model="checkpoints/z-image-turbo", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** For `z-image`, `steps` defaults to 50 and `guidance` defaults to 4.0. For
`z-image-turbo`, they default to 9 and 0.0. `width`/`height` default to 1024 for both and must be
multiples of 16. The supported options are `negative_prompt`, `num_images`, `seed`, `latents`,
`cfg_normalization`, `cfg_truncation` and `max_sequence_length`. Unrecognized settings raise an
error.

**Seed compatibility.** On CPU, the same seed gives the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** There is no safety checker or invisible watermark. Omni, edit, controlnet, inpaint,
img2img, LoRA and fp16/bf16 parity are not in this release.

**License.** The weights are licensed under Apache-2.0. imagehub does not distribute weights.
