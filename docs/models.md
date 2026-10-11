# Image input

Most models take a text prompt only. The models in this table also accept an input image through the
`image` option (a `PIL.Image.Image` or an `imagehub.Image`); every model not listed here takes no
image input, and passing `image` to one raises an error.

| Model | Image input | Use | Options |
| --- | --- | --- | --- |
| `flux1-kontext-dev` | optional | instruction editing | `image` |
| `sdxl-refiner-1.0` | required | refinement (img2img) | `image`, `strength` |

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


# Stable Diffusion 2.1

Catalog: [`stable-diffusion-2`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `sd-v2-1` · Task: `t2i`

**Architecture.** A latent diffusion model with a frozen OpenCLIP-H text encoder, a U-Net with
linear-projection attention and an f=8 KL autoencoder. It is trained with v-prediction and sampled
with the checkpoint's DDIM scheduler (`eta=0`, `set_alpha_to_one=false`, `steps_offset=1`). The
U-Net config sets `upcast_attention`; the reference uses scaled dot-product attention, where the
flag has no effect, so the same output is produced. Its native image size is 768×768. The weights
come from `sd2-community/stable-diffusion-2-1`, a community copy of the withdrawn
`stabilityai/stable-diffusion-2-1`.

**Usage.**

```bash
python -m imagehub.pretrained sd-v2-1 checkpoints/sd-v2-1
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="sd-v2-1", model="checkpoints/sd-v2-1", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** `steps` defaults to 50, `guidance` defaults to 7.5 and `width`/`height` default to
768. `negative_prompt`, `num_images`, `seed` and `latents` are also supported. Unrecognized
settings raise an error.

**Seed compatibility.** On CPU, the same seed gives the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** There is no safety checker or invisible watermark. img2img, other schedulers and
fp16 parity are not in this release.

**License.** The weights use the CreativeML-OpenRAIL++-M license. imagehub does not distribute
weights.


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
(the refiner `sdxl-refiner-1.0` has its own section below)

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

**Out of scope.** There is no safety checker, and the optional image watermark is off. Image input
(img2img) is available only for the SDXL refiner, which has its own section; other schedulers and
fp16 parity are not in this release.

**License.** The SDXL base and SDXL-Lightning weights are licensed under CreativeML Open RAIL++-M;
the license's use restrictions bind whoever downloads and uses the weights. The Hyper-SD model
card has no license metadata, and the repository's license file covers only the FLUX-related files,
so the Hyper-SDXL weights are recorded as `Unspecified`. The DMD2 weights are licensed under
CC-BY-NC-4.0, which prohibits commercial use. imagehub does not distribute weights.


# SDXL refiner

Catalog: [`sdxl`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `sdxl-refiner-1.0` · Task: `t2i` (image input required)

**Architecture.** The refiner is an SDXL U-Net specialised in the low-noise end of the schedule. It
differs from the base model in four ways: it has a single text encoder (OpenCLIP bigG: penultimate
hidden state, 1280 channels, and its pooled output), a U-Net with four resolution levels (384, 768,
1536, 1536 channels; 4 transformer layers per block), a `text_time` conditioning vector of five time
ids per sample (original size, crop position and an aesthetic score) instead of six, and the VAE
and scheduler of the base model. `vae/` is used; the repository's second `vae_1_0/` is not.

**Usage.**

```bash
python -m imagehub.pretrained sdxl-refiner-1.0 checkpoints/sdxl-refiner-1.0
```

```python
from PIL import Image
from imagehub import pipeline
with pipeline("t2i", backend="sdxl-refiner-1.0", model="checkpoints/sdxl-refiner-1.0", device="mps") as pipe:
    result = pipe("a cat, oil painting", image=Image.open("cat.png"), strength=0.3, seed=0)
    result.images[0].save("refined.png")
```

- **The image is required.** It is converted to RGB, rounded down to a multiple of 8 and resized
  with Lanczos as diffusers' `VaeImageProcessor` does, encoded by the VAE by sampling the
  posterior (this draws from the seeded generator before the noise does) and scaled by 0.13025.
  The result has the size of the preprocessed input; `width`, `height` and `latents` are not
  accepted.
- **Strength.** `strength` (default 0.3) keeps the last `int(steps * strength)` of the `steps`
  timesteps: the encoded image is noised to the sigma at the first kept step and denoised from
  there. A `strength` so small that no step remains is rejected.
- **Defaults.** 50 steps, guidance 5.0, `strength` 0.3, `aesthetic_score` 6.0 and
  `negative_aesthetic_score` 2.5. Since `force_zeros_for_empty_prompt` is false, an empty negative
  prompt is encoded, not zeroed. Other settings: `negative_prompt`, `num_images`, `seed`,
  `original_size`, `crops_coords_top_left`, `target_size` and their `negative_*` counterparts.
  Unrecognized settings raise an error.
- **Not supported.** Running the refiner after the base model in latent space (`denoising_end` on
  the base and `denoising_start` on the refiner, ensemble of expert denoisers) is out of scope: the
  refiner takes an image. Also not supported: the other schedulers of the img2img pipeline, several
  input images, IP-Adapter, a safety checker, the invisible watermark and fp16 parity.
- **Seed compatibility.** On CPU, the same seed, image and settings give the same image as diffusers
  0.40.0 `StableDiffusionXLImg2ImgPipeline` with `generator=torch.Generator("cpu").manual_seed(seed)`
  ([validation](validation.md)). MPS and CUDA runs are smoke tests only.
- **License.** CreativeML Open RAIL++-M; the license's use restrictions bind whoever downloads and
  uses the weights. The repository is not gated. imagehub does not distribute weights.
- **Memory.** In float32: U-Net about 9.4 GB, text encoder 2 about 2.8 GB, VAE 0.3 GB.


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

**Out of scope.** There is no safety checker, and the optional image watermark is off. Image input
(img2img) is available only for the SDXL refiner, which has its own section; other schedulers and
fp16 parity are not in this release.

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

**Out of scope.** There is no safety checker, and the optional image watermark is off. Image input
(img2img) is available only for the SDXL refiner, which has its own section; other schedulers and
fp16 parity are not in this release.

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
Models: `flux1-schnell`, `flux1-dev`, `flux1-krea-dev`, `flux1-kontext-dev` · Task: `t2i`

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

**Out of scope.** Fill, Redux, Depth and Canny variants, ControlNet, LoRA and fp16/bf16 parity are
not supported. `flux1-schnell`, `flux1-dev` and `flux1-krea-dev` take no image input; for Kontext, see below.

**License.** The schnell weights use the Apache-2.0 license. The dev, Krea [dev] and Kontext weights use the
FLUX.1-dev Non-Commercial License; see the license. imagehub does not distribute weights.

## FLUX.1 Krea [dev]

`flux1-krea-dev` (`black-forest-labs/FLUX.1-Krea-dev`) is an aesthetics-focused fine-tune of the
FLUX.1 [dev] architecture, per its model card. Its published configs are identical to those of
`flux1-dev`, so it runs on the same backend, text encoders, transformer and VAE with no new code.

```python
from imagehub import pipeline
with pipeline("t2i", backend="flux1-krea-dev", model="checkpoints/flux1-krea-dev", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

- **Defaults.** 28 steps (the diffusers `FluxPipeline` default; the model card gives no step
  count), guidance 4.5 (the model card), `max_sequence_length` 512 and 1024×1024. The other
  options are the same as for `flux1-dev`.
- **License and access.** The repository is gated and the weights use the FLUX.1-dev
  Non-Commercial License. Accept the terms and provide `HF_TOKEN` when downloading.
- **Memory.** The same as `flux1-dev`: about 47.6 GB for the transformer, 19 GB for T5-XXL, 0.5 GB
  for CLIP-L and 0.3 GB for the VAE in float32.

## FLUX.1 Kontext [dev]

`flux1-kontext-dev` (`black-forest-labs/FLUX.1-Kontext-dev`) is the FLUX.1 transformer trained to
edit an image from a text instruction. It shares the FLUX.1 backend, text encoders and VAE; the
only difference is that the tokens of the encoded input image are appended to the sequence, and
only the generated tokens are decoded.

```python
from PIL import Image
from imagehub import pipeline
with pipeline("t2i", backend="flux1-kontext-dev", model="checkpoints/flux1-kontext-dev", device="mps") as pipe:
    result = pipe("Add a red hat to the scene", image=Image.open("photo.png"), seed=0)
    result.images[0].save("edited.png")
```

- **The image is optional.** Without `image`, the model generates from the prompt alone, like
  `flux1-dev`.
- **Input resizing.** The input is converted to RGB (grayscale, RGBA and palette images included),
  rounded down to a multiple of 16, and then resized to the preferred Kontext resolution whose
  aspect ratio is closest to its own (672×1568 up to 1568×672, 1024×1024 for a square). It is
  encoded by the VAE (the mode of the posterior, shifted and scaled) before it joins the sequence.
- **Output size.** The size of the result comes from `height` and `width` (default 1024×1024), not
  from the input image. They are rescaled to at most `max_area` pixels (default 1024²) at the
  same aspect ratio and rounded down to multiples of 16. To keep the input's shape, pass its
  size as `height` and `width`.
- **Defaults.** 28 steps, guidance 3.5, `max_sequence_length` 512, `true_cfg_scale` 1.0 (no
  negative prompt). The model card's examples use guidance 2.5; pass `guidance=2.5` to use it.
  The other FLUX.1 options (`num_images`, `seed`, `latents`, `negative_prompt`, `true_cfg_scale`)
  work as for `flux1-dev`; `max_area` is the one added option besides `image`.
- **License and access.** The repository is gated and the weights use the FLUX.1-dev
  Non-Commercial License. Accept the terms and provide `HF_TOKEN` when downloading.
- **Memory.** The same as `flux1-dev`: about 47.6 GB for the transformer, 19 GB for T5-XXL, 0.5 GB
  for CLIP-L and 0.3 GB for the VAE in float32.
- **Limits.** One input image only. Inpainting (`FluxKontextInpaintPipeline`), several input
  images, IP-Adapter and fp16/bf16 parity are not supported.
- **Seed compatibility.** On CPU, the same seed, image and settings give the same image as
  diffusers 0.40.0 `FluxKontextPipeline` ([validation](validation.md)).


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
Models: `hunyuan-dit-v1.1-distilled`, `hunyuan-dit-v1.2`, `hunyuan-dit-v1.2-distilled` · Task: `t2i`

**Architecture.** Bilingual Chinese/English BERT-large and mT5-XL text encoders condition a 1.5B
diffusion transformer with 40 blocks, width 1408, 16 attention heads of width 88, long skip
connections and 2D RoPE. It also uses timestep, image-size, style and pooled-text conditioning and
predicts a learned sigma. A 4-channel VAE with scaling factor 0.13025 decodes the latents. The DDPM
scheduler uses v-prediction, and requested resolutions are binned to supported shapes. The v1.2
releases drop the image-size and style conditioning (`use_style_cond_and_image_meta_size` is false)
and use a smaller final beta (0.018) in the scheduler.

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

**Settings.** `steps` defaults to 25 (50 for `hunyuan-dit-v1.2`), `guidance` defaults to 5.0 and `width`/`height` default to
1024. The other supported options are `seed`, `latents`, `negative_prompt`, `guidance_rescale` and
`use_resolution_binning`. Unrecognized settings raise an error.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** Other HunyuanDiT releases (v1.0 and the non-distilled v1.1), ControlNet, PAG, LoRA
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


# Stable Cascade

Catalog: [`wuerstchen`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `stable-cascade`, `stable-cascade-lite` · Task: `t2i`

**Architecture.** Three stages work at very different resolutions. Stage C (the prior) denoises a
small 16-channel image embedding that is about 42 times smaller than the output, conditioned on the
CLIP ViT-bigG text states and pooled text embedding. Stage B (the decoder) denoises a 4-channel
latent about 10.7 times smaller than the output, conditioned on the stage C embedding and the same
pooled text embedding. Stage A, a Paella VQGAN, decodes that latent to pixels. Stages B and C use the
same U-Net module with a different block layout, and both use a DDPM Wuerstchen scheduler with a
cosine schedule.

| Model | Stage C (prior) | Stage B (decoder) |
| --- | --- | --- |
| `stable-cascade` | `prior`, 3.6B parameters | `decoder`, 1.5B parameters |
| `stable-cascade-lite` | `prior_lite`, 1B parameters | `decoder_lite`, 700M parameters |

**Usage.**

```bash
python -m imagehub.pretrained stable-cascade-lite checkpoints/stable-cascade-lite
```

```python
from imagehub import pipeline
with pipeline("t2i", model="stable-cascade-lite", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** The prior defaults to 60 steps and guidance 4.0; the decoder defaults to 12 steps,
guidance 0.0 and 512×512 output. The supported options are `steps`, `guidance`, `prior_steps`,
`prior_guidance`, `width`, `height`, `seed` and `negative_prompt`. Width and height must be
multiples of 8. Decoder guidance above 1 needs the negative embeddings, so it also requires
`prior_guidance` above 1. Unrecognized settings raise an error.

**Memory.** In float32 the weights take about 14.4 GB for the stage C prior (4.1 GB for lite), 6.3 GB
for the stage B decoder (2.8 GB for lite), 2.8 GB for the text encoder and 0.07 GB for stage A.
The full pair therefore needs about 24 GB, and the lite pair about 10 GB.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers
`StableCascadeCombinedPipeline` with `generator=torch.Generator("cpu").manual_seed(seed)`
([validation](validation.md)). Both stages draw from the one generator. MPS and CUDA runs are smoke
tests only; there is no claim of exact equality on these devices.

**Out of scope.** Only text-to-image is supported. Image variation (the prior's CLIP image encoder),
ControlNet, the stage A encoder, the previewer and fp16/bf16 parity are not implemented. The decoder
repository's own text encoder is not used: the combined pipeline passes the prior's text embeddings
to the decoder, and so does imagehub.

**License.** The weights use the Stable Cascade Non-Commercial Community License
(`stable-cascade-nc-community`), which prohibits commercial use; see the license. imagehub does not
distribute weights.


# Würstchen v2

Catalog: [`wuerstchen`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `wuerstchen-v2` · Task: `t2i`

**Architecture.** Würstchen v2 is the model that Stable Cascade grew out of, and it works the same
way in three stages. The prior (stage C) denoises a 16-channel image embedding about 42.67 times
smaller than the output, conditioned on the CLIP ViT-bigG text states. The decoder (stage B, a
DiffNeXt U-Net) denoises a 4-channel latent about 10.67 times smaller than the output, conditioned on
the prior's embedding and on the text states of a second text encoder, CLIP ViT-H. A Paella VQGAN
(stage A) decodes that latent to pixels. Both stages use the DDPM Wuerstchen scheduler.

The differences from [Stable Cascade](#stable-cascade) are in the networks and the conditioning. The
prior is a stack of 32 residual, timestep and attention blocks at a single resolution (width 1536)
rather than the two-level U-Net. The decoder has four resolution levels and takes the whole
sequence of text states, not a pooled text embedding. The decoder has its own text encoder, so
`wuerstchen-v2` loads two CLIP text models, and each stage encodes the prompt with its own: the
last hidden state after the final layer norm, with the padding masked out. The prior's output is
rescaled as `latents * 42 - 1` before it conditions the decoder. The prior uses the batched-matmul
attention of the diffusers default processor and the decoder uses fused scaled-dot-product
attention, which is what the reference does and what exact parity requires. The sampling loop,
the shared random generator and the decoding follow Stable Cascade.

**Usage.**

```bash
python -m imagehub.pretrained wuerstchen-v2 checkpoints/wuerstchen-v2
```

```python
from imagehub import pipeline
with pipeline("t2i", model="wuerstchen-v2", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** The prior defaults to 60 steps and guidance 4.0; the decoder defaults to 12 steps,
guidance 0.0 and 512×512 output. The supported options are `steps`, `guidance`, `prior_steps`,
`prior_guidance`, `width`, `height`, `seed` and `negative_prompt`. The reference sizes the latents as
`ceil(size / 42.67)` for the prior and `int(that * 10.67)` for the decoder, and the decoder needs a
latent size that is a multiple of 16, so sizes are checked before loading and multiples of 128
always work; the output can differ slightly from the requested size for other values, and the
result metadata records the size produced. Decoder guidance above 1 does not depend on the prior
guidance, because the decoder encodes its own negative prompt. Unrecognized settings raise an error.

**Memory.** In float32 the weights take about 4.0 GB for the prior (1.0B parameters), 4.2 GB for the
decoder (1.06B), 2.8 GB for the ViT-bigG text encoder, 1.4 GB for the ViT-H text encoder and 0.07 GB
for stage A, about 12.5 GB in all.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers
`WuerstchenCombinedPipeline` with `generator=torch.Generator("cpu").manual_seed(seed)`
([validation](validation.md)). Both stages draw from the one generator. MPS and CUDA runs are smoke
tests only; there is no claim of exact equality on these devices.

**Out of scope.** Only text-to-image is supported, for one prompt at a time. Image variation,
custom timestep lists, precomputed prompt embeddings and fp16/bf16 parity are not implemented.

**License.** The decoder and prior weights are published under the MIT license. imagehub does not
distribute weights.


# PixArt

Catalog: [`pixart-sigma`](https://github.com/gorkemgul/awesome-text-to-image-architectures),
[`pixart-alpha`](https://github.com/gorkemgul/awesome-text-to-image-architectures) · Models:
`pixart-sigma-xl-1024`, `pixart-sigma-xl-512`, `pixart-sigma-xl-256`, `pixart-sigma-xl-2k`,
`pixart-alpha-xl-1024`, `pixart-alpha-xl-512` · Task: `t2i`

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

**Resolutions.** Each resolution is a separate checkpoint with its own transformer: Sigma at 256,
512, 1024 and 2K (interpolation scales 0.5, 1, 2 and 4) and Alpha at 512 and 1024. The Sigma 256, 512 and
2K repositories publish only a transformer, so `python -m imagehub.pretrained` fetches the text
encoder, tokenizer, VAE and scheduler from the Sigma 1024 repository, as the model cards describe. The
Alpha resolution and aspect-ratio embeddings exist only in the 1024 model. Resolution binning
uses the table that matches the transformer (256, 512, 1024 or 2048).

**Settings.** All models default to 20 steps and guidance 4.5, and to a square image of their native
size (256, 512, 1024 or 2048 pixels). Sigma defaults to a maximum sequence length of 300 and Alpha to 120. The other supported options are `seed`, `latents`,
`negative_prompt`, `use_resolution_binning` and `max_sequence_length`. Unrecognized settings raise
an error.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** PixArt-LCM, PixArt-DMD, ControlNet,
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


# Playground v2

Catalog: [`playground-v2`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `playground-v2` · Task: `t2i`

**Architecture.** An SDXL-family model with the same two frozen text encoders, U-Net layout and
`text_time` micro-conditioning as SDXL. Unlike Playground v2.5, it keeps the standard SDXL
autoencoder (scaling factor 0.13025) and the Euler ancestral scheduler with the scaled-linear
noise schedule and `leading` timestep spacing. Its native image size is 1024×1024.

**Usage.**

```bash
python -m imagehub.pretrained playground-v2 checkpoints/playground-v2
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="playground-v2", model="checkpoints/playground-v2", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** `steps` defaults to 50, `guidance` defaults to 3.0 (the value recommended on the model
card) and `width`/`height` default to 1024. Other supported settings match SDXL. Unrecognized
settings raise an error.

**Seed compatibility.** On CPU, the same seed gives the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). The Euler
ancestral scheduler draws its per-step noise from the same seeded generator. MPS and CUDA runs are
smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** There is no safety checker, and the optional image watermark is off. The SDXL
refiner model, img2img, other schedulers and fp16 parity are not in this release.

**License.** The weights use the `playground-v2-community` license, which restricts commercial use
under its terms; see the license. imagehub does not distribute weights.


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

**Out of scope.** There is no safety checker, and the optional image watermark is off. Image input
(img2img) is available only for the SDXL refiner, which has its own section; other schedulers and
fp16 parity are not in this release.

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

# Sana

Catalog: [`sana`](https://github.com/gorkemgul/awesome-text-to-image-architectures) · Models:
`sana-1600m-1024`, `sana-600m-1024` · Task: `t2i`

**Architecture.** A Gemma-2 2B text model (26 layers, width 2304, soft-capped attention and
alternating sliding-window layers) encodes the prompt, and its last hidden state conditions a linear
diffusion transformer. The transformer applies ReLU linear attention over the image tokens,
cross-attention to the masked text, GLUMBConv feed-forward blocks and AdaLN-single timestep
modulation. The 1600M model has 20 blocks of width 2240 and the 600M model has 28 blocks of width
1152. Images are decoded by the DC-AE autoencoder, which compresses 32× into 32 latent channels
(scaling factor 0.41407). The scheduler is the order-2 midpoint DPM-Solver++ with flow sigmas and
`flow_shift` 3.0. Before encoding, a "complex human instruction" is prepended to the prompt, and the
first token plus the last 299 tokens of the encoder output are kept.

**Usage.**

```bash
python -m imagehub.pretrained sana-1600m-1024 checkpoints/sana-1600m-1024
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="sana-1600m-1024", model="checkpoints/sana-1600m-1024", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** Both models default to 20 steps, guidance 4.5 and 1024×1024, with an empty negative
prompt. The supported options are `seed`, `steps`, `guidance`, `width`, `height`,
`negative_prompt`, `use_resolution_binning`, `max_sequence_length` (default 300) and
`complex_human_instruction`. With `use_resolution_binning` on (the default), the requested size is
snapped to the closest aspect-ratio bin, the image is generated at the bin size, and the decoded
image is resized and centre-cropped back to the request; turn it off to generate at the exact size,
which must then be a multiple of 32. `complex_human_instruction` is a list of strings; `None` selects
the published instruction and an empty list disables it. The instruction is never added to the
negative prompt. Guidance 1.0 or lower skips the unconditional pass. Unrecognized settings raise an
error.

**Memory.** In float32 the transformer needs about 6.4 GB (2.4 GB for the 600M model), the Gemma-2
text model about 10.5 GB and DC-AE about 1.3 GB. The Gemma-2 weights are stored in bfloat16 and
upcast on load.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** The 2K and 4K checkpoints, SANA 1.5, SANA-Sprint, ControlNet, PAG, fp16/bf16
parity and caption cleaning are not supported. Caption cleaning requires external packages; parity
is against `clean_caption=False`. Only decoding is exposed for DC-AE.

**License.** Both repositories declare the Apache-2.0 license. The text encoder is a Gemma-2
checkpoint; check Google's Gemma terms of use before redistributing it. imagehub does not
distribute weights.

# Kolors

Catalog: [`kolors`](https://github.com/gorkemgul/awesome-text-to-image-architectures) · Models:
`kolors` · Task: `t2i`

**Architecture.** A ChatGLM3-6B text encoder (28 layers, width 4096, grouped-query attention with
two key/value groups, rotary embeddings, RMSNorm) encodes the prompt, left-padded to 256 tokens
with a `[gMASK] sop` prefix. The input of the last encoder block, as `[256, 4096]`, is the
cross-attention context and the last position of the last block's output, as `[4096]`, is the
pooled vector. The generator is an SDXL-style U-Net (three levels, 320/640/1280 channels,
transformer depths 1/2/10) whose cross-attention width of 2048 is reached by a linear projection of
the 4096-wide context. The pooled vector and the six size and crop values are embedded as in SDXL
and added to the timestep embedding. Images are decoded by the SDXL-style KL autoencoder (scaling
factor 0.13025). The scheduler is Euler discrete with 1100 training steps, leading spacing and a
beta range ending at 0.014.

**Usage.**

```bash
python -m imagehub.pretrained kolors checkpoints/kolors
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="kolors", model="checkpoints/kolors", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** The defaults are 50 steps, guidance 5.0 and 1024×1024. The negative prompt defaults
to the empty string, which is encoded like any other (Kolors does not zero it). The supported
options are `seed`, `latents`, `steps`, `guidance`, `width`, `height`, `num_images`,
`negative_prompt`, `original_size`, `crops_coords_top_left` and `target_size`; the last three are
the SDXL micro-conditioning values and default to the image size and `(0, 0)`. Width and height
must be multiples of 8 and `steps` at most 1100. Guidance 1.0 or lower skips the unconditional
pass. Prompts are truncated to 256 tokens. Unrecognized settings raise an error.

**Weights.** Only float16 weights are published, so the snapshot holds `*.fp16.safetensors` files
and a float16 shard index for the text encoder. They are read directly and upcast to float32 on
load; parity is measured on that upcast, against diffusers loaded with `variant="fp16"` and
`torch_dtype=torch.float32`. A float16 VAE with `force_upcast` decodes in float32 and is restored
afterwards.

**Memory.** In float32 the U-Net needs about 10.3 GB, the ChatGLM3 text encoder about 25 GB and
the autoencoder about 0.3 GB; the float16 files total about half of that.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** Kolors IP-Adapter, ControlNet and inpainting variants, LoRA, prompts longer
than 256 tokens, other schedulers, `denoising_end`, negative micro-conditioning and
precomputed prompt embeddings are not supported.

**License.** The repository declares the Apache-2.0 license. imagehub does not distribute weights.

# Lumina-Image 2.0

Catalog: [`lumina-image-2`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `lumina-image-2.0` · Task: `t2i`

**Architecture.** A Gemma-2 2B text model (26 layers, width 2304) encodes the prompt, and its
second-to-last hidden state conditions a 2.6B-parameter Next-DiT transformer. The transformer has 26
joint blocks of width 2304 (24 attention heads, 8 key-value heads) and two refiner blocks each for the
text and for the image tokens. It uses 3D rotary position embeddings over the text position and the
image row and column, sandwich normalization, and timestep modulation in the image refiner and joint blocks. The transformer works on
2×2 patches of the 16-channel latents from the FLUX-style autoencoder (scaling factor 0.3611, shift
factor 0.1159). The scheduler is flow-match Euler with a static shift of 6.0. The model treats t=0 as
noise, so the timestep is reversed and the prediction is negated before each step. Before encoding, a
system prompt and the marker `<Prompt Start>` are prepended to the prompt; the negative prompt is
encoded without them.

**Usage.**

```bash
python -m imagehub.pretrained lumina-image-2.0 checkpoints/lumina-image-2.0
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="lumina-image-2.0", model="checkpoints/lumina-image-2.0", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** The model defaults to 30 steps, guidance 4.0 and 1024×1024, with an empty negative
prompt. The supported options are `seed`, `steps`, `guidance`, `width`, `height`,
`negative_prompt`, `system_prompt`, `cfg_trunc_ratio` (default 1.0), `cfg_normalization` (default
true) and `max_sequence_length` (default 256, at most 299 for the published model). `width` and
`height` must be multiples of 16. `system_prompt` replaces the published system prompt; `None`
selects it. Guidance is applied as `uncond + guidance * (cond - uncond)`; with `cfg_normalization`
the result is rescaled to the norm of the conditional prediction. The unconditional pass is skipped
for steps after the fraction `cfg_trunc_ratio` of the schedule, so 1.0 guides every step. Guidance
1.0 or lower skips the unconditional pass entirely. Unrecognized settings raise an error.

**Memory.** In float32 the transformer needs about 10.4 GB, the Gemma-2 text model about 10.5 GB and
the autoencoder about 0.3 GB.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** LoRA, fp16/bf16 parity, batches of several prompts or images per call, and
caller-supplied latents or embeddings are not supported.

**License.** The repository declares the Apache-2.0 license. The text encoder is a Gemma-2
checkpoint; check Google's Gemma terms of use before redistributing it. imagehub does not
distribute weights.

# Lumina-Next

Catalog: [`lumina-next`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `lumina-next` · Task: `t2i`

This is the 2B Next-DiT from the Lumina-T2X family (`Lumina-Next-SFT`), published for diffusers as
`LuminaPipeline`. It is a different architecture from Lumina-Image 2.0 (`lumina-image-2.0`).

**Architecture.** A Gemma 2B text model (the first-generation Gemma: 18 layers, width 2048, eight
query heads and one key-value head of size 256) encodes the prompt, and its second-to-last hidden
state conditions a 2B Next-DiT transformer. The transformer has 24 blocks of width 2304 (32 attention
heads, 8 key-value heads). Each block runs self-attention over the image tokens and cross-attention
to the text; the cross-attention output is multiplied by a per-head `tanh` gate before both are
summed and projected. Queries and keys are normalized with a LayerNorm over all heads, and both
attentions use two-dimensional rotary position embeddings. A pooled caption feature is added to the
timestep embedding, which modulates every block through adaptive RMS norms. The transformer works on
2×2 patches of the four-channel latents of the SDXL autoencoder (scaling factor 0.13025) and predicts
a variance in four extra channels that is discarded. The scheduler is flow-match Euler without
shifting. The model treats t=0 as noise, so the timestep is reversed and the prediction is negated
before each step. The prompt is lower-cased and stripped, then padded to a multiple of eight tokens
(at most 256, including the start and end tokens). The negative prompt is used as written and padded
to the prompt's length.

Two options let the model generate at sizes other than the training size of 1024×1024. With
proportional attention, the self-attention logits are rescaled by the square root of the base-4096
logarithm of the token count. With time-aware scaled RoPE the rotary positions are stretched
linearly by the factor `sqrt(width * height / 1024²)` while the reversed timestep is below
`scaling_watershed`, and the rotary base is multiplied by that factor afterwards. At the training
size the factor is 1 and both options change nothing. Guidance is applied to the first three latent
channels only, as the reference pipeline does.

**Usage.**

```bash
python -m imagehub.pretrained lumina-next checkpoints/lumina-next
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="lumina-next", model="checkpoints/lumina-next", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** The model defaults to 30 steps, guidance 4.0 and 1024×1024, with an empty negative
prompt. The supported options are `seed`, `steps`, `guidance`, `width`, `height`,
`negative_prompt`, `scaling_watershed` (default 1.0) and `proportional_attn` (default true).
`width` and `height` must be multiples of 16 and at most 6144. Guidance is applied as
`uncond + guidance * (cond - uncond)`; guidance 1.0 or lower skips the unconditional pass
entirely. The default `scaling_watershed` of 1.0 stretches the rotary positions at every step;
a lower value switches to the base-scaling form for the later steps. Unrecognized settings raise
an error.

**Memory.** In float32 the transformer needs about 7.0 GB, the Gemma 2B text model about 10 GB and
the autoencoder about 0.3 GB. The autoencoder is the SDXL one, which overflows in float16; use
float32 or bfloat16.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)); the pending
real-checkpoint run will record the measurements. MPS and CUDA runs are smoke tests only; there is no
claim of exact equality on these devices.

**Out of scope.** `clean_caption` (the reference needs `beautifulsoup4` and `ftfy`, and falls back
to lower-casing when they are missing, which is what this backend does), `max_sequence_length` (the
reference pipeline ignores it and always uses 256), LoRA, fp16/bf16 parity, batches of several
prompts or images per call, and caller-supplied latents or embeddings.

**License.** The repository declares the Apache-2.0 license. The text encoder is a Gemma
checkpoint; check Google's Gemma terms of use before redistributing it. imagehub does not
distribute weights.

# aMUSEd

Catalog: [`amused`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `amused-256`, `amused-512` · Task: `t2i`

**Architecture.** aMUSEd is a masked image-token model, not a diffusion model. A VQGAN with a
codebook of 8192 entries turns an image into a grid of token ids at 16× downsampling (16×16 tokens
for the 256 model, 32×32 for the 512 model, which downsamples and upsamples the grid inside the
network). A U-ViT predicts the ids of masked tokens. It has a ConvNeXt stem, a stack of 22
transformer layers of width 1024 with cross-attention to the text, and a ConvNeXt head, and it is
conditioned on the CLIP ViT-L/14 text encoder: the projected pooled embedding and the second-to-last
hidden state. Micro-conditioning (original size, crop coordinates and aesthetic score) is added to the
pooled embedding. Generation starts from a grid of mask tokens. At every step the transformer predicts
all tokens, one id is sampled for each masked position, and the least confident tokens are masked
again, following a cosine schedule. The final tokens are looked up in the VQGAN codebook and decoded
to pixels.

**Usage.**

```bash
python -m imagehub.pretrained amused-512 checkpoints/amused-512
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="amused-512", model="checkpoints/amused-512", device="mps") as pipe:
    result = pipe("a photo of an astronaut riding a horse on mars", seed=42)
    result.images[0].save("astronaut.png")
```

**Settings.** The models default to 12 steps, guidance 10.0, an empty negative prompt and the size
of their token grid (256×256 or 512×512). The supported options are `seed`, `steps`, `guidance`,
`width`, `height`, `negative_prompt`, `aesthetic_score` (default 6), `crop_coord` (default `(0, 0)`)
and `temperature` (default `(2, 0)`). `width` and `height` must be multiples of 16 for `amused-256`
and of 32 for `amused-512`. `temperature` is either a number, which falls linearly from that number
to 0.01 over the steps, or a `(start, end)` pair, which is interpolated linearly. It scales the random
noise that is added to the token confidences before re-masking. Guidance is applied to the token
logits as `uncond + guidance * (cond - uncond)`; guidance 1.0 or lower skips the unconditional pass.
The seed drives a CPU generator that is used for the token sampling and the confidence noise, so
the same seed gives the same tokens on every device up to numerical differences. Unrecognized
settings raise an error.

**Memory.** In float32 the transformer needs about 2.4 GB, the CLIP text encoder about 0.5 GB and the
VQGAN about 0.6 GB.

**Seed compatibility.** On CPU, the same seed and settings give the same token ids and the same image
as diffusers' `AmusedPipeline` with `generator=torch.Generator("cpu").manual_seed(seed)`
([validation](validation.md)). MPS and CUDA runs are smoke tests only; there is no claim of exact
equality on these devices.

**Out of scope.** LoRA, fp16/bf16 parity, batches of several prompts or images per call, image
inpainting and variation, and caller-supplied tokens or embeddings are not supported.

**License.** Both repositories declare the OpenRAIL++ license; check its use restrictions before
redistributing generated images or weights. imagehub does not distribute weights.

# AuraFlow

Catalog: [`auraflow`](https://github.com/gorkemgul/awesome-text-to-image-architectures) · Models:
`auraflow-v0.3` · Task: `t2i`

**Architecture.** A UMT5 encoder (24 layers, width 2048, a relative position bias in every layer)
encodes the prompt, and its last hidden state, zeroed at padded positions, conditions a 6.8B-parameter
flow-matching transformer of width 3072 (12 heads of 256). Four MMDiT blocks process the image and
text streams jointly, then 32 single-stream DiT blocks process the two together. Eight learned
register tokens are prepended to the text, queries and keys are layer-normalized per head, and the
image patches (2×2) receive a learned positional embedding that is centre-cropped from a 96×96 grid.
Images are decoded by the 4-channel, 8× SDXL-style VAE (scaling factor 0.13025). The scheduler is
flow-match Euler with a static `shift` of 1.73; the transformer receives timesteps in [0, 1].

**Usage.**

```bash
python -m imagehub.pretrained auraflow-v0.3 checkpoints/auraflow-v0.3
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="auraflow-v0.3", model="checkpoints/auraflow-v0.3", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42)
    result.images[0].save("cat.png")
```

**Settings.** The model defaults to 50 steps, guidance 3.5 and 1024×1024, with an empty negative
prompt. The supported options are `seed`, `steps`, `guidance`, `width`, `height`,
`negative_prompt` and `max_sequence_length` (default 256). The size must be a multiple of 16 and at
most 1536 per side, which is the extent of the learned positional grid. Guidance 1.0 or lower skips
the unconditional pass. Unrecognized settings raise an error.

**Memory.** In float32 the transformer needs about 27 GB, the UMT5 encoder about 4.9 GB and the VAE
about 0.3 GB. The text encoder is stored in float16 and upcast on load.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** Other AuraFlow releases, LoRA, ControlNet, fused projections, several images per
prompt and fp16/bf16 parity are not supported.

**License.** The repository declares the Apache-2.0 license. imagehub does not distribute weights.

# LlamaGen

Catalog: [`llamagen`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `llamagen-t2i-xl-256`, `llamagen-t2i-xl-512` · Task: `t2i`

**Architecture.** LlamaGen is an autoregressive image model: the image is a grid of discrete tokens
and a decoder-only transformer predicts them one at a time, like a language model. A VQ-16
tokenizer (a codebook of 16384 entries of dimension 8, 16× downsampling) maps between tokens and
pixels, so the 256 model writes 16×16 = 256 tokens and the 512 model 32×32 = 1024 tokens. The
transformer (GPT-XL: 36 layers, width 1280, 20 heads, RMSNorm, SwiGLU, two-dimensional rotary
position embeddings over the token grid) is conditioned on the prompt by a prefix: the prompt is
encoded by the Flan-T5-XL encoder into 120 vectors of width 2048, a small MLP projects them to the
model width, and they are placed in front of the image tokens. Each new token is drawn from the
softmax of the logits after temperature, top-k and top-p filtering. Classifier-free guidance runs a
second pass in which the caption is replaced by a learned "no caption" embedding and combines the
logits as `uncond + cfg_scale * (cond - uncond)`.

**Usage.**

```bash
python -m imagehub.pretrained llamagen-t2i-xl-256 checkpoints/llamagen-t2i-xl-256
```

```python
from imagehub import pipeline
with pipeline(
    "t2i", backend="llamagen-t2i-xl-256", model="checkpoints/llamagen-t2i-xl-256", device="mps"
) as pipe:
    result = pipe("a blue Porsche 356 parked in front of a yellow brick wall", seed=42)
    result.images[0].save("porsche.png")
```

The snapshot holds `vq_ds16_t2i.pt` and the GPT checkpoint (`t2i_XL_stage1_256.pt` or
`t2i_XL_stage2_512.pt`) at its root, and the Flan-T5-XL files (config, tokenizer and the two
safetensors shards) in `text_encoder/`. The two image sizes are separate checkpoints; the size is
fixed by the checkpoint.

**Settings.** The defaults are those of the official sampling script: `cfg_scale` 7.5, `temperature`
1.0, `top_k` 1000, `top_p` 1.0 and `cfg_interval` -1. The supported options are `seed`,
`cfg_scale`, `temperature`, `top_k`, `top_p`, `cfg_interval`, `width` and `height`. `width` and
`height` can only repeat the size of the checkpoint (256 or 512). Guidance is used only when
`cfg_scale` is above 1.0. With `cfg_interval` of `n` (0 or larger), guidance is dropped for every
token after the first `n + 2` generated tokens; -1 keeps it for all. `top_k` of 0 and `top_p` of
1.0 disable the respective filter. Unrecognized settings raise an error.

The seed starts a CPU generator that supplies every random draw, in the same order as the
reference after `torch.manual_seed(seed)` immediately before sampling, so the same seed gives the
same tokens on every device up to numerical differences. Images have a batch size of one.

**Prompt handling.** The prompt is lower-cased and stripped and encoded to 120 tokens (longer
prompts are truncated), which is what the reference's text embedder does when its text
preprocessing is switched off. The reference enables that preprocessing by default: it also
removes URLs, HTML, some punctuation, file names and similar noise with the `ftfy` and
BeautifulSoup packages. imagehub does not implement this caption cleaning, so a prompt that
contains such text can give different tokens from the default reference script; plain prompts give
identical tokens.

**Memory.** In float32 the GPT needs about 3.1 GB, the Flan-T5-XL encoder about 4.9 GB (the
decoder in the downloaded shards is never loaded) and the VQ decoder about 0.3 GB. The download
holds the full Flan-T5-XL checkpoint, about 11 GB.

**Seed compatibility.** On CPU, the same seed and settings give the same token ids and the same
image as the official `sample_t2i.py` driver run on CPU in float32 ([validation](validation.md)).
MPS and CUDA runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** The class-conditional (ImageNet) checkpoints, the caption cleaning described
above, `torch.compile`, fp16/bf16 parity, several prompts or images per call, and caller-supplied
tokens or caption embeddings are not supported.

**License.** The repository declares the MIT license and the text encoder is Apache-2.0.
imagehub does not distribute weights.
