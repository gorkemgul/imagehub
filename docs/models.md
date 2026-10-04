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


# Stable Diffusion XL

Catalog: [`sdxl`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `sdxl-base-1.0` · Task: `t2i`

**Architecture.** Two frozen text encoders: CLIP ViT-L/14 (penultimate layer output) and OpenCLIP
bigG (penultimate layer output and pooled output). The concatenated outputs condition a
2.6B-parameter U-Net with `text_time` micro-conditioning: the original size, the crop position and
the target size of the image, encoded as a 6-element vector per sample. The VAE is an f=8 KL
autoencoder with a scaling factor of 0.13025 (1024×1024×3 image ↔ 128×128×4 latent). The
scheduler is the Euler published with the checkpoint (`timestep_spacing="leading"`,
`steps_offset=1`): 50 steps perform 50 U-Net evaluations.

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

**Settings.** `steps` (50, at most 1000), `guidance` (5.0; 1 and below turns guidance off),
`width`/`height` (1024, multiples of 8), `negative_prompt` (None; with an empty negative prompt,
zero embeddings are used), `num_images` (1), `seed`, `latents`, and the micro-conditioning inputs
`original_size`, `crops_coords_top_left` and `target_size` (default to the generated size and
`(0, 0)`). Unrecognized settings raise an error.

**Seed compatibility.** On CPU, the same seed gives the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). As with SD 1.x,
MPS and CUDA runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** There is no safety checker, and the optional image watermark is off. The SDXL
refiner model, img2img, other schedulers and fp16 parity are not in this release.

**License.** The weights are licensed under CreativeML Open RAIL++-M; the license's use
restrictions bind whoever downloads and uses the weights. imagehub does not distribute weights.
