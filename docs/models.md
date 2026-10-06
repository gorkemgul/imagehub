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
`width`/`height` (the U-Net config's native size, 1024 here, and multiples of 8), `negative_prompt`
(None; with an empty negative prompt, zero embeddings are used), `num_images` (1), `seed`,
`latents`, and the micro-conditioning inputs `original_size`, `crops_coords_top_left` and
`target_size` (default to the generated size and `(0, 0)`). Unrecognized settings raise an error.

**Seed compatibility.** On CPU, the same seed gives the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). As with SD 1.x,
MPS and CUDA runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** There is no safety checker, and the optional image watermark is off. The SDXL
refiner model, img2img, other schedulers and fp16 parity are not in this release.

**License.** The weights are licensed under CreativeML Open RAIL++-M; the license's use
restrictions bind whoever downloads and uses the weights. imagehub does not distribute weights.


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


# Stable Diffusion 3.5

Catalog: [`stable-diffusion-3`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Models: `sd3.5-medium`, `sd3.5-large`, `sd3.5-large-turbo` · Task: `t2i`

**Architecture.** Three frozen text encoders—CLIP-L, CLIP-G and T5-XXL—produce a joint sequence
of 77 CLIP tokens and 256 T5 tokens plus a 2048-wide pooled CLIP embedding. The medium checkpoint
uses MMDiT-X, with dual attention in its first 13 blocks; the large and large-turbo checkpoints use
the same 38-block MMDiT architecture. Large-turbo is distilled for 4 steps without guidance. A
16-channel VAE with a shift factor decodes the latents. The flow-match Euler scheduler uses shift 3.

**Usage.** All three repositories are gated. Accept the repository terms and provide `HF_TOKEN` when
downloading:

```bash
HF_TOKEN=... python -m imagehub.pretrained sd3.5-medium checkpoints/sd3.5-medium
```

```python
from imagehub import pipeline
with pipeline("t2i", backend="sd3.5-medium", model="checkpoints/sd3.5-medium", device="mps") as pipe:
    result = pipe("a cat, oil painting", seed=42, negative_prompt="blurry")
    result.images[0].save("cat.png")
```

**Settings.** Medium defaults to 40 steps and guidance 4.5; large defaults to 28 steps and
guidance 3.5; large-turbo defaults to 4 steps and guidance 0.0. All three default to 1024×1024,
and `width`/`height` must be multiples of 16. The other supported options are `negative_prompt`,
`num_images`, `seed`, `latents` and `max_sequence_length` (256). Unrecognized settings raise an
error.

**Seed compatibility.** On CPU, the same seed and settings give the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** There is no safety checker or invisible watermark. Img2img, other schedulers and
fp16/bf16 parity are not in this release.

**License.** The weights use the Stability AI Community License; see the license. imagehub does
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
Models: `z-image-turbo` · Task: `t2i`

**Architecture.** A Qwen3-4B text encoder renders the prompt through its chat template and supplies
penultimate hidden states. A 6B single-stream DiT uses 3-axis RoPE and refiner blocks. The
FLUX-style VAE has 16 latent channels and a shift factor. The flow-match Euler scheduler uses shift
3. The native image size is 1024×1024.

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

**Settings.** `steps` defaults to 9, `guidance` defaults to 0.0 and `width`/`height` default to
1024 and must be multiples of 16. The supported options are `negative_prompt`, `num_images`,
`seed`, `latents`, `cfg_normalization`, `cfg_truncation` and `max_sequence_length`. Unrecognized
settings raise an error.

**Seed compatibility.** On CPU, the same seed gives the same image as diffusers with
`generator=torch.Generator("cpu").manual_seed(seed)` ([validation](validation.md)). MPS and CUDA
runs are smoke tests only; there is no claim of exact equality on these devices.

**Out of scope.** There is no safety checker or invisible watermark. Omni, edit, controlnet, inpaint,
img2img, LoRA, the non-turbo checkpoint and fp16/bf16 parity are not in this release.

**License.** The weights are licensed under Apache-2.0. imagehub does not distribute weights.
