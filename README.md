# imagehub

An inference library for open-weight image generation architectures, natively reimplemented in
PyTorch behind a single API. **PyTorch is the only required runtime dependency.** A model is added
only once it is shown to reproduce the original implementation's output
([validation](docs/validation.md)). On CPU in float32, outputs are bit-identical to diffusers;
the details are on the [validation](docs/validation.md) page.
Architectures are linked to the
[awesome-text-to-image-architectures](https://github.com/gorkemgul/awesome-text-to-image-architectures)
catalog.

![sd-v1-5, seed 42](https://raw.githubusercontent.com/gorkemgul/imagehub/main/artifacts/sd-v1-5-seed42.png)

## Installation

```bash
pip install imagehub            # torch only
pip install "imagehub[image]"   # adds Pillow for Image.to_pil()
```

## Usage

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

Downloads happen only through `imagehub.pretrained`, and only when you run it.

- The validated commit is pinned.
- Every file is checked against its size. Weight files are also checked against the sha256 the
  Hub publishes, and all other files against their git blob sha1 (`blobId`).
- An `imagehub_snapshot.json` manifest records what was downloaded.

`pipeline()` never accesses the network.

## Models

| Family | Checkpoints | Task | Validation |
| --- | --- | --- | --- |
| Stable Diffusion 1.x | `sd-v1-1`, `sd-v1-2`, `sd-v1-3`, `sd-v1-4`, `sd-v1-5` | `t2i` | CPU fp32 parity |

[Catalog](docs/catalog.md) · [Model pages](docs/models.md) · [Validation](docs/validation.md) ·
[Third-party notices](THIRD_PARTY_NOTICES.md)

## Development

```bash
uv venv -p 3.13 .venv && uv pip install --python .venv -e ".[dev,image]"
.venv/bin/python -m pytest
```

Reference comparisons are optional and never download anything:

```bash
uv pip install --python .venv -e ".[dev,image,reference]"
IMAGEHUB_REFERENCE_TESTS=1 IMAGEHUB_SD_V1_5=checkpoints/sd-v1-5 .venv/bin/python -m pytest tests/reference
```

## License

The code is licensed under Apache-2.0. Model weights are subject to their own licenses and are not
distributed with imagehub.
