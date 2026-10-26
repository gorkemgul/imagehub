# Installation

imagehub requires Python 3.11 or newer. PyTorch is the only required dependency.

```bash
pip install imagehub
```

| Extra | Adds | Needed for |
| --- | --- | --- |
| `image` | Pillow | `Image.to_pil()` and passing PIL images as input |
| `reference` | diffusers, transformers, torchvision, sentencepiece, protobuf | running the optional parity tests against the reference |
| `dev` | pytest, ruff, mypy, build | working on imagehub itself |
| `docs` | MkDocs Material, static i18n | building this site |

```bash
pip install "imagehub[image]"
pip install "imagehub[docs]"
```

PNG saving and reading is built in, so `Image.save("out.png")` works without Pillow.

!!! tip "Install PyTorch first for your hardware"
    If you want CUDA or a CPU-only build, install PyTorch from the
    [official selector](https://pytorch.org/get-started/locally/) before `pip install imagehub`.

## Check the install

```python
import imagehub
print(imagehub.__version__)
print(len(imagehub.available_models()["t2i"]), "models in the catalog")
```
