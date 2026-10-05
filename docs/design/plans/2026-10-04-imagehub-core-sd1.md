# imagehub Core + Stable Diffusion 1.x Implementation Plan

**Goal:** Build imagehub's core infrastructure (catalog, registry, verified downloads, strict checkpoint loading, pipeline API) and a native PyTorch Stable Diffusion 1.x that reproduces diffusers' output on CPU fp32 for `sd-v1-1` … `sd-v1-5`, released as `imagehub 0.1.0`.

**Architecture:** A flat package layout (`src/imagehub/` with `models/`, `backends/`, `pipelines/`, a hand-written `catalog.py` and a lazy `registry.py`). Model modules keep the exact tensor names of the published diffusers/transformers checkpoints, so weights load strictly with no conversion. Parity is proven with opt-in reference tests against diffusers 0.40.0 / transformers 5.18.0.

**Tech Stack:** Python ≥ 3.11, PyTorch ≥ 2.2 (only runtime dependency), hatchling, pytest, ruff, mypy, uv (dev environment only).

**Spec:** [docs/design/2026-10-04-imagehub-core-sd1-design.md](../2026-10-04-imagehub-core-sd1-design.md)

## Global Constraints

- Runtime dependencies: exactly `torch>=2.2`. Never import numpy, PIL, safetensors, diffusers or transformers from `src/imagehub` (Pillow only inside `Image.to_pil()`).
- `requires-python = ">=3.11"`; ruff `line-length = 100`, lint rules `E, F, I, UP, B`; mypy `disallow_untyped_defs = true` on `src/imagehub`.
- License: Apache-2.0. Package name `imagehub`, version `0.1.0`.
- Reference extra pins: `diffusers==0.40.0`, `transformers==5.18.0` (verified installable on 2026-10-04 with Python 3.13).
- Normal tests never touch the network. Reference tests run only with `IMAGEHUB_REFERENCE_TESTS=1` and never download.
- Model code must not branch on device type. Noise is generated on CPU. No float64 in model or scheduler code.
- Downloads: HTTPS only, pinned 40-char commit, sha256/size verified, `Authorization` stripped on cross-host redirects, never download `.py`, pickles loaded only with `weights_only=True`.
- Checkpoint loading is strict: missing, unexpected or mis-shaped tensors raise `CheckpointError`; nothing is randomly initialized.
- Commit messages are in English and carry no Co-Authored-By or other AI attribution trailer.
- All prototype code in this plan was verified on 2026-10-04: with random weights at full SD 1.x size, text encoder, U-Net, VAE, PNDM and the full sampling loop matched diffusers/transformers with a maximum difference of exactly 0. Every one of the 5 checkpoints has the same 197/686/248 tensor names. Copy code verbatim; do not "improve" numerics.

## File Map

```
pyproject.toml                         Task 1
.gitignore                             Task 1
LICENSE                                Task 1
src/imagehub/__init__.py               Task 1 (grows in Tasks 2, 4, 11, 12)
src/imagehub/exceptions.py             Task 1
src/imagehub/image.py                  Task 2
src/imagehub/types.py                  Task 2
src/imagehub/checkpoints.py            Task 3
src/imagehub/catalog.py                Task 4
src/imagehub/registry.py               Task 4
src/imagehub/download.py               Task 5
src/imagehub/pretrained.py             Task 5
src/imagehub/tokenization.py           Task 6
src/imagehub/models/__init__.py        Task 7
src/imagehub/models/layers.py          Task 7
src/imagehub/models/clip_text.py       Task 7
src/imagehub/models/autoencoder_kl.py  Task 8
src/imagehub/models/unet_2d.py         Task 9
src/imagehub/schedulers/__init__.py    Task 10
src/imagehub/schedulers/pndm.py        Task 10
src/imagehub/backends/__init__.py      Task 11
src/imagehub/backends/base.py          Task 11
src/imagehub/pipelines/__init__.py     Task 11
src/imagehub/pipelines/base.py         Task 11
src/imagehub/pipelines/text_to_image.py Task 11
src/imagehub/factory.py                Task 11
src/imagehub/backends/stable_diffusion.py Task 12
tests/conftest.py                      Task 1
tests/helpers.py                       Tasks 6, 7, 12
tests/fixtures/sd1_tensor_shapes.json  Task 7 (copied from docs/design/plans/assets/)
tests/test_*.py                        Tasks 1–12
tests/reference/conftest.py            Task 7
tests/reference/test_ref_*.py          Tasks 6–10, 12, 13
scripts/build_validation.py            Task 14
docs/validation.json, docs/validation.md Task 14
README.md, docs/catalog.md, docs/models.md, THIRD_PARTY_NOTICES.md, licenses/ Task 15
```

---

### Task 1: Project scaffold and exceptions

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `LICENSE`, `src/imagehub/__init__.py`, `src/imagehub/exceptions.py`, `tests/conftest.py`, `tests/test_package.py`

**Interfaces:**
- Produces: `imagehub.__version__ == "0.1.0"`; exceptions `ImageHubError`, `MissingDependencyError(ImageHubError, ImportError)`, `BackendBusyError(ImageHubError, RuntimeError)`, `BackendContractError(ImageHubError, TypeError)`, `CheckpointError(ImageHubError, ValueError)`, `DownloadError(ImageHubError, RuntimeError)`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "imagehub"
version = "0.1.0"
description = "Open-weight image generation architectures, natively reimplemented in PyTorch with parity evidence"
readme = "README.md"
requires-python = ">=3.11"
license = "Apache-2.0"
license-files = ["LICENSE"]
authors = [{ name = "gorkemgul" }]
dependencies = ["torch>=2.2"]

[project.optional-dependencies]
image = ["pillow>=10"]
reference = ["diffusers==0.40.0", "transformers==5.18.0"]
dev = ["pytest>=8", "pytest-cov>=5", "ruff>=0.9", "mypy>=1.13", "build>=1.2"]

[tool.hatch.build.targets.wheel]
packages = ["src/imagehub"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]
addopts = "-ra"
markers = [
  "reference: optional comparison against upstream implementations",
  "mps: requires Apple MPS",
  "cuda: requires CUDA",
]

[tool.ruff]
target-version = "py311"
line-length = 100
# ruff >= 0.16 also formats Python blocks inside Markdown; docs keep their own layout.
extend-exclude = ["*.md"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
files = ["src/imagehub"]
check_untyped_defs = true
disallow_untyped_defs = true
ignore_missing_imports = true
warn_unused_ignores = true
```

- [ ] **Step 2: Write `.gitignore` and `LICENSE`**

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
dist/
build/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
/checkpoints/
```

`LICENSE` (Apache 2.0 full text):
```bash
curl -fsSL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE && head -3 LICENSE
```
Expected: first lines show `Apache License` / `Version 2.0, January 2004`.

- [ ] **Step 3: Create the dev environment**

```bash
uv venv -p 3.13 .venv
uv pip install --python .venv -e ".[dev,image]"
```
This fails until Step 5 creates the package; run it after Step 5 if needed.

- [ ] **Step 4: Write the failing test `tests/test_package.py`**

```python
import imagehub
from imagehub.exceptions import (
    BackendBusyError,
    BackendContractError,
    CheckpointError,
    DownloadError,
    ImageHubError,
    MissingDependencyError,
)


def test_version():
    assert imagehub.__version__ == "0.1.0"


def test_exception_hierarchy():
    assert issubclass(MissingDependencyError, ImageHubError)
    assert issubclass(MissingDependencyError, ImportError)
    assert issubclass(BackendBusyError, RuntimeError)
    assert issubclass(BackendContractError, TypeError)
    assert issubclass(CheckpointError, ValueError)
    assert issubclass(DownloadError, RuntimeError)
    for error in (BackendBusyError, BackendContractError, CheckpointError, DownloadError):
        assert issubclass(error, ImageHubError)
```

`tests/conftest.py`:
```python
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
```

- [ ] **Step 5: Write `src/imagehub/exceptions.py` and `src/imagehub/__init__.py`**

```python
"""Errors callers can handle without depending on a particular model runtime."""


class ImageHubError(Exception):
    """Base class for library-specific errors."""


class MissingDependencyError(ImageHubError, ImportError):
    """An optional dependency is not installed."""


class BackendBusyError(ImageHubError, RuntimeError):
    """A backend was unloaded or re-entered while an inference was running."""


class BackendContractError(ImageHubError, TypeError):
    """An adapter returned a value that violates its task's contract."""


class CheckpointError(ImageHubError, ValueError):
    """A checkpoint is missing, corrupt, or does not match the model architecture."""


class DownloadError(ImageHubError, RuntimeError):
    """A model download failed verification or could not be completed safely."""
```

`src/imagehub/__init__.py`:
```python
"""imagehub: open-weight image generation architectures in native PyTorch."""

__version__ = "0.1.0"

__all__ = ["__version__"]
```

- [ ] **Step 6: Install and run tests**

```bash
uv pip install --python .venv -e ".[dev,image]"
.venv/bin/python -m pytest tests/test_package.py -v
```
Expected: 2 passed.

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
git add pyproject.toml .gitignore LICENSE src tests
git commit -m "Scaffold imagehub package and error types"
```
`ruff format` rewrites long lines in this plan's code; that is expected.

---

### Task 2: Image type (stdlib PNG) and request/result types

**Files:**
- Create: `src/imagehub/image.py`, `src/imagehub/types.py`, `tests/test_image.py`, `tests/test_types.py`
- Modify: `src/imagehub/__init__.py`

**Interfaces:**
- Consumes: `MissingDependencyError` (Task 1).
- Produces:
  - `Image(tensor: Tensor)` — uint8 `(H, W, 3|4)`; attributes `.tensor`, `.width`, `.height`, `.channels`; methods `.to_png() -> bytes`, `.save(path) -> Path` (`.png` only), `.to_pil()`, classmethods `Image.from_png(data: bytes)`, `Image.open(path)`.
  - `Task` (`StrEnum`, member `T2I = "t2i"`), `T2IRequest(prompt: str, parameters: Mapping[str, Any] = {})`, `ImageResult(images: tuple[Image, ...], metadata: Mapping[str, Any] = {})`.

- [ ] **Step 1: Write failing tests `tests/test_image.py`**

```python
import io
import sys

import pytest
import torch

from imagehub.exceptions import MissingDependencyError
from imagehub.image import Image


def random_image(h=5, w=7, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (h, w, c), generator=g, dtype=torch.uint8)


@pytest.mark.parametrize("channels", [3, 4])
def test_png_roundtrip(tmp_path, channels):
    pixels = random_image(c=channels)
    path = Image(pixels).save(tmp_path / "x.png")
    restored = Image.open(path)
    assert torch.equal(restored.tensor, pixels)
    assert (restored.width, restored.height, restored.channels) == (7, 5, channels)


def test_rejects_invalid_tensors():
    with pytest.raises(TypeError):
        Image(torch.zeros(4, 4, 3))
    with pytest.raises(ValueError):
        Image(torch.zeros(4, 4, 2, dtype=torch.uint8))
    with pytest.raises(ValueError):
        Image(torch.zeros(4, 4, dtype=torch.uint8))


def test_save_requires_png(tmp_path):
    with pytest.raises(ValueError, match="png"):
        Image(random_image()).save(tmp_path / "x.jpg")


def test_from_png_rejects_corruption():
    data = bytearray(Image(random_image()).to_png())
    with pytest.raises(ValueError, match="PNG"):
        Image.from_png(b"not a png")
    data[20] ^= 0xFF  # inside IHDR body -> CRC mismatch
    with pytest.raises(ValueError, match="CRC"):
        Image.from_png(bytes(data))


def test_reads_pillow_filtered_png_and_pillow_reads_ours():
    PIL = pytest.importorskip("PIL.Image")
    pixels = random_image(h=33, w=29, c=3, seed=1)
    gradient = torch.arange(33 * 29 * 3, dtype=torch.int64).reshape(33, 29, 3) % 256
    for source in (pixels, gradient.to(torch.uint8)):
        buffer = io.BytesIO()
        PIL.frombytes("RGB", (29, 33), bytes(source.flatten().tolist())).save(buffer, "PNG")
        assert torch.equal(Image.from_png(buffer.getvalue()).tensor, source)
        decoded = PIL.open(io.BytesIO(Image(source).to_png())).convert("RGB")
        assert bytes(decoded.tobytes()) == bytes(source.flatten().tolist())


def test_to_pil_without_pillow(monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(MissingDependencyError, match=r"imagehub\[image\]"):
        Image(random_image()).to_pil()
```

`tests/test_types.py`:
```python
import pytest
import torch

from imagehub.image import Image
from imagehub.types import ImageResult, T2IRequest, Task


def test_task_values():
    assert Task("t2i") is Task.T2I


def test_request_validation_and_copy():
    params = {"seed": 1}
    request = T2IRequest("a cat", params)
    params["seed"] = 2
    assert request.parameters == {"seed": 1}
    with pytest.raises(ValueError):
        T2IRequest("   ")


def test_result_validation():
    image = Image(torch.zeros(2, 2, 3, dtype=torch.uint8))
    result = ImageResult([image], {"seed": 3})
    assert result.images == (image,)
    with pytest.raises(ValueError):
        ImageResult([])
    with pytest.raises(ValueError):
        ImageResult([torch.zeros(2, 2, 3)])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_image.py tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imagehub.image'`.

- [ ] **Step 3: Write `src/imagehub/image.py`**

```python
"""Image container with dependency-free PNG encoding and decoding."""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imagehub.exceptions import MissingDependencyError

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {2: 3, 6: 4}  # PNG color type -> channels (8-bit RGB, 8-bit RGBA)


def _chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _unfilter(raw: bytes, width: int, height: int, channels: int) -> bytearray:
    stride = width * channels
    out = bytearray(height * stride)
    previous = bytearray(stride)
    pos = 0
    for y in range(height):
        kind = raw[pos]
        line = bytearray(raw[pos + 1 : pos + 1 + stride])
        pos += stride + 1
        if kind == 1:
            for x in range(channels, stride):
                line[x] = (line[x] + line[x - channels]) & 0xFF
        elif kind == 2:
            for x in range(stride):
                line[x] = (line[x] + previous[x]) & 0xFF
        elif kind == 3:
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                line[x] = (line[x] + ((left + previous[x]) >> 1)) & 0xFF
        elif kind == 4:
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                upper_left = previous[x - channels] if x >= channels else 0
                line[x] = (line[x] + _paeth(left, previous[x], upper_left)) & 0xFF
        elif kind != 0:
            raise ValueError(f"Unsupported PNG filter type {kind}")
        out[y * stride : (y + 1) * stride] = line
        previous = line
    return out


class Image:
    """An 8-bit RGB or RGBA image stored as a (height, width, channels) uint8 tensor."""

    __slots__ = ("tensor",)

    def __init__(self, tensor: Tensor) -> None:
        if not isinstance(tensor, Tensor) or tensor.dtype != torch.uint8:
            raise TypeError("Image expects a uint8 tensor")
        if tensor.ndim != 3 or tensor.shape[2] not in (3, 4) or min(tensor.shape[:2]) < 1:
            raise ValueError("Image tensor must have shape (height, width, 3 or 4)")
        self.tensor = tensor.detach().cpu().contiguous()

    @property
    def height(self) -> int:
        return int(self.tensor.shape[0])

    @property
    def width(self) -> int:
        return int(self.tensor.shape[1])

    @property
    def channels(self) -> int:
        return int(self.tensor.shape[2])

    def _bytes(self) -> bytes:
        return bytes(self.tensor.flatten().tolist())

    def to_png(self) -> bytes:
        h, w, c = self.height, self.width, self.channels
        rows = self.tensor.reshape(h, w * c)
        raw = torch.cat([torch.zeros(h, 1, dtype=torch.uint8), rows], dim=1)
        header = struct.pack(">IIBBBBB", w, h, 8, 2 if c == 3 else 6, 0, 0, 0)
        return (
            _SIGNATURE
            + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(bytes(raw.flatten().tolist()), 6))
            + _chunk(b"IEND", b"")
        )

    def save(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        if target.suffix.lower() != ".png":
            raise ValueError("Only .png output is supported; use to_pil() for other formats")
        target.write_bytes(self.to_png())
        return target

    @classmethod
    def from_png(cls, data: bytes) -> Image:
        if not data.startswith(_SIGNATURE):
            raise ValueError("Not a PNG file")
        pos, header, idat = len(_SIGNATURE), None, bytearray()
        while pos + 8 <= len(data):
            length, kind = struct.unpack(">I4s", data[pos : pos + 8])
            body = data[pos + 8 : pos + 8 + length]
            crc = data[pos + 8 + length : pos + 12 + length]
            if len(body) != length or len(crc) != 4:
                raise ValueError("Truncated PNG chunk")
            if zlib.crc32(kind + body) & 0xFFFFFFFF != struct.unpack(">I", crc)[0]:
                raise ValueError("PNG chunk CRC mismatch")
            pos += 12 + length
            if kind == b"IHDR":
                header = struct.unpack(">IIBBBBB", body)
            elif kind == b"IDAT":
                idat += body
            elif kind == b"IEND":
                break
        if header is None:
            raise ValueError("PNG is missing its IHDR chunk")
        width, height, depth, color, compression, filtering, interlace = header
        if depth != 8 or color not in _CHANNELS or compression or filtering or interlace:
            raise ValueError("Only 8-bit non-interlaced RGB/RGBA PNG files are supported")
        channels = _CHANNELS[color]
        raw = zlib.decompress(bytes(idat))
        if len(raw) != height * (width * channels + 1):
            raise ValueError("PNG image data has the wrong size")
        pixels = _unfilter(raw, width, height, channels)
        tensor = torch.frombuffer(pixels, dtype=torch.uint8).reshape(height, width, channels)
        return cls(tensor.clone())

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> Image:
        return cls.from_png(Path(path).read_bytes())

    def to_pil(self) -> Any:
        try:
            from PIL import Image as PILImage
        except ImportError as error:
            raise MissingDependencyError(
                "Image.to_pil() requires Pillow; install it with `pip install imagehub[image]`"
            ) from error
        mode = "RGB" if self.channels == 3 else "RGBA"
        return PILImage.frombytes(mode, (self.width, self.height), self._bytes())

    def __repr__(self) -> str:
        return f"Image(width={self.width}, height={self.height}, channels={self.channels})"
```

- [ ] **Step 4: Write `src/imagehub/types.py`**

```python
"""Shared task identifiers, requests and results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from imagehub.image import Image


class Task(StrEnum):
    T2I = "t2i"


@dataclass(frozen=True, slots=True)
class T2IRequest:
    prompt: str
    # Request-level generation settings. Options passed to a pipeline call override them.
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        object.__setattr__(self, "parameters", dict(self.parameters))


@dataclass(frozen=True, slots=True)
class ImageResult:
    images: tuple[Image, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        images = tuple(self.images)
        if not images or not all(isinstance(image, Image) for image in images):
            raise ValueError("images must be a nonempty sequence of Image objects")
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "metadata", dict(self.metadata))
```

- [ ] **Step 5: Export from `src/imagehub/__init__.py`**

```python
"""imagehub: open-weight image generation architectures in native PyTorch."""

from imagehub.image import Image
from imagehub.types import ImageResult, T2IRequest, Task

__version__ = "0.1.0"

__all__ = ["Image", "ImageResult", "T2IRequest", "Task", "__version__"]
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests -v`
Expected: all pass (Pillow test runs because the `image` extra is installed).

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
git add src tests
git commit -m "Add Image type with stdlib PNG codec and t2i request/result types"
```

---

### Task 3: Strict checkpoint I/O

**Files:**
- Create: `src/imagehub/checkpoints.py`, `tests/test_checkpoints.py`

**Interfaces:**
- Consumes: `CheckpointError` (Task 1).
- Produces:
  - `read_json(path) -> dict[str, Any]`
  - `load_safetensors(path) -> dict[str, Tensor]`, `save_safetensors(tensors: Mapping[str, Tensor], path) -> None`
  - `load_weights(path) -> dict[str, Tensor]` — file (`.safetensors`, or tensor-only pickle via `torch.load(weights_only=True)`) or directory (sharded index, then `WEIGHT_FILES` in order)
  - `WEIGHT_FILES = ("model.safetensors", "diffusion_pytorch_model.safetensors", "model.pt", "pytorch_model.bin", "diffusion_pytorch_model.bin")`
  - `class NativeModel(nn.Module)`: `__init__(config: Mapping)` stores `self.config: dict`; `prepare_state_dict(state) -> state`; `load_checkpoint(state) -> None` (strict, raises `CheckpointError`); `classmethod from_pretrained(directory, *, device="cpu", dtype=None) -> Self` (reads `config.json`, builds on meta, strict-loads, moves, `.eval()`).

- [ ] **Step 1: Write failing tests `tests/test_checkpoints.py`**

```python
import json
import struct

import pytest
import torch
from torch import nn

from imagehub.checkpoints import (
    NativeModel,
    load_safetensors,
    load_weights,
    save_safetensors,
)
from imagehub.exceptions import CheckpointError


class TinyModel(NativeModel):
    def __init__(self, config):
        super().__init__({"width": 3} | dict(config))
        self.linear = nn.Linear(self.config["width"], 2)

    def prepare_state_dict(self, state):
        state = dict(state)
        state.pop("legacy.buffer", None)
        return state


def tensors():
    return {
        "f32": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "bf16": torch.ones(4, dtype=torch.bfloat16),
        "i64": torch.tensor([1, -2], dtype=torch.int64),
        "flag": torch.tensor([True, False]),
        "empty": torch.zeros(0, 5),
    }


def test_safetensors_roundtrip(tmp_path):
    path = tmp_path / "w.safetensors"
    save_safetensors(tensors(), path)
    restored = load_safetensors(path)
    assert set(restored) == set(tensors())
    for name, value in tensors().items():
        assert restored[name].dtype == value.dtype
        torch.testing.assert_close(restored[name], value, rtol=0, atol=0)


def _raw(header: dict, payload: bytes) -> bytes:
    blob = json.dumps(header).encode()
    return struct.pack("<Q", len(blob)) + blob + payload


@pytest.mark.parametrize(
    ("header", "payload", "message"),
    [
        ({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, b"\0" * 4, "offsets"),
        ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, b"\0" * 8, "trailing"),
        (
            {
                "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                "b": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            },
            b"\0" * 4,
            "overlapping",
        ),
        ({"a": {"dtype": "X9", "shape": [1], "data_offsets": [0, 4]}}, b"\0" * 4, "descriptor"),
    ],
)
def test_safetensors_rejects_corruption(tmp_path, header, payload, message):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(_raw(header, payload))
    with pytest.raises(CheckpointError, match=message):
        load_safetensors(path)


def test_safetensors_rejects_bad_header_length(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", 10**9) + b"{}")
    with pytest.raises(CheckpointError, match="header length"):
        load_safetensors(path)


def test_load_weights_directory_order_and_bin(tmp_path):
    torch.save({"x": torch.ones(1)}, tmp_path / "diffusion_pytorch_model.bin")
    assert torch.equal(load_weights(tmp_path)["x"], torch.ones(1))
    save_safetensors({"x": torch.zeros(1)}, tmp_path / "diffusion_pytorch_model.safetensors")
    assert torch.equal(load_weights(tmp_path)["x"], torch.zeros(1))
    with pytest.raises(CheckpointError, match="No supported checkpoint"):
        load_weights(tmp_path / "missing_dir_is_file_not_dir")


class Unsafe:
    pass


def test_load_weights_refuses_arbitrary_pickles(tmp_path):
    torch.save({"x": torch.ones(1), "obj": Unsafe()}, tmp_path / "model.pt")
    with pytest.raises(CheckpointError, match="tensor-only"):
        load_weights(tmp_path / "model.pt")


def test_strict_loading_reports_mismatches():
    model = TinyModel({})
    good = {"linear.weight": torch.zeros(2, 3), "linear.bias": torch.zeros(2)}
    model.load_checkpoint(good | {"legacy.buffer": torch.zeros(1)})
    with pytest.raises(CheckpointError, match="Missing 1 tensors: linear.bias"):
        model.load_checkpoint({"linear.weight": torch.zeros(2, 3)})
    with pytest.raises(CheckpointError, match="Unexpected 1 tensors: extra"):
        model.load_checkpoint(good | {"extra": torch.zeros(1)})
    with pytest.raises(CheckpointError, match="shapes"):
        model.load_checkpoint(good | {"linear.bias": torch.zeros(3)})


def test_from_pretrained(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"width": 4, "_class_name": "Tiny"}))
    weights = {"linear.weight": torch.randn(2, 4), "linear.bias": torch.randn(2)}
    save_safetensors(weights, tmp_path / "model.safetensors")
    model = TinyModel.from_pretrained(tmp_path, dtype=torch.float64)
    assert not model.training
    assert model.linear.weight.dtype == torch.float64
    assert model.config["width"] == 4
    torch.testing.assert_close(model.linear.weight.float(), weights["linear.weight"])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_checkpoints.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imagehub.checkpoints'`.

- [ ] **Step 3: Write `src/imagehub/checkpoints.py`**

```python
"""Local, strict checkpoint loading without transformers, diffusers or safetensors."""

from __future__ import annotations

import ctypes
import json
import math
import os
import pickle
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor, nn

from imagehub.exceptions import CheckpointError

_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
_NAMES = {dtype: name for name, dtype in _DTYPES.items()}
WEIGHT_FILES = (
    "model.safetensors",
    "diffusion_pytorch_model.safetensors",
    "model.pt",
    "pytorch_model.bin",
    "diffusion_pytorch_model.bin",
)
_INDEX_FILES = (
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "diffusion_pytorch_model.bin.index.json",
)


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CheckpointError(f"Expected a JSON object in {path}")
    return value


def load_safetensors(path: str | os.PathLike[str]) -> dict[str, Tensor]:
    """Read the documented safetensors format using byte buffers and torch.frombuffer."""
    result: dict[str, Tensor] = {}
    with Path(path).open("rb") as source:
        size = os.fstat(source.fileno()).st_size
        prefix = source.read(8)
        if len(prefix) != 8:
            raise CheckpointError("Truncated safetensors header")
        header_length = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_length <= min(100_000_000, size - 8):
            raise CheckpointError("Invalid safetensors header length")
        header = json.loads(source.read(header_length))
        if not isinstance(header, dict):
            raise CheckpointError("Invalid safetensors header")
        payload_start = 8 + header_length
        spans = []
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            dtype = _DTYPES.get(entry.get("dtype"))
            shape, offsets = entry.get("shape"), entry.get("data_offsets")
            if dtype is None or not isinstance(shape, list) or not isinstance(offsets, list):
                raise CheckpointError(f"Invalid tensor descriptor: {name}")
            if any(type(n) is not int or n < 0 for n in shape) or len(offsets) != 2:
                raise CheckpointError(f"Invalid tensor dimensions: {name}")
            start, end = offsets
            expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end - start != expected
                or payload_start + end > size
            ):
                raise CheckpointError(f"Invalid tensor offsets: {name}")
            spans.append((start, end, name, dtype, shape))
        cursor = 0
        for start, end, name, dtype, shape in sorted(spans):
            if start != cursor:
                raise CheckpointError("Safetensors payload has overlapping or missing data")
            source.seek(payload_start + start)
            buffer = bytearray(source.read(end - start))
            result[name] = (
                torch.frombuffer(buffer, dtype=dtype).reshape(shape)
                if buffer
                else torch.empty(shape, dtype=dtype)
            )
            cursor = end
        if payload_start + cursor != size:
            raise CheckpointError("Safetensors payload has unreferenced trailing data")
    return result


def save_safetensors(tensors: Mapping[str, Tensor], path: str | os.PathLike[str]) -> None:
    """Write tensors in the safetensors format (used for tests and converted snapshots)."""
    header: dict[str, Any] = {}
    payloads: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        if tensor.dtype not in _NAMES:
            raise TypeError(f"Unsupported dtype for safetensors: {tensor.dtype}")
        nbytes = tensor.numel() * tensor.element_size()
        raw = ctypes.string_at(tensor.data_ptr(), nbytes) if nbytes else b""
        header[name] = {
            "dtype": _NAMES[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        payloads.append(raw)
        offset += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with Path(path).open("wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for raw in payloads:
            out.write(raw)


def _load_pickle(source: Path) -> dict[str, Tensor]:
    try:
        state = torch.load(source, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as error:
        raise CheckpointError(
            f"{source} is not a tensor-only checkpoint and was not loaded: {error}"
        ) from error
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not all(
        isinstance(name, str) and isinstance(value, Tensor) for name, value in state.items()
    ):
        raise CheckpointError(f"{source} must contain a tensor-only state dictionary")
    return state


def load_weights(path: str | os.PathLike[str]) -> dict[str, Tensor]:
    """Read safetensors, tensor-only PT/BIN, or a (sharded) model directory."""
    source = Path(path)
    if source.is_dir():
        for name in _INDEX_FILES:
            index_path = source / name
            if index_path.is_file():
                index = read_json(index_path)["weight_map"]
                result: dict[str, Tensor] = {}
                for filename in sorted(set(index.values())):
                    shard = (source / filename).resolve()
                    if not shard.is_relative_to(source.resolve()):
                        raise CheckpointError("Checkpoint shard escapes its model directory")
                    values = load_weights(shard)
                    if result.keys() & values.keys():
                        raise CheckpointError("Duplicate tensor names across checkpoint shards")
                    result.update(values)
                if set(result) != set(index):
                    raise CheckpointError("Checkpoint index does not match its shards")
                return result
        for name in WEIGHT_FILES:
            if (source / name).is_file():
                return load_weights(source / name)
        raise CheckpointError(f"No supported checkpoint found in {source}")
    if not source.is_file():
        raise CheckpointError(f"No supported checkpoint found at {source}")
    if source.suffix == ".safetensors":
        return load_safetensors(source)
    return _load_pickle(source)


def _describe(missing: list[str], unexpected: list[str]) -> str:
    lines = ["Checkpoint does not match the model architecture."]
    if missing:
        lines.append(f"Missing {len(missing)} tensors: " + ", ".join(missing[:20]))
    if unexpected:
        lines.append(f"Unexpected {len(unexpected)} tensors: " + ", ".join(unexpected[:20]))
    return "\n".join(lines)


class NativeModel(nn.Module):
    """Base for native model definitions; checkpoints are always loaded strictly."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Override only for documented key conversion or constant buffers to drop."""
        return state

    def load_checkpoint(self, state: Mapping[str, Tensor]) -> None:
        prepared = self.prepare_state_dict(dict(state))
        expected = self.state_dict()
        missing = sorted(set(expected) - set(prepared))
        unexpected = sorted(set(prepared) - set(expected))
        if missing or unexpected:
            raise CheckpointError(_describe(missing, unexpected))
        mismatched = [
            f"{name}: checkpoint {tuple(prepared[name].shape)} != model {tuple(value.shape)}"
            for name, value in expected.items()
            if tuple(prepared[name].shape) != tuple(value.shape)
        ]
        if mismatched:
            raise CheckpointError(
                "Checkpoint tensor shapes do not match the model:\n  " + "\n  ".join(mismatched[:20])
            )
        self.load_state_dict(prepared, strict=True, assign=True)

    @classmethod
    def from_pretrained(
        cls,
        directory: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> Self:
        path = Path(directory)
        config = read_json(path / "config.json")
        with torch.device("meta"):
            model = cls(config)
        model.load_checkpoint(load_weights(path))
        model.to(device=device, dtype=dtype).eval()
        return model
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_checkpoints.py -v`
Expected: all pass. If `test_load_weights_directory_order_and_bin` fails on the last assertion, note that `load_weights(tmp_path / "missing_dir_is_file_not_dir")` must hit the `not source.is_file()` branch.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
git add src tests
git commit -m "Add strict safetensors/pickle checkpoint loading and NativeModel"
```

---

### Task 4: Model catalog and backend registry

**Files:**
- Create: `src/imagehub/catalog.py`, `src/imagehub/registry.py`, `tests/test_catalog.py`, `tests/test_registry.py`, `tests/test_catalog_link.py`
- Modify: `src/imagehub/__init__.py`

**Interfaces:**
- Consumes: `Task` (Task 2).
- Produces:
  - `ModelSpec` (frozen dataclass): `id, task, family, source, repo_id, files, format, catalog_id, revision, provenance, config={}, gated=False, license="See model card"`.
  - `MODEL_CATALOG: Mapping[str, ModelSpec]` with ids `sd-v1-1` … `sd-v1-5`; `model_info(model_id) -> ModelSpec` (raises `ValueError` listing ids); `available_models(task=None) -> dict[str, tuple[str, ...]]`.
  - `BackendRegistry` with `register(task, name, factory, *, replace=False)`, `names(task)`, `create(task, name, **options)`; global `backend_registry`; `available_backends(task=None)`. Family `"stable-diffusion"` → `"imagehub.backends.stable_diffusion:StableDiffusionBackend"` (module created in Task 12). Every catalog id is registered as an alias of its family factory.
  - The registry type-checks created backends against `imagehub.backends.base.Backend`, which Task 11 creates; `create()` imports it only after the name lookup, so Task 4's tests never import it.

- [ ] **Step 1: Write failing tests**

`tests/test_catalog.py`:
```python
import re

import pytest

from imagehub.catalog import MODEL_CATALOG, ModelSpec, available_models, model_info
from imagehub.types import Task


def test_catalog_contents():
    assert available_models() == {"t2i": ("sd-v1-1", "sd-v1-2", "sd-v1-3", "sd-v1-4", "sd-v1-5")}
    for spec in MODEL_CATALOG.values():
        assert spec.task is Task.T2I
        assert spec.family == "stable-diffusion"
        assert spec.catalog_id == "stable-diffusion-1"
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)
        assert spec.source == f"https://huggingface.co/{spec.repo_id}"
        assert spec.license == "CreativeML-OpenRAIL-M"
        assert len(set(spec.files)) == len(spec.files) == 12
        assert not any(name.endswith((".py", ".ckpt")) or "fp16" in name for name in spec.files)
        assert not any(name.startswith(("safety_checker/", "feature_extractor/")) for name in spec.files)


def test_provenance_and_formats():
    assert model_info("sd-v1-5").provenance == "community-mirror"
    assert model_info("sd-v1-5").revision == "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    for version in (1, 2, 3, 4):
        assert model_info(f"sd-v1-{version}").provenance == "author"
    assert "unet/diffusion_pytorch_model.bin" in model_info("sd-v1-1").files
    assert "unet/diffusion_pytorch_model.safetensors" in model_info("sd-v1-4").files


def test_unknown_model_lists_available():
    with pytest.raises(ValueError, match="sd-v1-4, sd-v1-5"):
        model_info("sd-v15")


def test_spec_validation():
    base = model_info("sd-v1-4")
    with pytest.raises(ValueError, match="revision"):
        ModelSpec(**{**base.__dict__, "revision": "main"})
    with pytest.raises(ValueError, match="provenance"):
        ModelSpec(**{**base.__dict__, "provenance": "unknown"})
```

`tests/test_registry.py`:
```python
import pytest

from imagehub.registry import BackendRegistry, available_backends


def test_family_and_model_aliases_are_registered():
    names = available_backends("t2i")["t2i"]
    assert "stable-diffusion" in names
    assert {"sd-v1-1", "sd-v1-5"} <= set(names)


def test_registration_rules():
    registry = BackendRegistry()
    registry.register("t2i", "x", "some.module:Factory")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("t2i", "x", "some.module:Factory")
    with pytest.raises(ValueError, match="module:class"):
        registry.register("t2i", "y", "no_colon")
    with pytest.raises(TypeError):
        registry.register("t2i", "z", 42)
    with pytest.raises(ValueError, match="Available: x"):
        registry.create("t2i", "missing")
```

`tests/test_catalog_link.py`:
```python
"""Checks catalog_id links against the sibling awesome-text-to-image-architectures checkout."""

import json
from pathlib import Path

import pytest

from imagehub.catalog import MODEL_CATALOG

CATALOG = Path(__file__).resolve().parents[2] / "awesome-text-to-image-architectures/data/models.json"


def _norm(value: str) -> str:
    return value.lower().replace("_", "-").replace(" ", "-")


def test_catalog_ids_exist_with_consistent_weight_licenses():
    if not CATALOG.is_file():
        pytest.skip(f"Catalog checkout not found at {CATALOG}")
    entries = {m["id"]: m for m in json.loads(CATALOG.read_text(encoding="utf-8"))["models"]}
    for spec in MODEL_CATALOG.values():
        assert spec.catalog_id in entries, spec.id
        weights = entries[spec.catalog_id]["license"]["weights"]
        assert _norm(weights) == _norm(spec.license), (spec.id, weights)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_catalog.py tests/test_registry.py tests/test_catalog_link.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imagehub.catalog'`.

- [ ] **Step 3: Write `src/imagehub/catalog.py`**

```python
"""Published, distinct model checkpoints, separate from reusable architecture backends.

A model is a released weight set, not an alias, precision variant or configuration.
Families share native code. ``catalog_id`` links each model to the entry in
awesome-text-to-image-architectures that describes its architecture, and ``revision``
is the commit whose files passed the parity tests in docs/validation.json.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from imagehub.types import Task

PROVENANCES = ("author", "community-mirror")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    task: Task
    family: str
    source: str
    repo_id: str | None
    files: tuple[str, ...]
    format: str
    catalog_id: str
    revision: str
    provenance: str
    config: Mapping[str, Any] = field(default_factory=dict)
    gated: bool = False
    license: str = "See model card"

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))
        object.__setattr__(self, "files", tuple(self.files))
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be a 40-character lowercase commit hash")
        if self.provenance not in PROVENANCES:
            raise ValueError(f"provenance must be one of {', '.join(PROVENANCES)}")


_entries: list[ModelSpec] = []

# Stable Diffusion 1.x: one architecture, five released weight sets. Only fp32 EMA weights
# are fetched; fp16, non-EMA, single-file .ckpt and safety checker files are excluded.
_SD1_COMMON = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
    "text_encoder/config.json",
    "unet/config.json",
    "vae/config.json",
)
_SD1_SAFETENSORS = (
    "text_encoder/model.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
)
# v1.1–v1.3 publish fp32 weights only as tensor-only pickles (loaded with weights_only=True).
_SD1_BIN = (
    "text_encoder/pytorch_model.bin",
    "unet/diffusion_pytorch_model.bin",
    "vae/diffusion_pytorch_model.bin",
)
for _version, _repo, _revision, _provenance, _weights in (
    ("1", "CompVis/stable-diffusion-v1-1", "574555a657f0c8d6e970a02c7e095e9d4220dcbf", "author", _SD1_BIN),
    ("2", "CompVis/stable-diffusion-v1-2", "f519eebf1372598b85ab317ec273a63ebeddd5eb", "author", _SD1_BIN),
    ("3", "CompVis/stable-diffusion-v1-3", "159f26fccae6e3ea26c797ad24c5e7033da5b1ee", "author", _SD1_BIN),
    (
        "4",
        "CompVis/stable-diffusion-v1-4",
        "133a221b8aa7292a167afc5127cb63fb5005638b",
        "author",
        _SD1_SAFETENSORS,
    ),
    (
        "5",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        "community-mirror",
        _SD1_SAFETENSORS,
    ),
):
    _entries.append(
        ModelSpec(
            id=f"sd-v1-{_version}",
            task=Task.T2I,
            family="stable-diffusion",
            source=f"https://huggingface.co/{_repo}",
            repo_id=_repo,
            files=_SD1_COMMON + _weights,
            format="diffusers",
            catalog_id="stable-diffusion-1",
            revision=_revision,
            provenance=_provenance,
            license="CreativeML-OpenRAIL-M",
        )
    )

MODEL_CATALOG: Mapping[str, ModelSpec] = MappingProxyType({entry.id: entry for entry in _entries})
if len(MODEL_CATALOG) != len(_entries):
    raise RuntimeError("Duplicate model IDs")


def model_info(model_id: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[model_id]
    except KeyError:
        available = ", ".join(sorted(MODEL_CATALOG))
        raise ValueError(f"Unknown model {model_id!r}. Available: {available}") from None


def available_models(task: Task | str | None = None) -> dict[str, tuple[str, ...]]:
    tasks = (Task(task),) if task is not None else tuple(Task)
    return {
        kind.value: tuple(sorted(spec.id for spec in MODEL_CATALOG.values() if spec.task == kind))
        for kind in tasks
    }
```

If ruff reports E501 on the tuple lines for v1.1–v1.3, break them the same way as v1.4.

- [ ] **Step 4: Write `src/imagehub/registry.py`**

```python
"""Explicit, extensible backend selection with lazy architecture imports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from imagehub.catalog import MODEL_CATALOG
from imagehub.types import Task

BackendFactory = Callable[..., Any]


@dataclass(frozen=True)
class _Registration:
    factory: BackendFactory | str


class BackendRegistry:
    def __init__(self) -> None:
        self._entries: dict[tuple[Task, str], _Registration] = {}

    def register(
        self, task: Task | str, name: str, factory: BackendFactory | str, *, replace: bool = False
    ) -> None:
        """Register a callable or ``module:class`` import path; no model is loaded."""
        if not name or name.strip() != name:
            raise ValueError("Backend name must be nonempty and have no surrounding whitespace")
        key = (Task(task), name)
        if key in self._entries and not replace:
            raise ValueError(f"Backend {name!r} is already registered for {key[0].value}")
        if isinstance(factory, str):
            if len(factory.split(":")) != 2 or not all(factory.split(":")):
                raise ValueError("Lazy backend paths must use 'module:class'")
        elif not callable(factory):
            raise TypeError("Backend factory must be callable or a 'module:class' string")
        self._entries[key] = _Registration(factory)

    def names(self, task: Task | str) -> tuple[str, ...]:
        selected = Task(task)
        return tuple(sorted(name for (kind, name) in self._entries if kind == selected))

    def factory(self, task: Task | str, name: str) -> BackendFactory | str:
        selected = Task(task)
        entry = self._entries.get((selected, name))
        if entry is None:
            available = ", ".join(self.names(selected)) or "none"
            raise ValueError(f"Unknown {selected.value} backend {name!r}. Available: {available}")
        return entry.factory

    def create(self, task: Task | str, name: str, **options: Any) -> Any:
        selected = Task(task)
        factory = self.factory(selected, name)
        # Imported here: backends depend on the registry's package, not the other way round.
        from imagehub.backends.base import Backend

        if isinstance(factory, str):
            module, attribute = factory.split(":")
            factory = getattr(import_module(module), attribute)
        backend = factory(**options)
        if not isinstance(backend, Backend) or backend.task != selected:
            raise TypeError(f"Factory {name!r} must return a {selected.value} Backend")
        return backend


backend_registry = BackendRegistry()
for _task, _name, _path in (
    (Task.T2I, "stable-diffusion", "imagehub.backends.stable_diffusion:StableDiffusionBackend"),
):
    backend_registry.register(_task, _name, _path)

# Published model IDs are also valid selectors. Their checkpoint files stay local
# and explicit, just like family selectors; listing a model never downloads it.
for _spec in MODEL_CATALOG.values():
    if _spec.id not in backend_registry.names(_spec.task):
        backend_registry.register(
            _spec.task, _spec.id, backend_registry.factory(_spec.task, _spec.family)
        )


def available_backends(task: Task | str | None = None) -> dict[str, tuple[str, ...]]:
    tasks = (Task(task),) if task is not None else tuple(Task)
    return {kind.value: backend_registry.names(kind) for kind in tasks}
```

- [ ] **Step 5: Export from `src/imagehub/__init__.py`**

```python
"""imagehub: open-weight image generation architectures in native PyTorch."""

from imagehub.catalog import ModelSpec, available_models, model_info
from imagehub.image import Image
from imagehub.registry import BackendRegistry, available_backends, backend_registry
from imagehub.types import ImageResult, T2IRequest, Task

__version__ = "0.1.0"

__all__ = [
    "BackendRegistry",
    "Image",
    "ImageResult",
    "ModelSpec",
    "T2IRequest",
    "Task",
    "__version__",
    "available_backends",
    "available_models",
    "backend_registry",
    "model_info",
]
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests -v`
Expected: all pass; `test_catalog_link` passes when `../awesome-text-to-image-architectures` exists (it does on the dev machine), otherwise skips.

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
git add src tests
git commit -m "Add hand-written model catalog with catalog_id links and lazy backend registry"
```

---
### Task 5: Verified downloads and `python -m imagehub.pretrained`

**Files:**
- Create: `src/imagehub/download.py`, `src/imagehub/pretrained.py`, `tests/test_download.py`

**Interfaces:**
- Consumes: `model_info`, `ModelSpec` (Task 4); `DownloadError` (Task 1).
- Produces:
  - `download._open(url, headers)` (monkeypatched in tests), `hf_headers(token) -> dict`, `resolve_revision(repo_id, revision, headers) -> tuple[str, dict[str, dict]]`, `safe_relative(name) -> Path`, `fetch(url, target, headers, *, sha256, size) -> str` (returns sha256 hex).
  - `pretrained.SNAPSHOT_FILE = "imagehub_snapshot.json"`, `download_pretrained(model_id, destination, *, revision=None, token=None, progress=None) -> Path`, `read_snapshot(directory) -> dict | None`, `main(argv=None) -> int`.
  - Manifest JSON: `{"model_id", "repo_id", "revision", "verified_revision": bool, "files": {path: sha256}}`.

- [ ] **Step 1: Write failing tests `tests/test_download.py`**

```python
import hashlib
import io
import json
import ssl
import urllib.error
import urllib.request

import pytest

from imagehub import download
from imagehub.catalog import model_info
from imagehub.download import _Redirect, safe_relative
from imagehub.exceptions import DownloadError
from imagehub.pretrained import SNAPSHOT_FILE, download_pretrained, main, read_snapshot

SPEC = model_info("sd-v1-5")


def fake_hub(monkeypatch, *, commit=SPEC.revision, corrupt=None, short=None, drop=None):
    contents = {name: f"content of {name}".encode() for name in SPEC.files}
    siblings = []
    for name, data in contents.items():
        if name == drop:
            continue
        entry = {"rfilename": name, "size": len(data)}
        if name.endswith((".safetensors", ".bin")):
            entry["lfs"] = {"sha256": hashlib.sha256(data).hexdigest()}
        siblings.append(entry)
    siblings.append({"rfilename": "safety_checker/model.safetensors", "size": 1})
    calls = []

    def request(url, headers):
        calls.append((url, dict(headers)))
        if "/api/models/" in url:
            return io.BytesIO(json.dumps({"sha": commit, "siblings": siblings}).encode())
        name = url.split(f"/resolve/{commit}/", 1)[1]
        data = contents[name]
        if name == corrupt:
            data = b"X" + data[1:]
        if name == short:
            data = data[:-1]
        return io.BytesIO(data)

    monkeypatch.setattr(download, "_open", request)
    return contents, calls


def test_downloads_pinned_verified_snapshot(monkeypatch, tmp_path):
    contents, calls = fake_hub(monkeypatch)
    dest = download_pretrained("sd-v1-5", tmp_path / "sd15")
    manifest = read_snapshot(dest)
    assert manifest["model_id"] == "sd-v1-5"
    assert manifest["revision"] == SPEC.revision
    assert manifest["verified_revision"] is True
    assert set(manifest["files"]) == set(SPEC.files)
    for name, data in contents.items():
        assert (dest / name).read_bytes() == data
        assert manifest["files"][name] == hashlib.sha256(data).hexdigest()
    assert not (dest / "safety_checker").exists()
    assert f"/revision/{SPEC.revision}" in calls[0][0]
    assert all(f"/resolve/{SPEC.revision}/" in url for url, _ in calls[1:])
    assert all("Authorization" not in headers for _, headers in calls)
    with pytest.raises(FileExistsError):
        download_pretrained("sd-v1-5", dest)


def test_token_is_sent_to_hugging_face(monkeypatch, tmp_path):
    _, calls = fake_hub(monkeypatch)
    download_pretrained("sd-v1-5", tmp_path / "sd15", token="hf_test")
    assert calls[0][1]["Authorization"] == "Bearer hf_test"


def test_unvalidated_revision_is_flagged(monkeypatch, tmp_path):
    fake_hub(monkeypatch, commit="c" * 40)
    with pytest.warns(UserWarning, match="not the validated revision"):
        dest = download_pretrained("sd-v1-5", tmp_path / "sd15", revision="main")
    assert read_snapshot(dest)["verified_revision"] is False


def test_missing_artifact_fails_before_downloading(monkeypatch, tmp_path):
    _, calls = fake_hub(monkeypatch, drop="vae/config.json")
    with pytest.raises(DownloadError, match="vae/config.json"):
        download_pretrained("sd-v1-5", tmp_path / "sd15")
    assert not any("/resolve/" in url for url, _ in calls)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"corrupt": "unet/diffusion_pytorch_model.safetensors"}, "Checksum"),
        ({"short": "model_index.json"}, "Incomplete"),
    ],
)
def test_failed_verification_leaves_nothing_behind(monkeypatch, tmp_path, kwargs, message):
    fake_hub(monkeypatch, **kwargs)
    with pytest.raises(DownloadError, match=message):
        download_pretrained("sd-v1-5", tmp_path / "sd15")
    assert list(tmp_path.iterdir()) == []


def test_read_snapshot_missing(tmp_path):
    assert read_snapshot(tmp_path) is None


@pytest.mark.parametrize("name", ["../escape", "/abs/path", ""])
def test_safe_relative_rejects(name):
    with pytest.raises(DownloadError):
        safe_relative(name)


def test_cross_host_redirect_does_not_forward_credentials():
    original = urllib.request.Request(
        "https://huggingface.co/owner/repo", headers={"Authorization": "Bearer example"}
    )
    redirected = _Redirect().redirect_request(
        original, None, 302, "Found", {}, "https://cdn.example/model"
    )
    assert redirected.get_header("Authorization") is None
    same_host = _Redirect().redirect_request(
        original, None, 302, "Found", {}, "https://huggingface.co/new"
    )
    assert same_host.get_header("Authorization") == "Bearer example"
    with pytest.raises(DownloadError, match="HTTPS"):
        _Redirect().redirect_request(original, None, 302, "Found", {}, "http://cdn.example/model")


def test_certificate_failure_explains_fix(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            raise urllib.error.URLError(ssl.SSLCertVerificationError("unable to get issuer"))

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(DownloadError, match="Install Certificates"):
        download._open("https://huggingface.co/api/models/x", {})


def test_main_prints_destination(monkeypatch, tmp_path, capsys):
    fake_hub(monkeypatch)
    assert main(["sd-v1-5", str(tmp_path / "sd15")]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == str((tmp_path / "sd15").resolve())
    assert "unet/diffusion_pytorch_model.safetensors" in out.err
    assert (tmp_path / "sd15" / SNAPSHOT_FILE).is_file()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_download.py -v`
Expected: FAIL with `ImportError: cannot import name 'download' from 'imagehub'` or `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/imagehub/download.py`**

```python
"""Explicit, standard-library Hugging Face downloads; never run during inference."""

from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from imagehub.exceptions import DownloadError

USER_AGENT = "imagehub/0.1"
_SSL_HINT = (
    "TLS certificate verification failed. On python.org builds for macOS run "
    "'Install Certificates.command' from the Python application folder, or set "
    "SSL_CERT_FILE to a CA bundle."
)


class _Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise DownloadError("Model downloads require HTTPS redirects")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and (
            urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


def _open(url: str, headers: dict[str, str]) -> Any:
    opener = urllib.request.build_opener(_Redirect())
    try:
        return opener.open(urllib.request.Request(url, headers=headers), timeout=30)
    except urllib.error.URLError as error:
        if isinstance(error.reason, ssl.SSLCertVerificationError):
            raise DownloadError(_SSL_HINT) from error
        raise


def _read_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    with _open(url, headers) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise DownloadError(f"Unexpected response from {url}")
    return value


def hf_headers(token: str | None) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def resolve_revision(
    repo_id: str, revision: str, headers: dict[str, str]
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Return the immutable commit for ``revision`` and its files keyed by path."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
        raise DownloadError("repo_id must have the form owner/model")
    quoted = urllib.parse.quote(revision, safe="")
    info = _read_json(
        f"https://huggingface.co/api/models/{repo_id}/revision/{quoted}?blobs=true", headers
    )
    commit = info.get("sha")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise DownloadError("The model API did not return an immutable commit hash")
    return commit, {entry["rfilename"]: entry for entry in info.get("siblings", [])}


def safe_relative(name: str) -> Path:
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise DownloadError(f"Unsafe path in model snapshot: {name!r}")
    return path


def fetch(
    url: str, target: Path, headers: dict[str, str], *, sha256: str | None, size: int | None
) -> str:
    """Download ``url`` to ``target`` atomically; verify size and sha256 when published."""
    if not url.startswith("https://"):
        raise DownloadError("Model downloads require HTTPS")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".imagehub-", delete=False
        ) as sink:
            temporary = sink.name
            digest = hashlib.sha256()
            received = 0
            with _open(url, headers) as response:
                while data := response.read(1 << 20):
                    sink.write(data)
                    digest.update(data)
                    received += len(data)
        if size is not None and received != size:
            raise DownloadError(f"Incomplete download for {target.name}: {received} of {size} bytes")
        if sha256 is not None and digest.hexdigest() != sha256:
            raise DownloadError(f"Checksum mismatch for {target.name}")
        os.replace(temporary, target)
        temporary = None
        return digest.hexdigest()
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
```

- [ ] **Step 4: Write `src/imagehub/pretrained.py`**

```python
"""Explicitly download a catalog model as a pinned, verified local snapshot.

Usage: python -m imagehub.pretrained <model-id> <empty-directory> [--revision REF]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.parse
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from imagehub import download
from imagehub.catalog import model_info
from imagehub.exceptions import DownloadError

SNAPSHOT_FILE = "imagehub_snapshot.json"


def download_pretrained(
    model_id: str,
    destination: str | os.PathLike[str],
    *,
    revision: str | None = None,
    token: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Download the catalog's file list for ``model_id`` into an empty directory.

    The default revision is the commit validated in docs/validation.json. Any other
    revision is downloaded with ``verified_revision: false`` in the manifest.
    """
    spec = model_info(model_id)
    dest = Path(destination)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise FileExistsError(f"Use an empty destination for a reproducible model snapshot: {dest}")
    if spec.repo_id is None:
        raise DownloadError(f"{model_id} has no Hugging Face repository")
    headers = download.hf_headers(token)
    commit, siblings = download.resolve_revision(spec.repo_id, revision or spec.revision, headers)
    missing = [name for name in spec.files if name not in siblings]
    if missing:
        raise DownloadError(f"Published artifacts are missing at {commit}: {', '.join(missing)}")
    verified = commit == spec.revision
    if not verified:
        warnings.warn(
            f"{model_id}: commit {commit} is not the validated revision {spec.revision}; "
            "imagehub's parity evidence does not apply to it",
            UserWarning,
            stacklevel=2,
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".imagehub-download-", dir=dest.parent) as tmp:
        staging = Path(tmp) / "snapshot"
        files: dict[str, str] = {}
        for name in spec.files:
            relative = download.safe_relative(name)
            entry = siblings[name]
            url = f"https://huggingface.co/{spec.repo_id}/resolve/{commit}/" + urllib.parse.quote(
                name, safe="/"
            )
            if progress is not None:
                progress(name)
            files[name] = download.fetch(
                url,
                staging / relative,
                headers,
                sha256=entry.get("lfs", {}).get("sha256"),
                size=entry.get("size"),
            )
        manifest = {
            "model_id": spec.id,
            "repo_id": spec.repo_id,
            "revision": commit,
            "verified_revision": verified,
            "files": files,
        }
        (staging / SNAPSHOT_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if dest.exists():
            dest.rmdir()  # verified empty above; never replace user files
        shutil.move(str(staging), dest)
    return dest


def read_snapshot(directory: str | os.PathLike[str]) -> dict[str, Any] | None:
    path = Path(directory) / SNAPSHOT_FILE
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m imagehub.pretrained",
        description="Download a catalog model as a pinned, sha256-verified local snapshot.",
    )
    parser.add_argument("model_id", help="catalog id, e.g. sd-v1-5")
    parser.add_argument("destination", type=Path, help="empty or missing directory")
    parser.add_argument(
        "--revision", default=None, help="branch, tag or commit (default: validated commit)"
    )
    args = parser.parse_args(argv)
    path = download_pretrained(
        args.model_id,
        args.destination,
        revision=args.revision,
        token=os.environ.get("HF_TOKEN"),
        progress=lambda name: print(f"downloading {name}", file=sys.stderr, flush=True),
    )
    print(path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: `pretrained` calls `download.fetch`/`download._open` through the module so the tests' `monkeypatch.setattr(download, "_open", ...)` takes effect.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_download.py -v`
Expected: all pass.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
git add src tests
git commit -m "Add pinned, sha256-verified snapshot downloads"
```

---

### Task 6: Native CLIP BPE tokenizer

**Files:**
- Create: `src/imagehub/tokenization.py`, `tests/helpers.py`, `tests/test_tokenization.py`, `tests/reference/conftest.py`, `tests/reference/test_ref_tokenizer.py` (no `__init__.py` files under `tests/`; `pythonpath = ["tests"]` makes `helpers` importable)

**Interfaces:**
- Produces:
  - `bytes_to_unicode() -> dict[int, str]`, `normalize(text) -> str`, `pretokenize(text) -> list[str]`, constants `BOS = "<|startoftext|>"`, `EOS = "<|endoftext|>"`.
  - `CLIPTokenizer(directory, *, max_length=77)` with `.bos_id`, `.eos_id`, `.pad_id`, `.max_length`, `.tokenize(text) -> list[int]` (no specials/padding), `__call__(text) -> list[int]` (BOS + content truncated to `max_length - 2` + EOS, padded with EOS to `max_length`).
  - `tests/helpers.py`: `write_tiny_tokenizer(directory) -> Path`, `TINY_VOCAB_SIZE = 517`, `checkpoint_dir(version) -> Path`, `max_diff(actual, expected) -> dict[str, float]`.
  - `tests/reference/conftest.py`: skips `reference`-marked tests unless `IMAGEHUB_REFERENCE_TESTS=1`; fixture `record(name, **values)`; writes measurements to `$IMAGEHUB_RECORD` at session end; uses `IMAGEHUB_REFERENCE_THREADS` (default: all cores) threads.

- [ ] **Step 1: Write `tests/helpers.py`**

```python
"""Shared test helpers: tiny tokenizer, checkpoint locations and difference metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from imagehub.tokenization import BOS, EOS, bytes_to_unicode

TINY_MERGES = (("h", "i</w>"), ("c", "a"), ("ca", "t</w>"))
TINY_VOCAB_SIZE = 517


def write_tiny_tokenizer(directory: str | os.PathLike[str]) -> Path:
    """Byte-level vocab (512 symbols) + 3 merges + BOS/EOS; ids: hi</w>=512, ca=513, cat</w>=514."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    chars = list(bytes_to_unicode().values())
    vocab = {c: i for i, c in enumerate(chars)} | {c + "</w>": 256 + i for i, c in enumerate(chars)}
    for a, b in TINY_MERGES:
        vocab[a + b] = len(vocab)
    vocab[BOS] = len(vocab)
    vocab[EOS] = len(vocab)
    assert len(vocab) == TINY_VOCAB_SIZE
    (path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    merges = "#version: 0.2\n" + "".join(f"{a} {b}\n" for a, b in TINY_MERGES)
    (path / "merges.txt").write_text(merges, encoding="utf-8")
    special = {"bos_token": BOS, "eos_token": EOS, "pad_token": EOS, "unk_token": EOS}
    (path / "special_tokens_map.json").write_text(json.dumps(special), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps(special | {"model_max_length": 77, "do_lower_case": True}), encoding="utf-8"
    )
    return path


def checkpoint_dir(version: int) -> Path:
    """Local sd-v1-<version> snapshot from IMAGEHUB_SD_V1_<version>, or skip the test."""
    value = os.environ.get(f"IMAGEHUB_SD_V1_{version}")
    if not value:
        pytest.skip(f"Set IMAGEHUB_SD_V1_{version} to a local sd-v1-{version} snapshot")
    return Path(value)


def max_diff(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    a = actual.detach().cpu().double()
    e = expected.detach().cpu().double()
    d = (a - e).abs()
    return {
        "max_abs": float(d.max()),
        "max_rel": float((d / e.abs().clamp_min(1e-12)).max()),
    }
```

- [ ] **Step 2: Write `tests/reference/conftest.py`**

```python
import json
import os
from pathlib import Path

import pytest
import torch

_MEASUREMENTS: dict[str, dict] = {}


def pytest_collection_modifyitems(config, items):
    if os.environ.get("IMAGEHUB_REFERENCE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="Set IMAGEHUB_REFERENCE_TESTS=1 to run reference comparisons")
    for item in items:
        if "reference" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def reference_threads():
    threads = int(os.environ.get("IMAGEHUB_REFERENCE_THREADS", os.cpu_count() or 1))
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    _MEASUREMENTS["environment"] = {"torch_threads": threads}
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def record():
    def add(name: str, **values):
        _MEASUREMENTS[name] = values

    return add


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("IMAGEHUB_RECORD")
    if not path or len(_MEASUREMENTS) <= 1:
        return
    target = Path(path)
    existing = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    existing.update(_MEASUREMENTS)
    target.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

- [ ] **Step 3: Write failing tests `tests/test_tokenization.py`**

```python
import pytest

from helpers import write_tiny_tokenizer
from imagehub.tokenization import CLIPTokenizer, bytes_to_unicode, normalize, pretokenize


def test_bytes_to_unicode_is_a_bijection():
    table = bytes_to_unicode()
    assert len(table) == 256 and len(set(table.values())) == 256
    assert table[ord("a")] == "a" and table[ord(" ")] == "Ġ"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello   World\t\nX", "hello world x"),
        ("ΣΟΦΟΣ", "σοφοσ"),  # per-character lowercase: no final-sigma rule
        ("é", "é"),  # NFC
        ("İ", "i̇"),
    ],
)
def test_normalize(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("don't stop!!! x2 <|endoftext|>", ["don", "'t", "stop", "!!!", "x", "2", "<|endoftext|>"]),
        ("a1b22", ["a", "1", "b", "2", "2"]),
        ("hi...there", ["hi", "...", "there"]),
        ("x!<|endoftext|>y", ["x", "!", "<|endoftext|>", "y"]),
        ("'''s", ["'''", "s"]),
        ("日本語 テスト", ["日本語", "テスト"]),
    ],
)
def test_pretokenize(text, expected):
    assert pretokenize(normalize(text)) == expected


def test_tiny_vocab_bpe(tmp_path):
    tok = CLIPTokenizer(write_tiny_tokenizer(tmp_path), max_length=8)
    assert (tok.bos_id, tok.eos_id, tok.pad_id) == (515, 516, 516)
    assert tok.tokenize("hi cat") == [512, 514]
    assert tok.tokenize("cats") == [513, 83, 338]
    assert tok.tokenize("Hi CAT!") == [512, 514, 256]
    assert tok("hi cat") == [515, 512, 514, 516, 516, 516, 516, 516]
    assert tok("a b c d e f g h") == [515, 320, 321, 322, 323, 324, 325, 516]
    assert tok("") == [515, 516, 516, 516, 516, 516, 516, 516]
```

- [ ] **Step 4: Write the reference test `tests/reference/test_ref_tokenizer.py`**

```python
import pytest

from helpers import checkpoint_dir, write_tiny_tokenizer
from imagehub.tokenization import CLIPTokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

PROMPTS = [
    "a photo of an astronaut riding a horse on mars",
    "Bir kedi, yağlı boya; İSTANBUL'da ĞÜŞİÖÇ ığüşöç",
    "emoji 🐱🚀 and   multiple\tspaces\nnewline",
    "don't won't they're I've I'm we'll he'd 's '' '''s",
    "numbers 12345 3.14 1e-5 ½ ² Ⅷ ٣",
    "punct!!! ??? ... --- (a) [b] {c} <d> @#$%^&*",
    "x!<|endoftext|>y <|startoftext|> mid",
    "Ünïcödé NFC é ﬁ ligature Ωmega ΣΟΦΟΣ",
    "日本語のテキストと中文和한국어",
    "word " * 100,
    "",
]


def test_tiny_tokenizer_matches_transformers(tmp_path):
    path = write_tiny_tokenizer(tmp_path)
    ours = CLIPTokenizer(path, max_length=8)
    ref = transformers.CLIPTokenizer.from_pretrained(path, model_max_length=8)
    for text in ["hi cat", "cats", "Hi CAT!", "a b c d e f g h", ""]:
        expected = ref(text, padding="max_length", max_length=8, truncation=True)["input_ids"]
        assert ours(text) == expected, text


def test_sd1_tokenizer_matches_transformers(record):
    path = checkpoint_dir(4) / "tokenizer"
    ours = CLIPTokenizer(path)
    ref = transformers.CLIPTokenizer.from_pretrained(path)
    for text in PROMPTS:
        expected = ref(text, padding="max_length", max_length=77, truncation=True)["input_ids"]
        assert ours(text) == expected, text
    record("tokenizer", prompts=len(PROMPTS), mismatches=0, tolerance="exact")
```

- [ ] **Step 5: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_tokenization.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imagehub.tokenization'`.

- [ ] **Step 6: Write `src/imagehub/tokenization.py`**

```python
"""Native CLIP byte-level BPE tokenizer matching transformers' ``CLIPTokenizer``.

The reference normalizes with NFC, collapses Unicode whitespace runs to one space and
lowercases character by character, then splits with the pattern
``<|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+``.
Python's ``re`` has no ``\\p{..}`` classes, so the split is an explicit scanner.
"""

from __future__ import annotations

import json
import os
import unicodedata
from functools import lru_cache
from pathlib import Path

# Unicode White_Space property (what the reference Rust regex ``\s`` matches).
_WHITESPACE = frozenset(
    "\t\n\x0b\x0c\r \x85\xa0          "
    "      　"
)
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
BOS, EOS = "<|startoftext|>", "<|endoftext|>"


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 reversible byte -> printable unicode character table."""
    keep = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = keep[:]
    extra = 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(keep, map(chr, chars), strict=True))


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


def normalize(text: str) -> str:
    """NFC, collapse whitespace runs to one space, then lowercase each character.

    Per-character lowercasing matches the reference (no context-sensitive final sigma).
    """
    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    for ch in text:
        if ch in _WHITESPACE:
            if not out or out[-1] != " ":
                out.append(" ")
        else:
            out.append(ch)
    return "".join(ch.lower() for ch in out)


def pretokenize(text: str) -> list[str]:
    """Split like the reference regex: specials | contractions | letters+ | one digit | other+."""
    pieces: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _WHITESPACE:
            i += 1
            continue
        special = next((s for s in (BOS, EOS) if text.startswith(s, i)), None)
        if special is not None:
            pieces.append(special)
            i += len(special)
            continue
        contraction = next((c for c in _CONTRACTIONS if text.startswith(c, i)), None)
        if contraction is not None:
            pieces.append(contraction)
            i += len(contraction)
            continue
        j = i + 1
        if _is_letter(ch):
            while j < n and _is_letter(text[j]):
                j += 1
        elif not _is_number(ch):
            while j < n and not (
                text[j] in _WHITESPACE
                or _is_letter(text[j])
                or _is_number(text[j])
                or text.startswith(BOS, j)
                or text.startswith(EOS, j)
            ):
                j += 1
        pieces.append(text[i:j])
        i = j
    return pieces


class CLIPTokenizer:
    def __init__(self, directory: str | os.PathLike[str], *, max_length: int = 77) -> None:
        path = Path(directory)
        self.encoder: dict[str, int] = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        lines = (path / "merges.txt").read_text(encoding="utf-8").splitlines()
        merges = [tuple(line.split()) for line in lines if line and not line.startswith("#version")]
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.bos_id, self.eos_id = self.encoder[BOS], self.encoder[EOS]
        self.pad_id = self.eos_id
        self.max_length = max_length
        self._cache: dict[str, list[str]] = {}

    def _bpe(self, token: str) -> list[str]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            pairs = {(word[k], word[k + 1]) for k in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            merged: list[str] = []
            k = 0
            while k < len(word):
                if k < len(word) - 1 and (word[k], word[k + 1]) == best:
                    merged.append(word[k] + word[k + 1])
                    k += 2
                else:
                    merged.append(word[k])
                    k += 1
            word = merged
        self._cache[token] = word
        return word

    def tokenize(self, text: str) -> list[int]:
        """Content token ids without BOS/EOS, padding or truncation."""
        ids: list[int] = []
        for piece in pretokenize(normalize(text)):
            if piece in (BOS, EOS):
                ids.append(self.encoder[piece])
                continue
            mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            ids.extend(self.encoder[symbol] for symbol in self._bpe(mapped))
        return ids

    def __call__(self, text: str) -> list[int]:
        """BOS + content (truncated) + EOS, padded with EOS to ``max_length``."""
        content = self.tokenize(text)[: self.max_length - 2]
        ids = [self.bos_id, *content, self.eos_id]
        return ids + [self.pad_id] * (self.max_length - len(ids))
```

- [ ] **Step 7: Run normal tests, then the tiny reference test**

```bash
.venv/bin/python -m pytest tests/test_tokenization.py -v
uv pip install --python .venv -e ".[dev,image,reference]"
IMAGEHUB_REFERENCE_TESTS=1 .venv/bin/python -m pytest tests/reference/test_ref_tokenizer.py -v
```
Expected: normal tests pass; `test_tiny_tokenizer_matches_transformers` passes; `test_sd1_tokenizer_matches_transformers` is skipped (no `IMAGEHUB_SD_V1_4` yet — it runs in Task 14). Also confirm `.venv/bin/python -m pytest tests -q` skips reference tests without the env var.

- [ ] **Step 8: Lint and commit**

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
git add src tests
git commit -m "Add native CLIP BPE tokenizer and reference test harness"
```

---

### Task 7: Shared layers and the CLIP text encoder

**Files:**
- Create: `src/imagehub/models/__init__.py` (docstring only), `src/imagehub/models/layers.py`, `src/imagehub/models/clip_text.py`, `tests/fixtures/sd1_tensor_shapes.json`, `tests/test_clip_text.py`, `tests/reference/test_ref_random_parity.py`
- Modify: `tests/helpers.py` (append)

**Interfaces:**
- Consumes: `NativeModel` (Task 3), `write_tiny_tokenizer`/`TINY_VOCAB_SIZE` (Task 6).
- Produces:
  - `models.layers`: `timestep_embedding(timesteps, dim, *, flip_sin_to_cos, shift) -> Tensor`, `ResnetBlock(in, out, *, temb_channels, eps, groups=32)`, `Downsample(channels, *, padding)`, `Upsample(channels)` with `forward(x, size=None)`, `attention(q, k, v, heads) -> Tensor`.
  - `models.clip_text`: `CLIP_TEXT_DEFAULTS`, `CLIPTextModel(config)` (a `NativeModel`; `forward(input_ids) -> last_hidden_state`; `prepare_state_dict` drops `text_model.embeddings.position_ids`).
  - `tests/helpers.py`: `TINY_TEXT_CONFIG`, `published_shapes(component) -> dict[str, list[int]]`, `meta_shapes(model_cls, config) -> dict[str, list[int]]`, `text_state_for_ours(ref_state) -> dict`.

- [ ] **Step 1: Copy the published tensor-shape fixture**

```bash
mkdir -p tests/fixtures
cp docs/design/plans/assets/sd1_tensor_shapes.json tests/fixtures/sd1_tensor_shapes.json
```
This file holds the safetensors header (names, dtypes, shapes) of `CompVis/stable-diffusion-v1-4@133a221` for `text_encoder` (197), `unet` (686) and `vae` (248). All five SD 1.x checkpoints share these names.

- [ ] **Step 2: Append to `tests/helpers.py`**

```python
FIXTURES = Path(__file__).parent / "fixtures"

TINY_TEXT_CONFIG = {
    "vocab_size": TINY_VOCAB_SIZE,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 77,
}


def published_shapes(component: str) -> dict[str, list[int]]:
    data = json.loads((FIXTURES / "sd1_tensor_shapes.json").read_text(encoding="utf-8"))
    return {name: entry["shape"] for name, entry in data[component].items()}


def meta_shapes(model_cls, config) -> dict[str, list[int]]:
    with torch.device("meta"):
        model = model_cls(config)
    return {name: list(t.shape) for name, t in model.state_dict().items()}


def text_state_for_ours(ref_state: dict) -> dict:
    """transformers 5 CLIPTextModel state -> published-file names (``text_model.`` prefix)."""
    return {
        (name if name.startswith("text_model.") else "text_model." + name): value
        for name, value in ref_state.items()
    }
```

- [ ] **Step 3: Write failing tests `tests/test_clip_text.py`**

```python
import pytest
import torch

from helpers import TINY_TEXT_CONFIG, meta_shapes, published_shapes
from imagehub.models.clip_text import CLIPTextModel


def test_matches_published_tensor_layout():
    expected = published_shapes("text_encoder")
    expected.pop("text_model.embeddings.position_ids")
    assert meta_shapes(CLIPTextModel, {}) == expected


def test_forward_shape_and_causality():
    torch.manual_seed(0)
    model = CLIPTextModel(TINY_TEXT_CONFIG).eval()
    ids = torch.randint(0, 517, (2, 77))
    changed = ids.clone()
    changed[:, 10] = (changed[:, 10] + 1) % 517
    with torch.inference_mode():
        a, b = model(ids), model(changed)
    assert a.shape == (2, 77, 32)
    torch.testing.assert_close(a[:, :10], b[:, :10])
    assert not torch.equal(a[:, 10:], b[:, 10:])


def test_position_ids_are_dropped_on_load():
    model = CLIPTextModel(TINY_TEXT_CONFIG)
    state = dict(model.state_dict())
    state["text_model.embeddings.position_ids"] = torch.arange(77)[None]
    model.load_checkpoint(state)


def test_rejects_other_activations():
    with pytest.raises(ValueError, match="activation"):
        CLIPTextModel(TINY_TEXT_CONFIG | {"hidden_act": "gelu"})
```

- [ ] **Step 4: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_clip_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imagehub.models'`.

- [ ] **Step 5: Write `src/imagehub/models/__init__.py` and `src/imagehub/models/layers.py`**

`src/imagehub/models/__init__.py`:
```python
"""Native PyTorch network definitions. Tensor names follow the published checkpoints."""
```

`src/imagehub/models/layers.py`:
```python
"""Shared convolutional diffusion blocks (ResNet, attention, resampling)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def timestep_embedding(
    timesteps: Tensor, dim: int, *, flip_sin_to_cos: bool, shift: float
) -> Tensor:
    """Sinusoidal timestep features, numerically matching diffusers.get_timestep_embedding."""
    half = dim // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half - shift)
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
    return emb


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        temb_channels: int | None,
        eps: float,
        groups: int = 32,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels) if temb_channels else None
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x: Tensor, temb: Tensor | None = None) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if self.time_emb_proj is not None and temb is not None:
            h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class Downsample(nn.Module):
    """Stride-2 convolution. ``padding=0`` pads right/bottom by one, as in the LDM autoencoder."""

    def __init__(self, channels: int, *, padding: int) -> None:
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        if self.padding == 0:
            x = F.pad(x, (0, 1, 0, 1))
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor, size: tuple[int, int] | None = None) -> Tensor:
        if size is None:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        else:
            x = F.interpolate(x, size=size, mode="nearest")
        return self.conv(x)


def attention(q: Tensor, k: Tensor, v: Tensor, heads: int) -> Tensor:
    """Multi-head scaled dot-product attention over (batch, tokens, channels) inputs."""
    b = q.shape[0]
    q, k, v = (t.view(b, -1, heads, t.shape[-1] // heads).transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(b, -1, heads * out.shape[-1])
```

- [ ] **Step 6: Write `src/imagehub/models/clip_text.py`**

```python
"""CLIP text transformer (OpenAI ViT-L/14 text tower) with transformers ``CLIPTextModel`` names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

CLIP_TEXT_DEFAULTS: dict[str, Any] = {
    "vocab_size": 49408,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "max_position_embeddings": 77,
    "layer_norm_eps": 1e-5,
    "hidden_act": "quick_gelu",
}


class CLIPAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        q, k, v = (
            proj(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).reshape(b, n, d))


class CLIPMLP(nn.Module):
    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        return self.fc2(x * torch.sigmoid(1.702 * x))  # quick_gelu


class CLIPEncoderLayer(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        dim, eps = cfg["hidden_size"], cfg["layer_norm_eps"]
        self.self_attn = CLIPAttention(dim, cfg["num_attention_heads"])
        self.layer_norm1 = nn.LayerNorm(dim, eps=eps)
        self.mlp = CLIPMLP(dim, cfg["intermediate_size"])
        self.layer_norm2 = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class CLIPEmbeddings(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.position_embedding = nn.Embedding(cfg["max_position_embeddings"], cfg["hidden_size"])

    def forward(self, input_ids: Tensor) -> Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return self.token_embedding(input_ids) + self.position_embedding(positions)[None]


class CLIPEncoder(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(CLIPEncoderLayer(cfg) for _ in range(cfg["num_hidden_layers"]))


class CLIPTextTransformer(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.embeddings = CLIPEmbeddings(cfg)
        self.encoder = CLIPEncoder(cfg)
        self.final_layer_norm = nn.LayerNorm(cfg["hidden_size"], eps=cfg["layer_norm_eps"])

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embeddings(input_ids)
        for layer in self.encoder.layers:
            x = layer(x)
        return self.final_layer_norm(x)


class CLIPTextModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(CLIP_TEXT_DEFAULTS | dict(config))
        if self.config["hidden_act"] != "quick_gelu":
            raise ValueError(f"Unsupported CLIP activation {self.config['hidden_act']!r}")
        self.text_model = CLIPTextTransformer(self.config)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        # Published files store the constant position-id buffer; positions are recomputed.
        state = dict(state)
        state.pop("text_model.embeddings.position_ids", None)
        return state

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return ``last_hidden_state`` of shape (batch, tokens, hidden_size)."""
        return self.text_model(input_ids)
```

- [ ] **Step 7: Write the random-weight reference test `tests/reference/test_ref_random_parity.py`**

These tests compare against upstream implementations with random weights; they need the `reference` extra but no checkpoint.

```python
"""Structural parity with random weights: no checkpoint download needed."""

import pytest
import torch

from helpers import TINY_TEXT_CONFIG, text_state_for_ours
from imagehub.models.clip_text import CLIPTextModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")
diffusers = pytest.importorskip("diffusers")


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("config", [TINY_TEXT_CONFIG, {}], ids=["tiny", "sd1"])
def test_clip_text_random_weights(config):
    torch.manual_seed(0)
    ref = transformers.CLIPTextModel(transformers.CLIPTextConfig(**config)).eval()
    ours = CLIPTextModel(config).eval()
    ours.load_checkpoint(text_state_for_ours(ref.state_dict()))
    ids = torch.randint(0, ours.config["vocab_size"], (3, 77))
    with torch.inference_mode():
        exact(ours(ids), ref(ids)[0])
```

- [ ] **Step 8: Run tests**

```bash
.venv/bin/python -m pytest tests/test_clip_text.py -v
IMAGEHUB_REFERENCE_TESTS=1 .venv/bin/python -m pytest tests/reference/test_ref_random_parity.py -v
```
Expected: all pass with exact equality. If the reference comparison is not exact, stop and report the measured `max_diff` — do not loosen the tolerance.

- [ ] **Step 9: Lint and commit**

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
git add src tests
git commit -m "Add shared diffusion layers and native CLIP text encoder"
```

---

### Task 8: KL autoencoder (VAE)

**Files:**
- Create: `src/imagehub/models/autoencoder_kl.py`, `tests/test_autoencoder_kl.py`
- Modify: `tests/helpers.py` (append), `tests/reference/test_ref_random_parity.py` (append)

**Interfaces:**
- Consumes: `models.layers` (Task 7), `NativeModel` (Task 3).
- Produces: `VAE_DEFAULTS`; `AutoencoderKL(config)` with `.scaling_factor -> float`, `.encode(image) -> Tensor` (posterior mean, unscaled), `.decode(latents) -> Tensor` (unscaled latents → image ≈ [-1, 1]). Mid-block attention uses legacy names `group_norm`, `query`, `key`, `value`, `proj_attn`. `tests/helpers.py`: `TINY_VAE_CONFIG`, `legacy_vae_state(ref_state) -> dict`.

- [ ] **Step 1: Append to `tests/helpers.py`**

```python
TINY_VAE_CONFIG = {"block_out_channels": [32, 32, 64, 64], "norm_num_groups": 8}


def legacy_vae_state(ref_state: dict) -> dict:
    """diffusers AutoencoderKL state -> the legacy attention names used by published files."""
    renamed = {}
    for name, value in ref_state.items():
        if ".attentions." in name:
            for new, old in (("to_q", "query"), ("to_k", "key"), ("to_v", "value"), ("to_out.0", "proj_attn")):
                name = name.replace(f".{new}.", f".{old}.")
        renamed[name] = value
    return renamed
```

- [ ] **Step 2: Write failing tests `tests/test_autoencoder_kl.py`**

```python
import torch

from helpers import TINY_VAE_CONFIG, meta_shapes, published_shapes
from imagehub.models.autoencoder_kl import AutoencoderKL


def test_matches_published_tensor_layout():
    assert meta_shapes(AutoencoderKL, {}) == published_shapes("vae")


def test_shapes_and_scaling_factor():
    torch.manual_seed(0)
    vae = AutoencoderKL(TINY_VAE_CONFIG).eval()
    assert vae.scaling_factor == 0.18215
    assert AutoencoderKL(TINY_VAE_CONFIG | {"scaling_factor": 0.5}).scaling_factor == 0.5
    with torch.inference_mode():
        assert vae.decode(torch.randn(1, 4, 8, 6)).shape == (1, 3, 64, 48)
        assert vae.encode(torch.rand(1, 3, 64, 48) * 2 - 1).shape == (1, 4, 8, 6)
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_autoencoder_kl.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imagehub.models.autoencoder_kl'`.

- [ ] **Step 4: Write `src/imagehub/models/autoencoder_kl.py`**

```python
"""KL autoencoder (LDM f=8) with the weight names of the published SD 1.x diffusers checkpoints.

The mid-block attention keeps the legacy names ``query``/``key``/``value``/``proj_attn`` used by
every released SD 1.x VAE file, so checkpoints load strictly without key conversion.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.layers import Downsample, ResnetBlock, Upsample, attention

VAE_DEFAULTS: dict[str, Any] = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "scaling_factor": 0.18215,
}
_EPS = 1e-6


class VAEAttention(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, eps=_EPS)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj_attn = nn.Linear(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        residual = x
        hidden = self.group_norm(x).view(b, c, h * w).transpose(1, 2)
        out = attention(self.query(hidden), self.key(hidden), self.value(hidden), heads=1)
        out = self.proj_attn(out).transpose(1, 2).reshape(b, c, h, w)
        return out + residual


class VAEMidBlock(nn.Module):
    def __init__(self, ch: int, groups: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(ch, ch, temb_channels=None, eps=_EPS, groups=groups) for _ in range(2)
        )
        self.attentions = nn.ModuleList([VAEAttention(ch, groups)])

    def forward(self, x: Tensor) -> Tensor:
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, layers: int, groups: int, down: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(
                in_ch if i == 0 else out_ch, out_ch, temb_channels=None, eps=_EPS, groups=groups
            )
            for i in range(layers)
        )
        self.downsamplers = nn.ModuleList([Downsample(out_ch, padding=0)]) if down else None

    def forward(self, x: Tensor) -> Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, layers: int, groups: int, up: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(
                in_ch if i == 0 else out_ch, out_ch, temb_channels=None, eps=_EPS, groups=groups
            )
            for i in range(layers)
        )
        self.upsamplers = nn.ModuleList([Upsample(out_ch)]) if up else None

    def forward(self, x: Tensor) -> Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        ch, groups, layers = cfg["block_out_channels"], cfg["norm_num_groups"], cfg["layers_per_block"]
        self.conv_in = nn.Conv2d(cfg["in_channels"], ch[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(
            EncoderBlock(ch[max(i - 1, 0)], ch[i], layers, groups, down=i < len(ch) - 1)
            for i in range(len(ch))
        )
        self.mid_block = VAEMidBlock(ch[-1], groups)
        self.conv_norm_out = nn.GroupNorm(groups, ch[-1], eps=_EPS)
        self.conv_out = nn.Conv2d(ch[-1], 2 * cfg["latent_channels"], 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class Decoder(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        ch, groups, layers = cfg["block_out_channels"], cfg["norm_num_groups"], cfg["layers_per_block"]
        rev = list(reversed(ch))
        self.conv_in = nn.Conv2d(cfg["latent_channels"], rev[0], 3, padding=1)
        self.mid_block = VAEMidBlock(rev[0], groups)
        self.up_blocks = nn.ModuleList(
            DecoderBlock(rev[max(i - 1, 0)], rev[i], layers + 1, groups, up=i < len(ch) - 1)
            for i in range(len(ch))
        )
        self.conv_norm_out = nn.GroupNorm(groups, ch[0], eps=_EPS)
        self.conv_out = nn.Conv2d(ch[0], cfg["out_channels"], 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        x = self.mid_block(self.conv_in(z))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class AutoencoderKL(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(VAE_DEFAULTS | dict(config))
        latent = self.config["latent_channels"]
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.quant_conv = nn.Conv2d(2 * latent, 2 * latent, 1)
        self.post_quant_conv = nn.Conv2d(latent, latent, 1)

    @property
    def scaling_factor(self) -> float:
        return float(self.config["scaling_factor"])

    def encode(self, image: Tensor) -> Tensor:
        """Return the posterior mean (unscaled latents) for images in [-1, 1]."""
        moments = self.quant_conv(self.encoder(image))
        return moments.chunk(2, dim=1)[0]

    def decode(self, latents: Tensor) -> Tensor:
        """Decode unscaled latents to images in roughly [-1, 1]."""
        return self.decoder(self.post_quant_conv(latents))
```

- [ ] **Step 5: Append the random-weight reference test to `tests/reference/test_ref_random_parity.py`**

Add `TINY_VAE_CONFIG, legacy_vae_state` to the `from helpers import ...` line and `from imagehub.models.autoencoder_kl import AutoencoderKL` to the imports, then append (`SD1_VAE` lists the full SD 1.x VAE config because diffusers' `from_config` defaults differ):

```python
SD1_VAE = {
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "latent_channels": 4,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "sample_size": 512,
}


@pytest.mark.parametrize(
    ("config", "latent_hw"),
    [(SD1_VAE | TINY_VAE_CONFIG, (8, 6)), (SD1_VAE, (64, 64))],
    ids=["tiny", "sd1"],
)
def test_vae_random_weights(config, latent_hw):
    torch.manual_seed(0)
    ref = diffusers.AutoencoderKL.from_config(config).eval()
    for p in ref.parameters():
        p.data.normal_(0, 0.02)
    ours = AutoencoderKL(config).eval()
    ours.load_checkpoint(legacy_vae_state(ref.state_dict()))
    z = torch.randn(1, 4, *latent_hw)
    image = torch.rand(1, 3, latent_hw[0] * 8, latent_hw[1] * 8) * 2 - 1
    with torch.inference_mode():
        exact(ours.decode(z), ref.decode(z).sample)
        exact(ours.encode(image), ref.encode(image).latent_dist.mean)
```

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/test_autoencoder_kl.py -v
IMAGEHUB_REFERENCE_TESTS=1 .venv/bin/python -m pytest tests/reference/test_ref_random_parity.py -v
```
Expected: all pass with exact equality.

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
git add src tests
git commit -m "Add native KL autoencoder with legacy SD 1.x tensor names"
```

---

### Task 9: Conditional U-Net

**Files:**
- Create: `src/imagehub/models/unet_2d.py`, `tests/test_unet_2d.py`
- Modify: `tests/helpers.py` (append), `tests/reference/test_ref_random_parity.py` (append)

**Interfaces:**
- Consumes: `models.layers` (Task 7), `NativeModel` (Task 3).
- Produces: `UNET_DEFAULTS`; `UNet2DConditionModel(config)` with `forward(sample, timestep, context) -> Tensor` where `timestep` is a 0-d or 1-d tensor (broadcast to batch) and `context` is `(batch, tokens, cross_attention_dim)`. Spatial sizes need not be multiples of 8 (up-sampling targets the skip connection's size). `tests/helpers.py`: `TINY_UNET_CONFIG`, `SD1_UNET`.

- [ ] **Step 1: Append to `tests/helpers.py`**

```python
SD1_UNET = {
    "in_channels": 4,
    "out_channels": 4,
    "block_out_channels": [320, 640, 1280, 1280],
    "down_block_types": ["CrossAttnDownBlock2D"] * 3 + ["DownBlock2D"],
    "up_block_types": ["UpBlock2D"] + ["CrossAttnUpBlock2D"] * 3,
    "layers_per_block": 2,
    "attention_head_dim": 8,
    "cross_attention_dim": 768,
    "norm_num_groups": 32,
    "norm_eps": 1e-5,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "sample_size": 64,
}
TINY_UNET_CONFIG = SD1_UNET | {
    "block_out_channels": [32, 64, 64, 64],
    "cross_attention_dim": 32,
    "attention_head_dim": 2,
    "norm_num_groups": 8,
}
```

- [ ] **Step 2: Write failing tests `tests/test_unet_2d.py`**

```python
import math

import pytest
import torch

from helpers import TINY_UNET_CONFIG, meta_shapes, published_shapes
from imagehub.models.layers import timestep_embedding
from imagehub.models.unet_2d import UNet2DConditionModel


def test_matches_published_tensor_layout():
    assert meta_shapes(UNet2DConditionModel, {}) == published_shapes("unet")


def test_timestep_embedding_closed_form():
    emb = timestep_embedding(torch.tensor([3]), 4, flip_sin_to_cos=True, shift=0)
    freqs = [1.0, math.exp(-math.log(10000) / 2)]
    expected = [math.cos(3 * f) for f in freqs] + [math.sin(3 * f) for f in freqs]
    torch.testing.assert_close(emb[0], torch.tensor(expected))


@pytest.mark.parametrize("hw", [(8, 8), (9, 7)])
def test_forward_shapes(hw):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(TINY_UNET_CONFIG).eval()
    sample = torch.randn(2, 4, *hw)
    context = torch.randn(2, 77, 32)
    with torch.inference_mode():
        out = unet(sample, torch.tensor(981), context)
        batched_t = unet(sample, torch.tensor([981, 981]), context)
    assert out.shape == sample.shape
    torch.testing.assert_close(out, batched_t, rtol=0, atol=0)
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_unet_2d.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imagehub.models.unet_2d'`.

- [ ] **Step 4: Write `src/imagehub/models/unet_2d.py`**

```python
"""Stable Diffusion 1.x conditional U-Net with diffusers ``UNet2DConditionModel`` weight names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.layers import (
    Downsample,
    ResnetBlock,
    Upsample,
    attention,
    timestep_embedding,
)

UNET_DEFAULTS: dict[str, Any] = {
    "in_channels": 4,
    "out_channels": 4,
    "block_out_channels": [320, 640, 1280, 1280],
    "down_block_types": ["CrossAttnDownBlock2D"] * 3 + ["DownBlock2D"],
    "up_block_types": ["UpBlock2D"] + ["CrossAttnUpBlock2D"] * 3,
    "layers_per_block": 2,
    # SD 1.x configs store the number of heads under this historical name.
    "attention_head_dim": 8,
    "cross_attention_dim": 768,
    "norm_num_groups": 32,
    "norm_eps": 1e-5,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
}


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        context = x if context is None else context
        out = attention(self.to_q(x), self.to_k(context), self.to_v(context), self.heads)
        return self.to_out[0](out)


class GEGLU(nn.Module):
    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner * 2)

    def forward(self, x: Tensor) -> Tensor:
        hidden, gate = self.proj(x).chunk(2, dim=-1)
        return hidden * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        # Index 1 is dropout in the reference; the slot keeps ``net.2`` weight names aligned.
        self.net = nn.ModuleList([GEGLU(dim, dim * 4), nn.Identity(), nn.Linear(dim * 4, dim)])

    def forward(self, x: Tensor) -> Tensor:
        return self.net[2](self.net[0](x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, context_dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim, heads)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context) + x
        return self.ff(self.norm3(x)) + x


class SpatialTransformer(nn.Module):
    """diffusers ``Transformer2DModel`` with 1x1 convolution projections (SD 1.x)."""

    def __init__(self, channels: int, context_dim: int, heads: int, groups: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(channels, context_dim, heads)])
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.proj_in(self.norm(x))
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        for block in self.transformer_blocks:
            x = block(x, context)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return self.proj_out(x) + residual


def _transformers(cfg: Mapping[str, Any], channels: int, count: int) -> nn.ModuleList:
    return nn.ModuleList(
        SpatialTransformer(
            channels, cfg["cross_attention_dim"], cfg["attention_head_dim"], cfg["norm_num_groups"]
        )
        for _ in range(count)
    )


class DownBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, *, cfg: Mapping[str, Any], temb: int, attn: bool, down: bool
    ) -> None:
        super().__init__()
        layers, eps, groups = cfg["layers_per_block"], cfg["norm_eps"], cfg["norm_num_groups"]
        self.resnets = nn.ModuleList(
            ResnetBlock(in_ch if i == 0 else out_ch, out_ch, temb_channels=temb, eps=eps, groups=groups)
            for i in range(layers)
        )
        self.attentions = _transformers(cfg, out_ch, layers) if attn else None
        self.downsamplers = nn.ModuleList([Downsample(out_ch, padding=1)]) if down else None

    def forward(self, x: Tensor, temb: Tensor, context: Tensor) -> tuple[Tensor, list[Tensor]]:
        skips = []
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, temb)
            if self.attentions is not None:
                x = self.attentions[i](x, context)
            skips.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            skips.append(x)
        return x, skips


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        prev_ch: int,
        out_ch: int,
        *,
        cfg: Mapping[str, Any],
        temb: int,
        attn: bool,
        up: bool,
    ) -> None:
        super().__init__()
        layers, eps, groups = cfg["layers_per_block"] + 1, cfg["norm_eps"], cfg["norm_num_groups"]
        self.resnets = nn.ModuleList()
        for i in range(layers):
            skip_ch = in_ch if i == layers - 1 else out_ch
            res_in = prev_ch if i == 0 else out_ch
            self.resnets.append(
                ResnetBlock(res_in + skip_ch, out_ch, temb_channels=temb, eps=eps, groups=groups)
            )
        self.attentions = _transformers(cfg, out_ch, layers) if attn else None
        self.upsamplers = nn.ModuleList([Upsample(out_ch)]) if up else None

    def forward(
        self,
        x: Tensor,
        skips: list[Tensor],
        temb: Tensor,
        context: Tensor,
        size: tuple[int, int] | None,
    ) -> Tensor:
        for i, resnet in enumerate(self.resnets):
            x = resnet(torch.cat([x, skips.pop()], dim=1), temb)
            if self.attentions is not None:
                x = self.attentions[i](x, context)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, size)
        return x


class MidBlock(nn.Module):
    def __init__(self, ch: int, *, cfg: Mapping[str, Any], temb: int) -> None:
        super().__init__()
        eps, groups = cfg["norm_eps"], cfg["norm_num_groups"]
        self.resnets = nn.ModuleList(
            ResnetBlock(ch, ch, temb_channels=temb, eps=eps, groups=groups) for _ in range(2)
        )
        self.attentions = _transformers(cfg, ch, 1)

    def forward(self, x: Tensor, temb: Tensor, context: Tensor) -> Tensor:
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        return self.resnets[1](x, temb)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


class UNet2DConditionModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(UNET_DEFAULTS | dict(config))
        cfg = self.config
        ch = cfg["block_out_channels"]
        temb = ch[0] * 4
        self.conv_in = nn.Conv2d(cfg["in_channels"], ch[0], 3, padding=1)
        self.time_embedding = TimestepEmbedding(ch[0], temb)
        self.down_blocks = nn.ModuleList()
        out_ch = ch[0]
        for i, kind in enumerate(cfg["down_block_types"]):
            in_ch, out_ch = out_ch, ch[i]
            self.down_blocks.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    cfg=cfg,
                    temb=temb,
                    attn=kind == "CrossAttnDownBlock2D",
                    down=i < len(ch) - 1,
                )
            )
        self.mid_block = MidBlock(ch[-1], cfg=cfg, temb=temb)
        rev = list(reversed(ch))
        self.up_blocks = nn.ModuleList()
        out_ch = rev[0]
        for i, kind in enumerate(cfg["up_block_types"]):
            prev_ch, out_ch = out_ch, rev[i]
            in_ch = rev[min(i + 1, len(ch) - 1)]
            self.up_blocks.append(
                UpBlock(
                    in_ch,
                    prev_ch,
                    out_ch,
                    cfg=cfg,
                    temb=temb,
                    attn=kind == "CrossAttnUpBlock2D",
                    up=i < len(ch) - 1,
                )
            )
        self.conv_norm_out = nn.GroupNorm(cfg["norm_num_groups"], ch[0], eps=cfg["norm_eps"])
        self.conv_out = nn.Conv2d(ch[0], cfg["out_channels"], 3, padding=1)

    def forward(self, sample: Tensor, timestep: Tensor, context: Tensor) -> Tensor:
        timesteps = timestep.reshape(-1).to(sample.device).expand(sample.shape[0])
        t = timestep_embedding(
            timesteps,
            self.config["block_out_channels"][0],
            flip_sin_to_cos=self.config["flip_sin_to_cos"],
            shift=self.config["freq_shift"],
        ).to(sample.dtype)
        temb = self.time_embedding(t)
        x = self.conv_in(sample)
        skips = [x]
        for down in self.down_blocks:
            x, block_skips = down(x, temb, context)
            skips.extend(block_skips)
        x = self.mid_block(x, temb, context)
        n = len(self.up_blocks[0].resnets)
        for up in self.up_blocks:
            block_skips, skips = skips[-n:], skips[:-n]
            size = tuple(skips[-1].shape[2:]) if skips else None
            x = up(x, block_skips, temb, context, size)
        return self.conv_out(F.silu(self.conv_norm_out(x)))
```

The prototype passed `(int, int)` sizes; if mypy complains about `tuple[int, ...]` vs `tuple[int, int]`, annotate `size` as `tuple[int, ...] | None` in both `UpBlock.forward` and `Upsample.forward`.

- [ ] **Step 5: Append the random-weight reference test**

Add `SD1_UNET, TINY_UNET_CONFIG` to the helpers import and `from imagehub.models.unet_2d import UNet2DConditionModel`, then append:

```python
@pytest.mark.parametrize(
    ("config", "sizes"),
    [(TINY_UNET_CONFIG, [(16, 16), (13, 9)]), (SD1_UNET, [(64, 64), (65, 47)])],
    ids=["tiny", "sd1"],
)
def test_unet_random_weights(config, sizes):
    torch.manual_seed(0)
    ref = diffusers.UNet2DConditionModel.from_config(config).eval()
    for p in ref.parameters():
        p.data.normal_(0, 0.02)
    ours = UNet2DConditionModel(config).eval()
    ours.load_checkpoint(ref.state_dict())
    context = torch.randn(2, 77, config["cross_attention_dim"])
    for hw in sizes:
        sample = torch.randn(2, 4, *hw)
        for t in (torch.tensor(981), torch.tensor(1)):
            with torch.inference_mode():
                exact(ours(sample, t, context), ref(sample, t, context).sample)
```

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/test_unet_2d.py -v
IMAGEHUB_REFERENCE_TESTS=1 .venv/bin/python -m pytest tests/reference/test_ref_random_parity.py -v
```
Expected: all pass with exact equality (the full-size case takes tens of seconds).

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
git add src tests
git commit -m "Add native SD 1.x conditional U-Net"
```

---

### Task 10: PNDM scheduler (PLMS mode)

**Files:**
- Create: `src/imagehub/schedulers/__init__.py` (docstring only), `src/imagehub/schedulers/pndm.py`, `tests/test_pndm.py`
- Modify: `tests/helpers.py` (append), `tests/reference/test_ref_random_parity.py` (append)

**Interfaces:**
- Produces: `PNDMScheduler(config)` — accepts a `scheduler_config.json` dict (keys starting with `_` ignored); raises `ValueError` unless `skip_prk_steps=True`, `prediction_type="epsilon"`, `timestep_spacing="leading"`, `trained_betas=None`, `beta_schedule in {"scaled_linear", "linear"}`. Attributes `init_noise_sigma = 1.0`, `timesteps` (int64 tensor, `steps + 1` long), `alphas_cumprod`. Methods `set_timesteps(steps)`, `step(model_output, timestep, sample) -> Tensor` (raises `RuntimeError` before `set_timesteps`). Stateful: create one per trajectory. `tests/helpers.py`: `SD1_SCHEDULER`.

- [ ] **Step 1: Append to `tests/helpers.py`**

```python
SD1_SCHEDULER = {
    "_class_name": "PNDMScheduler",
    "_diffusers_version": "0.6.0",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "num_train_timesteps": 1000,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "trained_betas": None,
    "clip_sample": False,
}
```

- [ ] **Step 2: Write failing tests `tests/test_pndm.py`**

```python
import pytest
import torch

from helpers import SD1_SCHEDULER
from imagehub.schedulers.pndm import PNDMScheduler


def test_plms_timesteps_repeat_second_to_last():
    s = PNDMScheduler(SD1_SCHEDULER)
    s.set_timesteps(50)
    t = s.timesteps.tolist()
    assert len(t) == 51
    assert t[:4] == [981, 961, 961, 941] and t[-2:] == [21, 1]
    s.set_timesteps(1)
    assert s.timesteps.tolist() == [1]


def test_first_step_closed_form():
    s = PNDMScheduler(SD1_SCHEDULER)
    s.set_timesteps(50)
    torch.manual_seed(0)
    sample, eps = torch.randn(1, 4, 2, 2), torch.randn(1, 4, 2, 2)
    a_t, a_prev = s.alphas_cumprod[981], s.alphas_cumprod[961]
    expected = (a_prev / a_t) ** 0.5 * sample - (a_prev - a_t) * eps / (
        a_t * (1 - a_prev) ** 0.5 + (a_t * (1 - a_t) * a_prev) ** 0.5
    )
    torch.testing.assert_close(s.step(eps, s.timesteps[0], sample), expected)


@pytest.mark.parametrize(
    "override",
    [
        {"skip_prk_steps": False},
        {"prediction_type": "v_prediction"},
        {"timestep_spacing": "trailing"},
        {"beta_schedule": "squaredcos_cap_v2"},
    ],
)
def test_unsupported_settings(override):
    with pytest.raises(ValueError):
        PNDMScheduler(SD1_SCHEDULER | override)


def test_step_requires_timesteps():
    with pytest.raises(RuntimeError, match="set_timesteps"):
        PNDMScheduler(SD1_SCHEDULER).step(torch.zeros(1), 981, torch.zeros(1))
    with pytest.raises(ValueError):
        PNDMScheduler(SD1_SCHEDULER).set_timesteps(0)
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_pndm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imagehub.schedulers'`.

- [ ] **Step 4: Write `src/imagehub/schedulers/__init__.py` and `src/imagehub/schedulers/pndm.py`**

`src/imagehub/schedulers/__init__.py`:
```python
"""Native diffusion samplers shared across architecture families."""
```

`src/imagehub/schedulers/pndm.py`:
```python
"""PNDM scheduler in PLMS-only mode (``skip_prk_steps=True``), as configured by SD 1.x releases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

PNDM_DEFAULTS: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "set_alpha_to_one": False,
    "skip_prk_steps": False,
    "steps_offset": 0,
    "prediction_type": "epsilon",
    "timestep_spacing": "leading",
    "trained_betas": None,
}
_REQUIRED = {
    "skip_prk_steps": True,
    "prediction_type": "epsilon",
    "timestep_spacing": "leading",
    "trained_betas": None,
}


class PNDMScheduler:
    """Stateful sampler. Create one per trajectory and call :meth:`set_timesteps` first."""

    init_noise_sigma = 1.0

    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = PNDM_DEFAULTS | {k: v for k, v in config.items() if not k.startswith("_")}
        for key, expected in _REQUIRED.items():
            if cfg[key] != expected:
                raise ValueError(f"Unsupported PNDM setting {key}={cfg[key]!r}")
        n = cfg["num_train_timesteps"]
        if cfg["beta_schedule"] == "scaled_linear":
            betas = (
                torch.linspace(cfg["beta_start"] ** 0.5, cfg["beta_end"] ** 0.5, n, dtype=torch.float32)
                ** 2
            )
        elif cfg["beta_schedule"] == "linear":
            betas = torch.linspace(cfg["beta_start"], cfg["beta_end"], n, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported beta_schedule {cfg['beta_schedule']!r}")
        self.config = cfg
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if cfg["set_alpha_to_one"] else self.alphas_cumprod[0]
        )
        self.timesteps = torch.empty(0, dtype=torch.int64)
        self.num_inference_steps = 0
        self._ets: list[Tensor] = []
        self._cur_sample: Tensor | None = None
        self._counter = 0

    def set_timesteps(self, num_inference_steps: int) -> None:
        n = self.config["num_train_timesteps"]
        if type(num_inference_steps) is not int or not 0 < num_inference_steps <= n:
            raise ValueError(f"num_inference_steps must be an int in [1, {n}]")
        self.num_inference_steps = num_inference_steps
        ratio = n // num_inference_steps
        base = [i * ratio + self.config["steps_offset"] for i in range(num_inference_steps)]
        # PLMS repeats the second-to-last timestep: num_inference_steps + 1 model evaluations.
        plms = base[:-1] + base[-2:-1] + base[-1:]
        self.timesteps = torch.tensor(plms[::-1], dtype=torch.int64)
        self._ets, self._cur_sample, self._counter = [], None, 0

    def step(self, model_output: Tensor, timestep: Tensor | int, sample: Tensor) -> Tensor:
        if self.num_inference_steps == 0:
            raise RuntimeError("Call set_timesteps before step")
        t = torch.as_tensor(timestep)
        ratio = self.config["num_train_timesteps"] // self.num_inference_steps
        prev_t = t - ratio
        if self._counter != 1:
            self._ets = self._ets[-3:]
            self._ets.append(model_output)
        else:
            prev_t = t
            t = t + ratio
        ets = self._ets
        if len(ets) == 1 and self._counter == 0:
            self._cur_sample = sample
        elif len(ets) == 1 and self._counter == 1:
            model_output = (model_output + ets[-1]) / 2
            assert self._cur_sample is not None
            sample, self._cur_sample = self._cur_sample, None
        elif len(ets) == 2:
            model_output = (3 * ets[-1] - ets[-2]) / 2
        elif len(ets) == 3:
            model_output = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12
        else:
            model_output = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
        self._counter += 1
        return self._previous(sample, t, prev_t, model_output)

    def _previous(self, sample: Tensor, t: Tensor, prev_t: Tensor, eps: Tensor) -> Tensor:
        """PNDM paper eq. (9), evaluated in the same order as diffusers."""
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        beta_t, beta_prev = 1 - alpha_t, 1 - alpha_prev
        sample_coeff = (alpha_prev / alpha_t) ** 0.5
        denom = alpha_t * beta_prev**0.5 + (alpha_t * beta_t * alpha_prev) ** 0.5
        return sample_coeff * sample - (alpha_prev - alpha_t) * eps / denom
```

- [ ] **Step 5: Append the reference test**

Add `SD1_SCHEDULER` to the helpers import and `from imagehub.schedulers.pndm import PNDMScheduler`, then append:

```python
def test_pndm_matches_diffusers():
    ref = diffusers.PNDMScheduler.from_config(SD1_SCHEDULER)
    ours = PNDMScheduler(SD1_SCHEDULER)
    for steps in (50, 25, 7, 1):
        ref.set_timesteps(steps)
        ours.set_timesteps(steps)
        assert torch.equal(ref.timesteps, ours.timesteps), steps
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, 8)
    x_ref = x.clone()
    ref.set_timesteps(50)
    ours.set_timesteps(50)
    for t in ref.timesteps:
        eps = torch.randn(2, 4, 8, 8)
        x_ref = ref.step(eps, t, x_ref).prev_sample
        x = ours.step(eps, t, x)
    exact(x, x_ref)
```

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/test_pndm.py -v
IMAGEHUB_REFERENCE_TESTS=1 .venv/bin/python -m pytest tests/reference/test_ref_random_parity.py -v -k pndm
```
Expected: all pass with exact equality.

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
git add src tests
git commit -m "Add PNDM (PLMS) scheduler matching SD 1.x release configs"
```

---
### Task 11: Backend lifecycle, pipelines and `pipeline()`

**Files:**
- Create: `src/imagehub/backends/__init__.py`, `src/imagehub/backends/base.py`, `src/imagehub/pipelines/__init__.py`, `src/imagehub/pipelines/base.py`, `src/imagehub/pipelines/text_to_image.py`, `src/imagehub/factory.py`, `tests/test_pipeline.py`
- Modify: `src/imagehub/__init__.py`

**Interfaces:**
- Consumes: `T2IRequest`, `ImageResult`, `Task` (Task 2); `BackendRegistry`, `backend_registry` (Task 4); `BackendBusyError`, `BackendContractError` (Task 1).
- Produces:
  - `Backend[InputT, OutputT]` (ABC): `task`, `is_loaded`, `load() -> Self`, `unload()`, `validate_options(inputs, options: dict) -> dict` (runs before any load; default returns options), `infer(inputs, **options)`, `infer_batch(inputs, **options) -> list`, hooks `_load`, `_unload`, abstract `_infer(inputs, **options)`; context manager.
  - `TextToImageBackend(Backend[T2IRequest, ImageResult])` with `task = Task.T2I`.
  - `Pipeline[InputT, PreparedT, OutputT]`: `__init__(backend, *, default_options=None)`, `preprocess`, `forward`, `postprocess`, `run`, `__call__`, `batch`, `map`, `arun`, `load`, `close`, context manager. Call options override `default_options`; the backend merges `request.parameters` underneath both.
  - `TextToImagePipeline` accepting `str | T2IRequest`.
  - `pipeline(task, *, backend: str | Backend, options=None, registry=None, **backend_options) -> Pipeline`.

- [ ] **Step 1: Write failing tests `tests/test_pipeline.py`**

```python
import asyncio

import pytest
import torch

from imagehub import pipeline
from imagehub.backends.base import TextToImageBackend
from imagehub.exceptions import BackendBusyError, BackendContractError
from imagehub.image import Image
from imagehub.pipelines import TextToImagePipeline
from imagehub.registry import BackendRegistry
from imagehub.types import ImageResult, T2IRequest


class Recording(TextToImageBackend):
    def __init__(self, fail_loads=0):
        super().__init__()
        self.loads = self.unloads = 0
        self.fail_loads = fail_loads
        self.calls = []

    def validate_options(self, inputs, options):
        merged = dict(inputs.parameters) | options
        unknown = set(merged) - {"seed"}
        if unknown:
            raise TypeError(f"Unknown options: {sorted(unknown)}")
        return merged

    def _load(self):
        self.loads += 1
        if self.fail_loads:
            self.fail_loads -= 1
            raise RuntimeError("load failed")

    def _unload(self):
        self.unloads += 1

    def _infer(self, inputs, **options):
        self.calls.append((inputs.prompt, options))
        pixels = torch.zeros(2, 2, 3, dtype=torch.uint8)
        return ImageResult([Image(pixels)], {"prompt": inputs.prompt} | options)


def test_run_with_backend_instance_and_option_precedence():
    backend = Recording()
    pipe = pipeline("t2i", backend=backend, options={"seed": 5})
    assert isinstance(pipe, TextToImagePipeline)
    assert pipe("a cat").metadata == {"prompt": "a cat", "seed": 5}
    assert pipe("a cat", seed=6).metadata["seed"] == 6
    assert pipe(T2IRequest("a dog", {"seed": 7})).metadata["seed"] == 5
    assert backend.loads == 1


def test_options_are_validated_before_loading():
    backend = Recording()
    pipe = pipeline("t2i", backend=backend)
    with pytest.raises(TypeError, match="stepz"):
        pipe("a cat", stepz=3)
    assert not backend.is_loaded
    with pytest.raises(ValueError):
        pipe("   ")
    with pytest.raises(TypeError):
        pipe(42)


def test_lifecycle_batch_map_and_async():
    backend = Recording()
    with pipeline("t2i", backend=backend) as pipe:
        assert backend.is_loaded
        assert [r.metadata["prompt"] for r in pipe.batch(["a", "b"])] == ["a", "b"]
        assert [r.metadata["prompt"] for r in pipe.map(iter(["c"]))] == ["c"]
        assert asyncio.run(pipe.arun("d")).metadata["prompt"] == "d"
    assert not backend.is_loaded and backend.unloads == 1


def test_failed_load_is_cleaned_up_and_retryable():
    backend = Recording(fail_loads=1)
    pipe = pipeline("t2i", backend=backend)
    with pytest.raises(RuntimeError, match="load failed"):
        pipe("a")
    assert not backend.is_loaded and backend.unloads == 1
    assert pipe("a").metadata["prompt"] == "a"


def test_nested_inference_is_rejected():
    class Nested(Recording):
        def _infer(self, inputs, **options):
            return self.infer(inputs)

    with pytest.raises(BackendBusyError):
        pipeline("t2i", backend=Nested())("a")


def test_contract_violation():
    class Broken(Recording):
        def _infer(self, inputs, **options):
            return "not a result"

    with pytest.raises(BackendContractError):
        pipeline("t2i", backend=Broken())("a")


def test_registry_selection():
    registry = BackendRegistry()
    registry.register("t2i", "rec", Recording)
    registry.register("t2i", "bad", lambda: object())
    assert isinstance(pipeline("t2i", backend="rec", registry=registry).backend, Recording)
    with pytest.raises(TypeError, match="must return"):
        pipeline("t2i", backend="bad", registry=registry)
    with pytest.raises(ValueError, match="construction options"):
        pipeline("t2i", backend=Recording(), device="cpu")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ImportError: cannot import name 'pipeline' from 'imagehub'`.

- [ ] **Step 3: Write `src/imagehub/backends/__init__.py` and `src/imagehub/backends/base.py`**

`src/imagehub/backends/__init__.py`:
```python
"""Model adapters: lazy loading, option validation and inference for one architecture family."""

from imagehub.backends.base import Backend, TextToImageBackend

__all__ = ["Backend", "TextToImageBackend"]
```

`src/imagehub/backends/base.py`:
```python
"""Shared task contracts and model lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from threading import RLock
from typing import Any, Generic, Self, TypeVar

from imagehub.exceptions import BackendBusyError
from imagehub.types import ImageResult, T2IRequest, Task

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Backend(ABC, Generic[InputT, OutputT]):
    """A lazy, reusable model with serialized inference per instance.

    Implement ``_infer`` and optionally ``_load``, ``_unload`` and ``validate_options``.
    Options are validated before any model is loaded. A failed load is cleaned up and can
    be retried. Exceptions from inference keep their original type and traceback.
    """

    task: Task

    def __init__(self) -> None:
        self._loaded = False
        self._active = False
        self._lock = RLock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> Self:
        with self._lock:
            if not self._loaded:
                try:
                    self._load()
                except BaseException:
                    # Cleanup must not replace the original model-loading error.
                    with suppress(Exception):
                        self._unload()
                    raise
                self._loaded = True
        return self

    def unload(self) -> None:
        with self._lock:
            if self._active:
                raise BackendBusyError("Cannot unload a backend while it is running an inference")
            if self._loaded:
                try:
                    self._unload()
                finally:
                    self._loaded = False

    @contextmanager
    def _session(self) -> Iterator[None]:
        with self._lock:
            if self._active:
                raise BackendBusyError("A backend cannot run nested inferences")
            self.load()
            self._active = True
            try:
                yield
            finally:
                self._active = False

    def validate_options(self, inputs: InputT, options: dict[str, Any]) -> dict[str, Any]:
        """Check and normalize request options without loading the model."""
        return options

    def infer(self, inputs: InputT, **options: Any) -> OutputT:
        validated = self.validate_options(inputs, dict(options))
        with self._session():
            return self._infer(inputs, **validated)

    def infer_batch(self, inputs: Sequence[InputT], **options: Any) -> list[OutputT]:
        if len(inputs) == 0:
            return []
        validated = [self.validate_options(item, dict(options)) for item in inputs]
        with self._session():
            return [self._infer(item, **opts) for item, opts in zip(inputs, validated, strict=True)]

    def _load(self) -> None:
        """Allocate model resources. Called once before the first inference."""

    def _unload(self) -> None:
        """Release owned references, including partially loaded resources."""

    @abstractmethod
    def _infer(self, inputs: InputT, **options: Any) -> OutputT:
        """Run one request and return the task's normalized result."""

    def __enter__(self) -> Self:
        return self.load()

    def __exit__(self, *exc: Any) -> None:
        self.unload()


class TextToImageBackend(Backend[T2IRequest, ImageResult]):
    task = Task.T2I
```

- [ ] **Step 4: Write the pipelines package**

`src/imagehub/pipelines/base.py`:
```python
"""A small, overridable preprocess → forward → postprocess pipeline."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Generic, Self, TypeVar

from imagehub.backends.base import Backend
from imagehub.exceptions import BackendContractError
from imagehub.types import Task

InputT = TypeVar("InputT")
PreparedT = TypeVar("PreparedT")
OutputT = TypeVar("OutputT")


class Pipeline(ABC, Generic[InputT, PreparedT, OutputT]):
    """Reusable task orchestration independent of the model's runtime.

    Call options override ``default_options`` with a shallow merge. Model construction
    options belong to the backend, never to the request defaults.
    """

    task: Task
    result_type: type[Any]

    def __init__(
        self,
        backend: Backend[PreparedT, OutputT],
        *,
        default_options: Mapping[str, Any] | None = None,
    ) -> None:
        if backend.task != self.task:
            raise ValueError(f"{self.task.value} pipeline cannot use a {backend.task.value} backend")
        self.backend = backend
        self.default_options = dict(default_options or {})

    @abstractmethod
    def preprocess(self, inputs: InputT) -> PreparedT:
        """Validate and normalize public inputs without loading a model."""

    def forward(self, inputs: PreparedT, **options: Any) -> OutputT:
        return self.backend.infer(inputs, **options)

    def postprocess(self, output: OutputT) -> OutputT:
        if not isinstance(output, self.result_type):
            raise BackendContractError(
                f"{self.task.value} backend must return {self.result_type.__name__}, "
                f"received {type(output).__name__}"
            )
        return output

    def run(self, inputs: InputT, **options: Any) -> OutputT:
        prepared = self.preprocess(inputs)
        result = self.forward(prepared, **(self.default_options | options))
        return self.postprocess(result)

    def __call__(self, inputs: InputT, **options: Any) -> OutputT:
        return self.run(inputs, **options)

    def batch(self, inputs: Iterable[InputT], **options: Any) -> list[OutputT]:
        """Materialize one batch; requests run sequentially inside one backend session."""
        prepared = [self.preprocess(item) for item in inputs]
        outputs = self.backend.infer_batch(prepared, **(self.default_options | options))
        return [self.postprocess(output) for output in outputs]

    def map(self, inputs: Iterable[InputT], **options: Any) -> Iterator[OutputT]:
        """Lazily process independent requests with bounded memory."""
        for item in inputs:
            yield self.run(item, **options)

    async def arun(self, inputs: InputT, **options: Any) -> OutputT:
        """Run off the event loop; cancellation does not stop an already running model."""
        return await asyncio.to_thread(self.run, inputs, **options)

    def load(self) -> Self:
        self.backend.load()
        return self

    def close(self) -> None:
        """Unload resources; a subsequent request may load them again."""
        self.backend.unload()

    def __enter__(self) -> Self:
        return self.load()

    def __exit__(self, *exc: Any) -> None:
        self.close()
```

`src/imagehub/pipelines/text_to_image.py`:
```python
from __future__ import annotations

from imagehub.pipelines.base import Pipeline
from imagehub.types import ImageResult, T2IRequest, Task


class TextToImagePipeline(Pipeline[str | T2IRequest, T2IRequest, ImageResult]):
    task = Task.T2I
    result_type = ImageResult

    def preprocess(self, inputs: str | T2IRequest) -> T2IRequest:
        if isinstance(inputs, T2IRequest):
            return inputs
        if isinstance(inputs, str):
            return T2IRequest(inputs)
        raise TypeError("t2i input must be a prompt string or a T2IRequest")
```

`src/imagehub/pipelines/__init__.py`:
```python
from imagehub.pipelines.base import Pipeline
from imagehub.pipelines.text_to_image import TextToImagePipeline

__all__ = ["Pipeline", "TextToImagePipeline"]
```

- [ ] **Step 5: Write `src/imagehub/factory.py`**

```python
from collections.abc import Mapping
from typing import Any, Literal, overload

from imagehub.backends.base import Backend
from imagehub.pipelines import Pipeline, TextToImagePipeline
from imagehub.registry import BackendRegistry, backend_registry
from imagehub.types import Task

_PIPELINES: dict[Task, type[Pipeline[Any, Any, Any]]] = {Task.T2I: TextToImagePipeline}


@overload
def pipeline(
    task: Literal["t2i", Task.T2I],
    *,
    backend: str | Backend[Any, Any],
    options: Mapping[str, Any] | None = None,
    registry: BackendRegistry | None = None,
    **backend_options: Any,
) -> TextToImagePipeline: ...


@overload
def pipeline(
    task: Task | str,
    *,
    backend: str | Backend[Any, Any],
    options: Mapping[str, Any] | None = None,
    registry: BackendRegistry | None = None,
    **backend_options: Any,
) -> Pipeline[Any, Any, Any]: ...


def pipeline(
    task: Task | str,
    *,
    backend: str | Backend[Any, Any],
    options: Mapping[str, Any] | None = None,
    registry: BackendRegistry | None = None,
    **backend_options: Any,
) -> Pipeline[Any, Any, Any]:
    """Create a task pipeline from a registered name or an existing backend.

    ``backend_options`` configure model construction (``model``, ``device``, ``dtype``);
    ``options`` supplies request defaults. Construction never downloads or loads weights.
    """
    selected = Task(task)
    if isinstance(backend, str):
        instance = (registry or backend_registry).create(selected, backend, **backend_options)
    else:
        if backend_options:
            raise ValueError("Model construction options cannot be used with an existing backend")
        instance = backend
    return _PIPELINES[selected](instance, default_options=options)
```

- [ ] **Step 6: Export from `src/imagehub/__init__.py`**

```python
"""imagehub: open-weight image generation architectures in native PyTorch."""

from imagehub.backends.base import Backend, TextToImageBackend
from imagehub.catalog import ModelSpec, available_models, model_info
from imagehub.factory import pipeline
from imagehub.image import Image
from imagehub.pipelines import Pipeline, TextToImagePipeline
from imagehub.registry import BackendRegistry, available_backends, backend_registry
from imagehub.types import ImageResult, T2IRequest, Task

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "BackendRegistry",
    "Image",
    "ImageResult",
    "ModelSpec",
    "Pipeline",
    "T2IRequest",
    "Task",
    "TextToImageBackend",
    "TextToImagePipeline",
    "__version__",
    "available_backends",
    "available_models",
    "backend_registry",
    "model_info",
    "pipeline",
]
```

- [ ] **Step 7: Run tests**

Run: `.venv/bin/python -m pytest tests -v`
Expected: all pass (reference tests skipped).

- [ ] **Step 8: Lint and commit**

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
git add src tests
git commit -m "Add backend lifecycle, text-to-image pipeline and pipeline() factory"
```

---

### Task 12: Stable Diffusion backend

**Files:**
- Create: `src/imagehub/backends/stable_diffusion.py`, `tests/test_stable_diffusion.py`, `tests/reference/test_ref_tiny_pipeline.py`
- Modify: `tests/helpers.py` (append)

**Interfaces:**
- Consumes: everything above. Registry path `imagehub.backends.stable_diffusion:StableDiffusionBackend` (Task 4).
- Produces:
  - `OPTION_DEFAULTS = {"negative_prompt": None, "steps": 50, "guidance": 7.5, "width": 512, "height": 512, "num_images": 1, "seed": None, "latents": None}`.
  - `resolve_device(device) -> torch.device` (raises `ValueError` listing available devices).
  - `initial_latents(seed, shape) -> Tensor` (CPU float32, identical to diffusers with `torch.Generator("cpu").manual_seed(seed)`).
  - `StableDiffusionComponents` dataclass (`tokenizer`, `text_encoder`, `unet`, `vae`, `scheduler_config`) with `classmethod load(directory, *, device, dtype)`.
  - `encode_prompt(components, prompt, device) -> Tensor`, `denoise(components, *, prompt, negative_prompt, steps, guidance, latents) -> Tensor` (final latents), `decode_latents(components, latents) -> Tensor` (uint8 `(B, H, W, 3)` on CPU).
  - `StableDiffusionBackend(model, *, device="cpu", dtype=None)`; metadata keys `model_id, revision, seed, steps, guidance, width, height, negative_prompt, scheduler, device, dtype`.
  - `tests/helpers.py`: `write_tiny_sd(directory, *, seed=0) -> Path` (a diffusers-loadable tiny snapshot with random weights).

- [ ] **Step 1: Append to `tests/helpers.py`**

```python
TINY_TEXT_FILE_CONFIG = TINY_TEXT_CONFIG | {
    "architectures": ["CLIPTextModel"],
    "model_type": "clip_text_model",
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
    "bos_token_id": 515,
    "eos_token_id": 516,
    "pad_token_id": 516,
    "projection_dim": 32,
}
TINY_VAE_FILE_CONFIG = TINY_VAE_CONFIG | {
    "_class_name": "AutoencoderKL",
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "layers_per_block": 2,
    "act_fn": "silu",
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "sample_size": 512,
}
TINY_UNET_FILE_CONFIG = TINY_UNET_CONFIG | {"_class_name": "UNet2DConditionModel"}
MODEL_INDEX = {
    "_class_name": "StableDiffusionPipeline",
    "feature_extractor": [None, None],
    "safety_checker": [None, None],
    "scheduler": ["diffusers", "PNDMScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "unet": ["diffusers", "UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}


def write_tiny_sd(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight SD 1.x snapshot in the published diffusers layout (loads in diffusers too)."""
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.clip_text import CLIPTextModel
    from imagehub.models.unet_2d import UNet2DConditionModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    write_tiny_tokenizer(root / "tokenizer")
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(SD1_SCHEDULER))
    (root / "model_index.json").write_text(json.dumps(MODEL_INDEX))
    for name, cls, config, weights in (
        ("text_encoder", CLIPTextModel, TINY_TEXT_FILE_CONFIG, "model.safetensors"),
        ("unet", UNet2DConditionModel, TINY_UNET_FILE_CONFIG, "diffusion_pytorch_model.safetensors"),
        ("vae", AutoencoderKL, TINY_VAE_FILE_CONFIG, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        state = dict(model.state_dict())
        if name == "text_encoder":
            state["text_model.embeddings.position_ids"] = torch.arange(77)[None]
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(state, root / name / weights)
    return root
```

- [ ] **Step 2: Write failing tests `tests/test_stable_diffusion.py`**

```python
import json

import pytest
import torch

from helpers import write_tiny_sd
from imagehub import pipeline
from imagehub.backends.stable_diffusion import (
    StableDiffusionBackend,
    initial_latents,
    resolve_device,
)
from imagehub.exceptions import CheckpointError
from imagehub.pretrained import SNAPSHOT_FILE


@pytest.fixture(scope="module")
def tiny_sd(tmp_path_factory):
    return write_tiny_sd(tmp_path_factory.mktemp("tiny-sd"))


def test_generates_images_with_metadata(tiny_sd):
    with pipeline("t2i", backend="stable-diffusion", model=tiny_sd) as pipe:
        result = pipe("a cat", seed=3, steps=4, width=64, height=48, num_images=2)
    assert len(result.images) == 2
    assert (result.images[0].width, result.images[0].height) == (64, 48)
    meta = result.metadata
    assert meta["seed"] == 3 and meta["steps"] == 4 and meta["scheduler"] == "pndm"
    assert meta["model_id"] is None and meta["revision"] is None
    assert meta["device"] == "cpu" and meta["dtype"] == "float32"


def test_seed_is_reproducible_and_reported(tiny_sd):
    pipe = pipeline("t2i", backend="sd-v1-5", model=tiny_sd)
    a = pipe("a cat", seed=11, steps=3, width=32, height=32)
    b = pipe("a cat", seed=11, steps=3, width=32, height=32)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    random = pipe("a cat", steps=3, width=32, height=32)
    assert isinstance(random.metadata["seed"], int)


def test_latents_option_matches_seed(tiny_sd):
    pipe = pipeline("t2i", backend="stable-diffusion", model=tiny_sd)
    by_seed = pipe("a cat", seed=5, steps=3, width=32, height=32)
    latents = initial_latents(5, (1, 4, 4, 4))
    by_latents = pipe("a cat", latents=latents, steps=3, width=32, height=32)
    assert torch.equal(by_seed.images[0].tensor, by_latents.images[0].tensor)
    assert by_latents.metadata["seed"] is None


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 30}, ValueError, "multiple of 8"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1, "latents": torch.zeros(1, 4, 64, 64)}, ValueError, "either"),
        ({"latents": torch.zeros(1, 4, 8, 8)}, ValueError, "shape"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
    ],
)
def test_invalid_options_fail_before_loading(tiny_sd, options, error, match):
    backend = StableDiffusionBackend(tiny_sd)
    with pytest.raises(error, match=match):
        pipeline("t2i", backend=backend)("a cat", **options)
    assert not backend.is_loaded


def test_missing_snapshot_points_to_pretrained(tmp_path):
    pipe = pipeline("t2i", backend="sd-v1-5", model=tmp_path / "nowhere")
    with pytest.raises(CheckpointError, match="python -m imagehub.pretrained"):
        pipe("a cat")


def test_snapshot_metadata(tiny_sd, tmp_path):
    import shutil

    copy = shutil.copytree(tiny_sd, tmp_path / "copy")
    manifest = {"model_id": "sd-v1-5", "revision": "f" * 40, "files": {}}
    (copy / SNAPSHOT_FILE).write_text(json.dumps(manifest))
    result = pipeline("t2i", backend="sd-v1-5", model=copy)("a cat", seed=0, steps=2, width=32, height=32)
    assert result.metadata["model_id"] == "sd-v1-5"
    assert result.metadata["revision"] == "f" * 40


def test_device_and_dtype_validation(tiny_sd):
    with pytest.raises(ValueError, match="Available: cpu"):
        resolve_device("quantum")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="not available"):
            StableDiffusionBackend(tiny_sd, device="cuda")
    with pytest.raises(ValueError, match="dtype"):
        StableDiffusionBackend(tiny_sd, dtype=torch.int8)
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_stable_diffusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imagehub.backends.stable_diffusion'`.

- [ ] **Step 4: Write `src/imagehub/backends/stable_diffusion.py`**

```python
"""Stable Diffusion 1.x text-to-image backend over a local diffusers-layout snapshot.

The sampling order (prompt encoding, classifier-free guidance batch order, noise scaling,
PLMS steps, VAE decoding and uint8 rounding) follows diffusers' StableDiffusionPipeline.
"""

from __future__ import annotations

import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.checkpoints import read_json
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.pretrained import read_snapshot
from imagehub.schedulers.pndm import PNDMScheduler
from imagehub.tokenization import CLIPTokenizer
from imagehub.types import ImageResult, T2IRequest

COMPONENTS = ("tokenizer", "text_encoder", "unet", "vae", "scheduler")
OPTION_DEFAULTS: dict[str, Any] = {
    "negative_prompt": None,
    "steps": 50,
    "guidance": 7.5,
    "width": 512,
    "height": 512,
    "num_images": 1,
    "seed": None,
    "latents": None,
}
LATENT_CHANNELS = 4
LATENT_SCALE = 8
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def resolve_device(device: str | torch.device) -> torch.device:
    available = ["cpu"]
    if torch.cuda.is_available():
        available.append("cuda")
    if torch.backends.mps.is_available():
        available.append("mps")
    try:
        selected = torch.device(device)
    except RuntimeError:
        raise ValueError(f"Unknown device {device!r}. Available: {', '.join(available)}") from None
    if selected.type not in available:
        raise ValueError(
            f"Device {str(selected)!r} is not available. Available: {', '.join(available)}"
        )
    return selected


def initial_latents(seed: int, shape: tuple[int, ...]) -> Tensor:
    """Noise drawn exactly like diffusers with ``generator=torch.Generator("cpu").manual_seed(seed)``."""
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def _hint(root: Path) -> str:
    return f"Download a snapshot with: python -m imagehub.pretrained sd-v1-5 {root}"


@dataclass
class StableDiffusionComponents:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    unet: UNet2DConditionModel
    vae: AutoencoderKL
    scheduler_config: dict[str, Any]

    @classmethod
    def load(
        cls, directory: str | os.PathLike[str], *, device: torch.device, dtype: torch.dtype
    ) -> StableDiffusionComponents:
        root = Path(directory)
        missing = [name for name in COMPONENTS if not (root / name).is_dir()]
        if missing:
            raise CheckpointError(
                f"{root} is not a Stable Diffusion snapshot (missing {', '.join(missing)}). "
                + _hint(root)
            )
        try:
            return cls(
                tokenizer=CLIPTokenizer(root / "tokenizer"),
                text_encoder=CLIPTextModel.from_pretrained(
                    root / "text_encoder", device=device, dtype=dtype
                ),
                unet=UNet2DConditionModel.from_pretrained(root / "unet", device=device, dtype=dtype),
                vae=AutoencoderKL.from_pretrained(root / "vae", device=device, dtype=dtype),
                scheduler_config=read_json(root / "scheduler" / "scheduler_config.json"),
            )
        except FileNotFoundError as error:
            raise CheckpointError(f"{error.filename} is missing. " + _hint(root)) from error


@torch.inference_mode()
def encode_prompt(components: StableDiffusionComponents, prompt: str, device: torch.device) -> Tensor:
    ids = torch.tensor([components.tokenizer(prompt)], device=device)
    return components.text_encoder(ids)


@torch.inference_mode()
def denoise(
    components: StableDiffusionComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance: float,
    latents: Tensor,
) -> Tensor:
    """Run the PLMS loop from initial noise ``latents`` (on the model device); return final latents."""
    dtype = components.unet.conv_in.weight.dtype
    batch = latents.shape[0]
    cond = encode_prompt(components, prompt, latents.device).to(dtype).repeat(batch, 1, 1)
    use_cfg = guidance > 1.0
    if use_cfg:
        uncond = encode_prompt(components, negative_prompt or "", latents.device)
        context = torch.cat([uncond.to(dtype).repeat(batch, 1, 1), cond])
    else:
        context = cond
    scheduler = PNDMScheduler(components.scheduler_config)
    scheduler.set_timesteps(steps)
    latents = latents.to(dtype) * scheduler.init_noise_sigma
    for t in scheduler.timesteps:
        model_input = torch.cat([latents] * 2) if use_cfg else latents
        noise = components.unet(model_input, t, context)
        if use_cfg:
            noise_uncond, noise_text = noise.chunk(2)
            noise = noise_uncond + guidance * (noise_text - noise_uncond)
        latents = scheduler.step(noise, t, latents)
    return latents


@torch.inference_mode()
def decode_latents(components: StableDiffusionComponents, latents: Tensor) -> Tensor:
    """Decode final latents to uint8 images of shape (batch, height, width, 3) on CPU."""
    vae = components.vae
    image = vae.decode(latents / vae.scaling_factor)
    image = (image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu()
    return (image * 255).round().to(torch.uint8)


def _positive_int(options: dict[str, Any], name: str) -> None:
    value = options[name]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")


class StableDiffusionBackend(TextToImageBackend):
    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.source = Path(model)
        self.device = resolve_device(device)
        self.dtype = dtype or torch.float32
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be torch.float32, torch.float16 or torch.bfloat16")
        self.components: StableDiffusionComponents | None = None
        self.snapshot: dict[str, Any] | None = None

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown Stable Diffusion options: {', '.join(unknown)}")
        o = OPTION_DEFAULTS | merged
        if o["negative_prompt"] is not None and not isinstance(o["negative_prompt"], str):
            raise TypeError("negative_prompt must be a string or None")
        for name in ("steps", "num_images", "width", "height"):
            _positive_int(o, name)
        if o["steps"] > 1000:
            raise ValueError("steps must be at most 1000")
        for name in ("width", "height"):
            if o[name] % LATENT_SCALE:
                raise ValueError(f"{name} must be a multiple of {LATENT_SCALE}")
        guidance = o["guidance"]
        if (
            isinstance(guidance, bool)
            or not isinstance(guidance, int | float)
            or not math.isfinite(guidance)
            or guidance < 0
        ):
            raise ValueError("guidance must be a finite number >= 0")
        seed, latents = o["seed"], o["latents"]
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be an int in [0, 2**64)")
        if latents is not None:
            if seed is not None:
                raise ValueError("Pass either seed or latents, not both")
            shape = (
                o["num_images"],
                LATENT_CHANNELS,
                o["height"] // LATENT_SCALE,
                o["width"] // LATENT_SCALE,
            )
            if (
                not isinstance(latents, Tensor)
                or not latents.is_floating_point()
                or tuple(latents.shape) != shape
            ):
                raise ValueError(f"latents must be a floating-point tensor of shape {shape}")
        return o

    def _load(self) -> None:
        self.components = StableDiffusionComponents.load(
            self.source, device=self.device, dtype=self.dtype
        )
        self.snapshot = read_snapshot(self.source)

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _infer(self, inputs: T2IRequest, **o: Any) -> ImageResult:
        assert self.components is not None
        seed, latents = o["seed"], o["latents"]
        if latents is None:
            if seed is None:
                seed = secrets.randbits(63)
            shape = (
                o["num_images"],
                LATENT_CHANNELS,
                o["height"] // LATENT_SCALE,
                o["width"] // LATENT_SCALE,
            )
            latents = initial_latents(seed, shape)
        final = denoise(
            self.components,
            prompt=inputs.prompt,
            negative_prompt=o["negative_prompt"],
            steps=o["steps"],
            guidance=float(o["guidance"]),
            latents=latents.to(torch.float32).to(self.device),
        )
        pixels = decode_latents(self.components, final)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "steps": o["steps"],
            "guidance": float(o["guidance"]),
            "width": o["width"],
            "height": o["height"],
            "negative_prompt": o["negative_prompt"],
            "scheduler": "pndm",
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(p) for p in pixels), metadata)
```

- [ ] **Step 5: Write `tests/reference/test_ref_tiny_pipeline.py`**

Loads the same tiny snapshot with diffusers' own loader (exercising legacy VAE names and `position_ids` on their side too) and compares whole generations.

```python
import pytest
import torch

from helpers import write_tiny_sd
from imagehub import pipeline

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")


@pytest.mark.parametrize(
    "case",
    [
        dict(prompt="a cat", negative_prompt=None, steps=50, guidance=7.5, num=1, hw=(64, 64), seed=42),
        dict(prompt="a dog", negative_prompt="blurry", steps=20, guidance=7.5, num=2, hw=(48, 72), seed=7),
        dict(prompt="no cfg", negative_prompt=None, steps=10, guidance=1.0, num=1, hw=(64, 64), seed=1),
    ],
    ids=["cfg", "negative-batch", "no-cfg"],
)
def test_tiny_snapshot_generation_matches_diffusers(tmp_path, case):
    root = write_tiny_sd(tmp_path / "tiny")
    ref = diffusers.StableDiffusionPipeline.from_pretrained(
        root, safety_checker=None, requires_safety_checker=False, local_files_only=True
    )
    ref.set_progress_bar_config(disable=True)
    h, w = case["hw"]
    expected = ref(
        case["prompt"],
        negative_prompt=case["negative_prompt"],
        num_inference_steps=case["steps"],
        guidance_scale=case["guidance"],
        num_images_per_prompt=case["num"],
        height=h,
        width=w,
        generator=torch.Generator("cpu").manual_seed(case["seed"]),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    result = pipeline("t2i", backend="stable-diffusion", model=root)(
        case["prompt"],
        negative_prompt=case["negative_prompt"],
        steps=case["steps"],
        guidance=case["guidance"],
        num_images=case["num"],
        height=h,
        width=w,
        seed=case["seed"],
    )
    actual = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual, expected)
```

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests -v
IMAGEHUB_REFERENCE_TESTS=1 .venv/bin/python -m pytest tests/reference -v
```
Expected: all normal tests pass; all reference tests pass except the checkpoint-gated ones, which skip.

- [ ] **Step 7: Lint and commit**

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
git add src tests
git commit -m "Add Stable Diffusion 1.x backend with diffusers-ordered PLMS sampling"
```

---

### Task 13: Checkpoint parity and device smoke tests

**Files:**
- Create: `tests/reference/test_ref_sd1.py`, `tests/reference/test_ref_devices.py`

**Interfaces:**
- Consumes: `StableDiffusionComponents`, `denoise`, `decode_latents`, `initial_latents` (Task 12); `checkpoint_dir`, `max_diff` (Task 6); fixture `record` (Task 6).
- Produces: measurements recorded under keys `text_encoder`, `unet`, `vae_decode`, `vae_encode`, `seed_noise`, `e2e_sd-v1-<n>` (n = 1…5), `seed_e2e_sd-v1-5`, `smoke_mps`, `smoke_cuda`. Each e2e record has `latents` (`max_abs`, `max_rel`), `pixels_max_abs` (int) and `passed` (bool). Task 14's `scripts/build_validation.py` reads exactly these keys.

- [ ] **Step 1: Write `tests/reference/test_ref_sd1.py`**

```python
"""Parity of the native SD 1.x components and pipeline against diffusers on CPU fp32.

Requires IMAGEHUB_REFERENCE_TESTS=1 and IMAGEHUB_SD_V1_<n> pointing at snapshots made by
``python -m imagehub.pretrained sd-v1-<n> <dir>``.
"""

import pytest
import torch

from helpers import checkpoint_dir, max_diff
from imagehub import pipeline
from imagehub.backends.stable_diffusion import (
    StableDiffusionComponents,
    decode_latents,
    denoise,
    initial_latents,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

# Maximum absolute difference allowed against diffusers 0.40.0 / transformers 5.18.0, CPU fp32.
# Rule (spec §6.2): 0 if the measured difference is 0, otherwise the next power of ten above it.
# Random-weight runs at full SD 1.x size measured exactly 0 for every entry on 2026-10-04.
TOLERANCES = {
    "text_encoder": 0.0,
    "unet": 0.0,
    "vae_decode": 0.0,
    "vae_encode": 0.0,
    "e2e_latents": 0.0,
    "e2e_pixels": 0,
}
PROMPT = "a photograph of an astronaut riding a horse"
CPU = torch.device("cpu")


def load_pair(version):
    root = checkpoint_dir(version)
    ours = StableDiffusionComponents.load(root, device=CPU, dtype=torch.float32)
    ref = diffusers.StableDiffusionPipeline.from_pretrained(
        root, safety_checker=None, requires_safety_checker=False, local_files_only=True
    )
    ref.set_progress_bar_config(disable=True)
    return root, ours, ref


@pytest.fixture(scope="module")
def sd14():
    return load_pair(4)


def check(record, name, actual, expected, tolerance_key):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=TOLERANCES[tolerance_key])
    torch.testing.assert_close(actual, expected, atol=TOLERANCES[tolerance_key], rtol=0)


def test_text_encoder(sd14, record):
    _, ours, ref = sd14
    prompts = [PROMPT, "", "Bir kedi, yağlı boya", "emoji 🐱🚀 " * 30]
    ids = torch.tensor([ours.tokenizer(p) for p in prompts])
    with torch.inference_mode():
        check(record, "text_encoder", ours.text_encoder(ids), ref.text_encoder(ids)[0], "text_encoder")


def test_unet_single_step(sd14, record):
    _, ours, ref = sd14
    g = torch.Generator().manual_seed(0)
    sample = torch.randn(2, 4, 64, 64, generator=g)
    ids = torch.tensor([ours.tokenizer(""), ours.tokenizer(PROMPT)])
    with torch.inference_mode():
        context = ref.text_encoder(ids)[0]
        worst = None
        for t in (981, 501, 1):
            timestep = torch.tensor(t)
            actual = ours.unet(sample, timestep, context)
            expected = ref.unet(sample, timestep, context).sample
            measured = max_diff(actual, expected)
            if worst is None or measured["max_abs"] > worst["max_abs"]:
                worst = measured
            torch.testing.assert_close(actual, expected, atol=TOLERANCES["unet"], rtol=0)
    record("unet", **worst, tolerance=TOLERANCES["unet"], timesteps=[981, 501, 1])


def test_vae(sd14, record):
    _, ours, ref = sd14
    g = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 4, 64, 64, generator=g)
    image = torch.rand(1, 3, 512, 512, generator=g) * 2 - 1
    with torch.inference_mode():
        check(record, "vae_decode", ours.vae.decode(latents), ref.vae.decode(latents).sample, "vae_decode")
        check(
            record,
            "vae_encode",
            ours.vae.encode(image),
            ref.vae.encode(image).latent_dist.mean,
            "vae_encode",
        )


def test_seed_noise(record):
    from diffusers.utils.torch_utils import randn_tensor

    shape = (2, 4, 64, 64)
    expected = randn_tensor(shape, generator=torch.Generator("cpu").manual_seed(42), dtype=torch.float32)
    assert torch.equal(initial_latents(42, shape), expected)
    record("seed_noise", exact=True)


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_end_to_end(version, record):
    _, ours, ref = load_pair(version)
    latents = initial_latents(1234, (1, 4, 64, 64))
    with torch.inference_mode():
        ref_latents = ref(
            PROMPT, num_inference_steps=50, guidance_scale=7.5, latents=latents.clone(), output_type="latent"
        ).images
        decoded = ref.vae.decode(ref_latents / ref.vae.config.scaling_factor).sample
        ref_np = ref.image_processor.postprocess(decoded, output_type="np")
    ref_pixels = torch.from_numpy((ref_np * 255).round().astype("uint8"))
    our_latents = denoise(
        ours, prompt=PROMPT, negative_prompt=None, steps=50, guidance=7.5, latents=latents.clone()
    )
    our_pixels = decode_latents(ours, our_latents)
    latent_diff = max_diff(our_latents, ref_latents)
    pixel_diff = int((our_pixels.int() - ref_pixels.int()).abs().max())
    passed = (
        latent_diff["max_abs"] <= TOLERANCES["e2e_latents"]
        and pixel_diff <= TOLERANCES["e2e_pixels"]
    )
    record(
        f"e2e_sd-v1-{version}",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=50,
        guidance=7.5,
        size=[512, 512],
        noise_seed=1234,
        tolerance={"latents": TOLERANCES["e2e_latents"], "pixels": TOLERANCES["e2e_pixels"]},
    )
    assert passed, (latent_diff, pixel_diff)


def test_seed_end_to_end_v15(record):
    root, _, ref = load_pair(5)
    with torch.inference_mode():
        expected = ref(
            PROMPT, num_inference_steps=50, generator=torch.Generator("cpu").manual_seed(42), output_type="np"
        ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    actual = pipeline("t2i", backend="sd-v1-5", model=root)(PROMPT, seed=42).images[0].tensor
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    record("seed_e2e_sd-v1-5", pixels_max_abs=pixel_diff, seed=42, passed=pixel_diff <= TOLERANCES["e2e_pixels"])
    assert pixel_diff <= TOLERANCES["e2e_pixels"]
```

- [ ] **Step 2: Write `tests/reference/test_ref_devices.py`**

```python
"""MPS/CUDA smoke tests: they run and produce finite images. No parity claim."""

import math
import platform

import pytest
import torch

from helpers import checkpoint_dir, max_diff
from imagehub.backends.stable_diffusion import (
    StableDiffusionComponents,
    decode_latents,
    denoise,
    initial_latents,
)

pytestmark = pytest.mark.reference
PROMPT = "a photograph of an astronaut riding a horse"


def _available(device):
    return torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()


def _name(device):
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return platform.processor() or platform.machine()


@pytest.mark.parametrize(
    "device",
    [pytest.param("mps", marks=pytest.mark.mps), pytest.param("cuda", marks=pytest.mark.cuda)],
)
def test_device_smoke(device, record):
    if not _available(device):
        pytest.skip(f"{device} is not available")
    root = checkpoint_dir(5)
    noise = initial_latents(7, (1, 4, 64, 64))
    outputs = {}
    for name in ("cpu", device):
        components = StableDiffusionComponents.load(root, device=torch.device(name), dtype=torch.float32)
        final = denoise(
            components, prompt=PROMPT, negative_prompt=None, steps=20, guidance=7.5, latents=noise.to(name)
        )
        assert torch.isfinite(final).all()
        outputs[name] = (final.cpu(), decode_latents(components, final))
        del components
    (cpu_latents, cpu_pixels), (dev_latents, dev_pixels) = outputs["cpu"], outputs[device]
    assert dev_pixels.shape == (1, 512, 512, 3)
    mse = float(((cpu_pixels.double() - dev_pixels.double()) ** 2).mean())
    psnr = math.inf if mse == 0 else 10 * math.log10(255**2 / mse)
    record(
        f"smoke_{device}",
        device_name=_name(device),
        steps=20,
        latents=max_diff(dev_latents, cpu_latents),
        pixels_max_abs=int((dev_pixels.int() - cpu_pixels.int()).abs().max()),
        psnr_db=psnr if math.isfinite(psnr) else "inf",
    )
```

- [ ] **Step 3: Verify collection and gating (no checkpoints needed)**

```bash
.venv/bin/python -m pytest tests -q
IMAGEHUB_REFERENCE_TESTS=1 .venv/bin/python -m pytest tests/reference -q
```
Expected: first command — reference tests reported as skipped; second — checkpoint tests skip with "Set IMAGEHUB_SD_V1_…", random-weight and tiny-pipeline tests pass.

- [ ] **Step 4: Lint and commit**

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
git add tests
git commit -m "Add SD 1.x checkpoint parity and device smoke tests"
```

---

### Task 14: Run parity on the real checkpoints and record validation

This task downloads ~21 GB and runs long CPU jobs. It is executed in the main session (not a subagent) so results can be reviewed before they are written down.

**Files:**
- Create: `scripts/build_validation.py`, `docs/validation.json`, `docs/validation.md`, `docs/validation-measurements.json`
- Modify (only if a measurement is non-zero): `tests/reference/test_ref_sd1.py` (`TOLERANCES`)

**Interfaces:**
- Consumes: recorded keys from Task 13; `read_snapshot`, `MODEL_CATALOG`.
- Produces: `docs/validation.json` with `version, date, environment, models[], components, devices, limits, tests`.

- [ ] **Step 1: Download the five checkpoints**

```bash
mkdir -p checkpoints
for v in 1 2 3 4 5; do .venv/bin/python -m imagehub.pretrained sd-v1-$v checkpoints/sd-v1-$v || break; done
```
If it fails with "TLS certificate verification failed", run it with `SSL_CERT_FILE="$(.venv/bin/python -m certifi)"` (certifi comes with the reference extra) or run the Python installer's `Install Certificates.command`. `checkpoints/` is git-ignored. Each `imagehub_snapshot.json` must show `"verified_revision": true`.

- [ ] **Step 2: Run the CPU reference suite and record measurements**

```bash
export IMAGEHUB_REFERENCE_TESTS=1
for v in 1 2 3 4 5; do export IMAGEHUB_SD_V1_$v="$PWD/checkpoints/sd-v1-$v"; done
IMAGEHUB_RECORD=docs/validation-measurements.json .venv/bin/python -m pytest tests/reference -v -m "reference and not mps and not cuda" 2>&1 | tee /tmp/imagehub-reference.log
```
Expected: all pass with tolerance 0. Runtime is dominated by `test_end_to_end` (5 × two 51-step 512² runs).

- [ ] **Step 3: If any comparison is not exact, apply the tolerance rule — do not guess**


- [ ] **Step 4: Run the MPS smoke test**

```bash
IMAGEHUB_RECORD=docs/validation-measurements.json .venv/bin/python -m pytest tests/reference/test_ref_devices.py -v -m mps
```
Expected: pass; `smoke_mps` recorded with its CPU difference and PSNR.

- [ ] **Step 5: Write `scripts/build_validation.py`**

```python
"""Assemble docs/validation.json from reference-test measurements and local snapshots.

Usage:
  python scripts/build_validation.py --measurements docs/validation-measurements.json \
      --checkpoints checkpoints --machine "Apple M2 Max, 96 GB" --tests-log /tmp/imagehub-reference.log
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import re
import sys
from pathlib import Path

import torch

import imagehub
from imagehub.catalog import MODEL_CATALOG
from imagehub.pretrained import read_snapshot

LIMITS = [
    "Parity is claimed only on CPU in float32 against diffusers 0.40.0 and transformers 5.18.0.",
    "MPS and CUDA runs are smoke tests (finite output, shape, CPU difference recorded); "
    "no parity claim.",
    "No float16 or bfloat16 parity.",
    "No safety checker or invisible watermark; the reference ran with safety_checker=None.",
    "Only the PNDM (PLMS) scheduler shipped with the checkpoints is implemented.",
    "No image-quality benchmark (FID, CLIP score).",
    "Text-to-image only; SD 1.x img2img is not implemented.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--tests-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/validation.json"))
    args = parser.parse_args()

    import diffusers
    import transformers

    measured = json.loads(args.measurements.read_text(encoding="utf-8"))
    models = []
    for spec in MODEL_CATALOG.values():
        snapshot = read_snapshot(args.checkpoints / spec.id)
        if snapshot is None or snapshot["revision"] != spec.revision:
            sys.exit(f"{spec.id}: missing snapshot or not the validated revision")
        e2e = measured.get(f"e2e_{spec.id}")
        models.append(
            {
                "model_id": spec.id,
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "provenance": spec.provenance,
                "catalog_id": spec.catalog_id,
                "license": spec.license,
                "status": "full" if e2e and e2e.get("passed") else "experimental",
                "files": snapshot["files"],
                "end_to_end": e2e,
            }
        )
    summary = re.findall(r"=+ (.+) in [\d.]+s", args.tests_log.read_text(encoding="utf-8"))
    document = {
        "version": imagehub.__version__,
        "date": datetime.date.today().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "machine": args.machine,
            "platform": platform.platform(),
            **measured.get("environment", {}),
        },
        "models": models,
        "components": {
            key: value
            for key, value in measured.items()
            if key in ("tokenizer", "text_encoder", "unet", "vae_decode", "vae_encode", "seed_noise")
        },
        "seed_end_to_end": measured.get("seed_e2e_sd-v1-5"),
        "devices": {k: v for k, v in measured.items() if k.startswith("smoke_")},
        "limits": LIMITS,
        "tests": {"reference_summary": summary[-1] if summary else None},
    }
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Build `docs/validation.json`**

```bash
.venv/bin/python scripts/build_validation.py --measurements docs/validation-measurements.json \
  --checkpoints checkpoints --machine "Apple M2 Max, 96 GB unified memory" \
  --tests-log /tmp/imagehub-reference.log
.venv/bin/python -m pytest tests -q 2>&1 | tail -1
```
Check that all five models have `"status": "full"`. Note the normal-test summary line from the second command for `validation.md`.

- [ ] **Step 7: Write `docs/validation.md` (Turkish) from `docs/validation.json`**

Use this structure and fill every number from `docs/validation.json` (no rounding beyond 2 significant digits):

```markdown
# Doğrulama — imagehub 0.1

<tarih> tarihinde CPU üzerinde doğrulandı. Ortam: Python <python> / PyTorch <torch>;
referans: diffusers <diffusers>, transformers <transformers>; makine: <machine>; <torch_threads> CPU thread'i.
Kaynaklar, commit'ler ve dosya parmak izleri [validation.json](validation.json) dosyasındadır.

## Kapsam

- **5 checkpoint, tam ağırlık (`full`):** sd-v1-1 … sd-v1-5. Yayımlanmış tensörler strict yüklendi,
  aynı başlangıç gürültüsüyle 50 adımlık üretim diffusers ile karşılaştırıldı.
- Bu ifade kalite benchmark'ı anlamına gelmez.

## Karşılaştırmalar (CPU, float32)

| Karşılaştırma | Checkpoint | Gözlenen en büyük fark | Tolerans |
| --- | --- | --- | --- |
| Tokenizer token id'leri (11 prompt) | v1.4 | birebir | birebir |
| CLIP metin kodlayıcı | v1.4 | <text_encoder.max_abs> | <tolerance> |
| U-Net tek adım (t = 981, 501, 1) | v1.4 | <unet.max_abs> | <tolerance> |
| VAE decode / encode | v1.4 | <vae_decode.max_abs> / <vae_encode.max_abs> | <tolerance> |
| Seed'den başlangıç gürültüsü | — | birebir | birebir |
| Uçtan uca, son latent / piksel | v1.1 … v1.5 (ayrı satırlar) | <latents.max_abs> / <pixels_max_abs> | <tolerance> |
| Uçtan uca, seed 42 | v1.5 | <pixels_max_abs> | <tolerance> |

## Cihaz duman testleri (parity iddiası yok)

| Cihaz | Sonuç | CPU'ya göre en büyük latent farkı | PSNR |
| --- | --- | --- | --- |
| MPS (<device_name>) | sonlu, 512×512 | <max_abs> | <psnr_db> dB |
| CUDA | <Task 16'dan sonra doldurulur: RunPod sonucu> | | |

## Sınırlar

<validation.json "limits" listesinin Türkçesi, madde madde>

## Otomatik testler

- Normal testler: <pytest özeti>. Ağdan model indirmez.
- Referans testleri: <reference_summary>.
```
The CUDA row is filled in Task 16 Step 3; until then write "henüz çalıştırılmadı".

- [ ] **Step 8: Commit (measurements, tolerances if changed, script, docs)**

```bash
git add scripts/build_validation.py docs/validation.json docs/validation.md docs/validation-measurements.json tests/reference/test_ref_sd1.py
git commit -m "Record SD 1.x CPU parity and MPS smoke validation"
```

---

### Task 15: Documentation, notices and example

**Files:**
- Create: `README.md` (replace the 10-byte placeholder), `docs/catalog.md`, `docs/models.md`, `THIRD_PARTY_NOTICES.md`, `licenses/Diffusers-Apache-2.0.txt`, `licenses/Transformers-Apache-2.0.txt`, `licenses/OpenAI-CLIP-MIT.txt`, `examples/text_to_image.py`, `artifacts/sd-v1-5-seed42.png`, `tests/test_docs.py`
- Modify: `pyproject.toml` (`license-files`)

- [ ] **Step 1: Fetch third-party license texts**

```bash
mkdir -p licenses
curl -fsSL https://raw.githubusercontent.com/huggingface/diffusers/v0.40.0/LICENSE -o licenses/Diffusers-Apache-2.0.txt
curl -fsSL https://raw.githubusercontent.com/huggingface/transformers/v5.18.0/LICENSE -o licenses/Transformers-Apache-2.0.txt
curl -fsSL https://raw.githubusercontent.com/openai/CLIP/main/LICENSE -o licenses/OpenAI-CLIP-MIT.txt
head -2 licenses/*.txt
```
Expected: two Apache 2.0 headers and the MIT header (`MIT License` / `Copyright (c) 2021 OpenAI`). If a tag URL 404s, use the tag listed on the project's GitHub releases page for that version.

- [ ] **Step 2: Write `THIRD_PARTY_NOTICES.md`**

```markdown
# Third-party architecture and license notices

imagehub implements native PyTorch inference for the model families listed below. These are new
implementations written with reference to upstream architecture definitions and checkpoint layouts;
they do not invoke the upstream inference packages at runtime. Layer structure, parameter names,
tokenizer rules and sampling order follow those projects so that published weights load unchanged
and outputs can be compared with them.

| Project | Attribution | Relevant imagehub components | License |
| --- | --- | --- | --- |
| [diffusers](https://github.com/huggingface/diffusers) v0.40.0 | The HuggingFace Team and contributors | U-Net, KL autoencoder, PNDM scheduler, SD sampling order, checkpoint layout | [Apache 2.0](licenses/Diffusers-Apache-2.0.txt) |
| [transformers](https://github.com/huggingface/transformers) v5.18.0 | The HuggingFace Team, OpenAI and contributors | CLIP text encoder, CLIP tokenizer normalization and splitting | [Apache 2.0](licenses/Transformers-Apache-2.0.txt) |
| [CLIP](https://github.com/openai/CLIP) | 2021 OpenAI | Byte-level BPE vocabulary conventions | [MIT](licenses/OpenAI-CLIP-MIT.txt) |

Model weights are not distributed with imagehub. Stable Diffusion 1.x weights are released under the
CreativeML OpenRAIL-M license, whose use restrictions bind whoever downloads and uses them.
```

- [ ] **Step 3: Generate the sample artifact and write the example**

`examples/text_to_image.py`:
```python
"""Offline text-to-image with a downloaded snapshot.

python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5
python examples/text_to_image.py checkpoints/sd-v1-5 "a photograph of an astronaut riding a horse"
"""

import sys

from imagehub import pipeline

model, prompt = sys.argv[1], sys.argv[2]
with pipeline("t2i", backend="sd-v1-5", model=model) as pipe:
    result = pipe(prompt, seed=42)
path = result.images[0].save("output.png")
print(path, result.metadata)
```

```bash
mkdir -p artifacts
.venv/bin/python examples/text_to_image.py checkpoints/sd-v1-5 "a photograph of an astronaut riding a horse"
mv output.png artifacts/sd-v1-5-seed42.png
```

- [ ] **Step 4: Write `docs/catalog.md`, `docs/models.md` and `README.md` (Turkish)**

`docs/catalog.md`:
```markdown
# Model kataloğu — imagehub 0.1

**1 mimari ailesi, 5 yayımlanmış checkpoint.** Bir model, ayrı yayımlanmış bir ağırlık setidir;
fp16 kopyalar, EMA olmayan ağırlıklar ve tek dosyalık `.ckpt` sürümleri ayrı model sayılmaz.

`pipeline("t2i", backend=model_id, model="yerel/klasor")` katalog kimliğiyle aileyi seçer; hangi
ağırlıkların kullanılacağını `model` klasörü belirler.

**Doğrulama seviyeleri:** `full` — tensörler strict yüklendi ve CPU fp32'de diffusers ile parity
kanıtlandı. `header_only` — tensör adları ve şekilleri eşleşti, üretim test edilmedi.
`experimental` — özgün ağırlıklarla test edilmedi. Bu sürümde bütün modeller `full`.

| Model kimliği | Aile | Kaynak | Revision | Köken | Doğrulama | Ağırlık lisansı | Awesome kataloğu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sd-v1-1` | `stable-diffusion` | [CompVis/stable-diffusion-v1-1](https://huggingface.co/CompVis/stable-diffusion-v1-1) | `574555a` | yazar | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-2` | `stable-diffusion` | [CompVis/stable-diffusion-v1-2](https://huggingface.co/CompVis/stable-diffusion-v1-2) | `f519eeb` | yazar | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-3` | `stable-diffusion` | [CompVis/stable-diffusion-v1-3](https://huggingface.co/CompVis/stable-diffusion-v1-3) | `159f26f` | yazar | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-4` | `stable-diffusion` | [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | `133a221` | yazar | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |
| `sd-v1-5` | `stable-diffusion` | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | `451f4fe` | topluluk aynası | full | CreativeML OpenRAIL-M | `stable-diffusion-1` |

`sd-v1-5`'in özgün deposu (`runwayml/stable-diffusion-v1-5`) kaldırılmıştır; kullanılan depo bir
topluluk aynasıdır. v1.1–v1.3 fp32 ağırlıkları yalnızca `.bin` olarak yayımlandığı için
`weights_only=True` ile, yalnızca tensör içerecek şekilde yüklenir.
```
If Task 14 left any model `experimental`, write its real status in the table.

`docs/models.md`:
```markdown
# Stable Diffusion 1.x

Katalog: [`stable-diffusion-1`](https://github.com/gorkemgul/awesome-text-to-image-architectures) ·
Modeller: `sd-v1-1` … `sd-v1-5` · Görev: `t2i`

**Mimari.** f=8 KL autoencoder (512×512×3 görsel ↔ 64×64×4 latent), cross-attention ile koşullanan
860M parametreli U-Net ve donmuş CLIP ViT-L/14 metin kodlayıcı (son katman, havuzlanmamış çıktı).
Zamanlayıcı, checkpoint'lerle yayımlanan PNDM'dir (`skip_prk_steps=True`, yani PLMS):
50 adım 51 U-Net değerlendirmesi yapar.

**Kullanım.**
    python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5

    from imagehub import pipeline
    with pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5", device="mps") as pipe:
        result = pipe("bir kedi, yağlı boya", seed=42, negative_prompt="bulanık")
        result.images[0].save("kedi.png")

**Ayarlar.** `steps` (50), `guidance` (7.5; 1 ve altı guidance'ı kapatır), `width`/`height`
(512, 8'in katı), `negative_prompt`, `num_images` (1), `seed`, `latents`. Tanınmayan ayarlar hata verir.

**Seed uyumu.** CPU'da aynı seed, diffusers'ta `generator=torch.Generator("cpu").manual_seed(seed)`
ile üretilen görselin aynısını verir ([doğrulama](validation.md)). MPS/CUDA'da çalışır, ancak birebir
aynılık iddiası yoktur.

**Kapsam dışı.** Safety checker ve invisible watermark yoktur. img2img, diğer zamanlayıcılar ve fp16
parity bu sürümde yoktur.

**Lisans.** Ağırlıklar CreativeML OpenRAIL-M lisanslıdır; lisansın kullanım kısıtları ağırlıkları
indiren ve kullanan kişiyi bağlar. imagehub ağırlık dağıtmaz.
```

`README.md`:
```markdown
# imagehub

Ağırlığı açık görsel üretim mimarileri için, tek bir API altında, yalnızca PyTorch ile yeniden
yazılmış inference kütüphanesi. **Tek zorunlu çalışma bağımlılığı PyTorch'tur.** Her model,
özgün uygulamayla aynı çıktıyı verdiği kanıtlanarak eklenir ([doğrulama](docs/validation.md)).
Mimariler [awesome-text-to-image-architectures](https://github.com/gorkemgul/awesome-text-to-image-architectures)
kataloğuna bağlıdır.

![sd-v1-5, seed 42](artifacts/sd-v1-5-seed42.png)

## Kurulum

    pip install imagehub            # yalnızca torch
    pip install "imagehub[image]"   # Image.to_pil() için Pillow

## Kullanım

    python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5

    from imagehub import pipeline

    with pipeline("t2i", backend="sd-v1-5", model="checkpoints/sd-v1-5") as pipe:
        result = pipe("a photograph of an astronaut riding a horse", seed=42)
    result.images[0].save("astronot.png")
    print(result.metadata)

İndirme yalnızca `imagehub.pretrained` ile ve açıkça yapılır: doğrulanmış commit sabitlenir,
her dosyanın sha256'sı kontrol edilir, `imagehub_snapshot.json` makbuzu yazılır. `pipeline()` hiçbir
zaman ağa çıkmaz.

## Modeller

| Aile | Checkpoint'ler | Görev | Doğrulama |
| --- | --- | --- | --- |
| Stable Diffusion 1.x | `sd-v1-1`, `sd-v1-2`, `sd-v1-3`, `sd-v1-4`, `sd-v1-5` | `t2i` | CPU fp32 parity |

[Katalog](docs/catalog.md) · [Model sayfaları](docs/models.md) · [Doğrulama](docs/validation.md) ·
[Üçüncü taraf bildirimleri](THIRD_PARTY_NOTICES.md)

## Geliştirme

    uv venv -p 3.13 .venv && uv pip install --python .venv -e ".[dev,image]"
    .venv/bin/python -m pytest

Referans karşılaştırmaları isteğe bağlıdır ve hiçbir şey indirmez:

    uv pip install --python .venv -e ".[dev,image,reference]"
    IMAGEHUB_REFERENCE_TESTS=1 IMAGEHUB_SD_V1_5=checkpoints/sd-v1-5 .venv/bin/python -m pytest tests/reference

## Lisans

Kod Apache-2.0. Model ağırlıkları kendi lisanslarına tabidir ve imagehub ile dağıtılmaz.
```

- [ ] **Step 5: Update `pyproject.toml` license files**

```toml
license-files = ["LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/*.txt"]
```

- [ ] **Step 6: Add a docs consistency test `tests/test_docs.py`**

```python
from pathlib import Path

from imagehub.catalog import MODEL_CATALOG

ROOT = Path(__file__).resolve().parents[1]


def test_every_model_is_documented():
    catalog = (ROOT / "docs/catalog.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for spec in MODEL_CATALOG.values():
        assert f"`{spec.id}`" in catalog and f"`{spec.id}`" in readme
        assert spec.revision[:7] in catalog
```

- [ ] **Step 7: Run tests, build, commit**

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m build
git add README.md docs/catalog.md docs/models.md THIRD_PARTY_NOTICES.md licenses examples artifacts pyproject.toml tests/test_docs.py
git commit -m "Document catalog, SD 1.x usage and third-party notices"
```
Expected: tests pass; `dist/imagehub-0.1.0-py3-none-any.whl` and the sdist are built.

---

### Task 16: Release checks and 0.1.0

Steps 3 and 5 need the user (RunPod account, PyPI credentials). Ask before each; never publish without an explicit "yes".

- [ ] **Step 1: Installed-wheel test on Python 3.13 (latest torch)**

```bash
uv venv -p 3.13 /tmp/imagehub-wheel313
uv pip install --python /tmp/imagehub-wheel313 dist/imagehub-0.1.0-py3-none-any.whl pytest
cd /tmp && /tmp/imagehub-wheel313/bin/python -c "import imagehub; print(imagehub.__version__, imagehub.available_models())"
/tmp/imagehub-wheel313/bin/python -m pytest "$OLDPWD/tests" -q -p no:cacheprovider --rootdir "$OLDPWD" --import-mode=importlib
cd "$OLDPWD"
```
Expected: version `0.1.0`; tests pass except the Pillow test, which skips (`[image]` not installed). Confirm `imagehub` resolves from site-packages, not `src/`: `/tmp/imagehub-wheel313/bin/python -c "import imagehub; print(imagehub.__file__)"`.

- [ ] **Step 2: Minimum-version test (Python 3.11, torch 2.2.2)**

```bash
uv venv -p 3.11 /tmp/imagehub-min
uv pip install --python /tmp/imagehub-min "torch==2.2.2" "numpy<2" dist/imagehub-0.1.0-py3-none-any.whl pytest
cd /tmp && /tmp/imagehub-min/bin/python -m pytest "$OLDPWD/tests" -q -p no:cacheprovider --rootdir "$OLDPWD" --import-mode=importlib
cd "$OLDPWD"
```
Expected: pass (Pillow test skipped). If torch 2.2 lacks an API used here, fix the code (not the floor) unless the fix would hurt parity; in that case raise the floor in `pyproject.toml` and the spec, and record why.

- [ ] **Step 3: CUDA smoke test on RunPod (user-run)**

Ask the user to start an RTX 4090 community pod (~$0.34/h) with a PyTorch image, then on the pod:
```bash
git clone <repo> imagehub && cd imagehub
pip install -e ".[dev,reference]"
python -m imagehub.pretrained sd-v1-5 checkpoints/sd-v1-5
IMAGEHUB_REFERENCE_TESTS=1 IMAGEHUB_SD_V1_5=$PWD/checkpoints/sd-v1-5 \
  IMAGEHUB_RECORD=cuda-measurements.json python -m pytest tests/reference/test_ref_devices.py -v -m cuda
cat cuda-measurements.json
```
Merge the `smoke_cuda` entry into `docs/validation-measurements.json`, re-run `scripts/build_validation.py` (Task 14 Step 6), fill the CUDA row of `docs/validation.md`, and commit:
```bash
git add docs/validation.json docs/validation.md docs/validation-measurements.json
git commit -m "Record CUDA smoke test"
```

- [ ] **Step 4: Final full check**

```bash
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/mypy
.venv/bin/python -m pytest tests -q
rm -rf dist && .venv/bin/python -m build
```

- [ ] **Step 5: Publish (only after the user explicitly approves)**

Show the user the wheel contents (`unzip -l dist/*.whl`) and ask for approval. On "yes":
```bash
uv publish dist/*
git tag -a v0.1.0 -m "imagehub 0.1.0: core + Stable Diffusion 1.x"
```
The user supplies the PyPI token (`UV_PUBLISH_TOKEN`). Do not push tags or publish without that approval.
