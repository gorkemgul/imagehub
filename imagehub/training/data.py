"""Image-caption datasets and the training image transform.

``preprocess_image`` reproduces torchvision's ``Resize`` / ``CenterCrop`` / ``RandomCrop`` /
``RandomHorizontalFlip`` / ``ToTensor`` / ``Normalize([0.5], [0.5])`` chain with pillow, drawing its
random numbers from an explicit generator.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def _pil() -> Any:
    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError(
            'Training needs pillow; install it with pip install "imagehub[training]"'
        ) from error
    return Image


class ImageFolderDataset:
    """Images listed in ``root/metadata.jsonl``, one ``{"file_name", "text"}`` object per line."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        image_module = _pil()
        self.root = Path(root)
        metadata = self.root / "metadata.jsonl"
        if not metadata.is_file():
            raise FileNotFoundError(f"{metadata} is missing")
        self.items: list[tuple[Path, str]] = []
        for number, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{metadata}:{number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{where}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{where}: expected an object")
            name, text = record.get("file_name"), record.get("text")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{where}: file_name must be a non-empty string")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{where}: text must be a non-empty string")
            path = self.root / name
            if not path.is_file():
                raise FileNotFoundError(f"{where}: {path} does not exist")
            try:
                with image_module.open(path) as image:
                    image.verify()
            except Exception as error:
                raise ValueError(f"{where}: {path} is not a readable image") from error
            self.items.append((path, text))
        if not self.items:
            raise ValueError(f"{metadata} lists no images")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, text = self.items[index]
        with _pil().open(path) as image:
            return {"image": image.convert("RGB"), "text": text}


def _resize(image: PILImage, resolution: int) -> PILImage:
    width, height = image.size
    short, long = (width, height) if width <= height else (height, width)
    new_short, new_long = resolution, int(resolution * long / short)
    size = (new_short, new_long) if width <= height else (new_long, new_short)
    return image.resize(size, _pil().BILINEAR)


def _offsets(
    height: int, width: int, size: int, center: bool, generator: torch.Generator | None
) -> tuple[int, int]:
    if height < size or width < size:
        raise ValueError(f"Image of {width}x{height} is smaller than the crop size {size}")
    if center:
        return int(round((height - size) / 2.0)), int(round((width - size) / 2.0))
    if height == size and width == size:
        return 0, 0
    top = int(torch.randint(0, height - size + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, width - size + 1, (1,), generator=generator).item())
    return top, left


def _resized_pixels(image: Any, resolution: int) -> tuple[Tensor, tuple[int, int]]:
    """``(3, H, W)`` pixels in ``[0, 1]`` with the short side ``resolution``, and the original
    ``(height, width)``."""
    if isinstance(image, Tensor):
        if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
            raise ValueError("Image tensors must be floating point with shape (3, H, W)")
        pixels = image.float()
        if min(pixels.shape[1:]) != resolution:
            raise ValueError(f"Image tensors must have a short side of {resolution}")
        return pixels, (int(pixels.shape[1]), int(pixels.shape[2]))
    rgb = image.convert("RGB")
    resized = _resize(rgb, resolution)
    data = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
    pixels = data.view(resized.size[1], resized.size[0], 3).permute(2, 0, 1).float() / 255
    return pixels, (rgb.size[1], rgb.size[0])


def preprocess_image(
    image: Any,
    *,
    resolution: int,
    center_crop: bool,
    random_flip: bool,
    generator: torch.Generator | None = None,
) -> Tensor:
    """A ``(3, resolution, resolution)`` tensor in ``[-1, 1]`` from a PIL image or a tensor.

    Tensors are ``(3, H, W)`` in ``[0, 1]``; they are cropped but not resized, so their short side
    must already be ``resolution``.
    """
    pixels, _ = _resized_pixels(image, resolution)
    top, left = _offsets(pixels.shape[1], pixels.shape[2], resolution, center_crop, generator)
    pixels = pixels[:, top : top + resolution, left : left + resolution]
    if random_flip and torch.rand(1, generator=generator) < 0.5:
        pixels = pixels.flip(-1)
    return (pixels - 0.5) / 0.5


def preprocess_image_xl(
    image: Any,
    *,
    resolution: int,
    center_crop: bool,
    random_flip: bool,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, tuple[int, int], tuple[int, int]]:
    """The SDXL script's transform: resize, flip, then crop.

    Returns the ``[-1, 1]`` pixels, the original ``(height, width)`` and the crop's
    ``(top, left)``, which SDXL takes as size conditioning.
    """
    pixels, original_size = _resized_pixels(image, resolution)
    if random_flip and torch.rand(1, generator=generator) < 0.5:
        pixels = pixels.flip(-1)
    top, left = _offsets(pixels.shape[1], pixels.shape[2], resolution, center_crop, generator)
    pixels = pixels[:, top : top + resolution, left : left + resolution]
    return (pixels - 0.5) / 0.5, original_size, (top, left)


class StableDiffusionCollator:
    """Batch ``{"image", "text"}`` items into ``{"pixel_values", "input_ids"}``.

    Images go through :func:`preprocess_image` with ``generator``; captions through the
    checkpoint's CLIP tokenizer.
    """

    def __init__(
        self,
        tokenizer: Callable[[str], list[int]],
        *,
        resolution: int,
        center_crop: bool,
        random_flip: bool,
        generator: torch.Generator | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.generator = generator

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
        pixels, ids = [], []
        for feature in features:
            pixels.append(
                preprocess_image(
                    feature["image"],
                    resolution=self.resolution,
                    center_crop=self.center_crop,
                    random_flip=self.random_flip,
                    generator=self.generator,
                )
            )
            ids.append(self.tokenizer(feature["text"]))
        return {"pixel_values": torch.stack(pixels), "input_ids": torch.tensor(ids)}


class StableDiffusionXLCollator:
    """Batch items for SDXL: pixels, ids for both CLIP tokenizers and ``time_ids``.

    ``time_ids`` rows are ``(original height, original width, crop top, crop left, resolution,
    resolution)``, as the SDXL training script computes them.
    """

    def __init__(
        self,
        tokenizer: Callable[[str], list[int]],
        tokenizer_2: Callable[[str], list[int]],
        *,
        resolution: int,
        center_crop: bool,
        random_flip: bool,
        generator: torch.Generator | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.generator = generator

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
        pixels, ids, ids_2, time_ids = [], [], [], []
        for feature in features:
            image, original_size, crop = preprocess_image_xl(
                feature["image"],
                resolution=self.resolution,
                center_crop=self.center_crop,
                random_flip=self.random_flip,
                generator=self.generator,
            )
            pixels.append(image)
            ids.append(self.tokenizer(feature["text"]))
            ids_2.append(self.tokenizer_2(feature["text"]))
            time_ids.append([*original_size, *crop, self.resolution, self.resolution])
        return {
            "pixel_values": torch.stack(pixels),
            "input_ids": torch.tensor(ids),
            "input_ids_2": torch.tensor(ids_2),
            "time_ids": torch.tensor(time_ids, dtype=torch.float32),
        }
