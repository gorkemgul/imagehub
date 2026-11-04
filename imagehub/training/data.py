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
    if isinstance(image, Tensor):
        if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
            raise ValueError("Image tensors must be floating point with shape (3, H, W)")
        pixels = image.float()
        if min(pixels.shape[1:]) != resolution:
            raise ValueError(f"Image tensors must have a short side of {resolution}")
    else:
        resized = _resize(image.convert("RGB"), resolution)
        data = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
        pixels = data.view(resized.size[1], resized.size[0], 3).permute(2, 0, 1).float() / 255
    top, left = _offsets(pixels.shape[1], pixels.shape[2], resolution, center_crop, generator)
    pixels = pixels[:, top : top + resolution, left : left + resolution]
    if random_flip and torch.rand(1, generator=generator) < 0.5:
        pixels = pixels.flip(-1)
    return (pixels - 0.5) / 0.5


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
