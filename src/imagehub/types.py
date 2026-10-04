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
    # Request-level generation settings: they override a pipeline's default options, and
    # options passed to a pipeline call override them.
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
