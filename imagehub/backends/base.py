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
