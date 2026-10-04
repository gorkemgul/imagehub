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

    Precedence, lowest first: ``default_options``, the request's own parameters, call
    options (shallow merges; see ``with_defaults``). Model construction options belong to the
    backend, never to the request defaults.
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
            raise ValueError(
                f"{self.task.value} pipeline cannot use a {backend.task.value} backend"
            )
        self.backend = backend
        self.default_options = dict(default_options or {})

    @abstractmethod
    def preprocess(self, inputs: InputT) -> PreparedT:
        """Validate and normalize public inputs without loading a model."""

    def with_defaults(self, prepared: PreparedT) -> PreparedT:
        """Place ``default_options`` beneath settings carried by the prepared input.

        Pipelines whose input has its own settings override this together with
        ``call_options``; the generic rule is ``default_options | call options``.
        """
        return prepared

    def call_options(self, options: dict[str, Any]) -> dict[str, Any]:
        """Options passed to the backend for a call; they win over everything else."""
        return self.default_options | options

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
        prepared = self.with_defaults(self.preprocess(inputs))
        result = self.forward(prepared, **self.call_options(options))
        return self.postprocess(result)

    def __call__(self, inputs: InputT, **options: Any) -> OutputT:
        return self.run(inputs, **options)

    def batch(self, inputs: Iterable[InputT], **options: Any) -> list[OutputT]:
        """Materialize one batch; requests run sequentially inside one backend session."""
        prepared = [self.with_defaults(self.preprocess(item)) for item in inputs]
        outputs = self.backend.infer_batch(prepared, **self.call_options(options))
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
