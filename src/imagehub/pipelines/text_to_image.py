from __future__ import annotations

from typing import Any

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

    def with_defaults(self, prepared: T2IRequest) -> T2IRequest:
        """Pipeline defaults sit below the request's own parameters."""
        if not self.default_options:
            return prepared
        return T2IRequest(prepared.prompt, self.default_options | dict(prepared.parameters))

    def call_options(self, options: dict[str, Any]) -> dict[str, Any]:
        return options
