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
