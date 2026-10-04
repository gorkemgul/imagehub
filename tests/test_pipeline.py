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
