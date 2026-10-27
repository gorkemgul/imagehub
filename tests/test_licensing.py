import warnings
from collections.abc import Iterator
from typing import Any

import pytest

from imagehub import ImagehubLicenseWarning, pipeline
from imagehub.backends.base import TextToImageBackend
from imagehub.licensing import _warned_model_ids
from imagehub.registry import BackendRegistry
from imagehub.types import ImageResult, T2IRequest


class _Backend(TextToImageBackend):
    def __init__(self, **options: object) -> None:
        super().__init__()

    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        raise AssertionError("The test backend does not run inference")


@pytest.fixture(autouse=True)
def _clear_license_warnings() -> Iterator[None]:
    _warned_model_ids.clear()
    yield
    _warned_model_ids.clear()


def _registry(model_id: str) -> BackendRegistry:
    registry = BackendRegistry()
    registry.register("t2i", model_id, _Backend, model_id=model_id)
    return registry


def test_noncommercial_weights_warn_when_a_pipeline_is_created() -> None:
    with pytest.warns(ImagehubLicenseWarning, match="flux1-dev: FLUX.1-dev Non-Commercial License"):
        pipeline("t2i", backend="flux1-dev", registry=_registry("flux1-dev"))


def test_noncommercial_implementation_is_included_in_the_warning() -> None:
    with pytest.warns(
        ImagehubLicenseWarning,
        match=r"pixeldit: NVIDIA License .* \(weights\) and NSCLv1 \(implementation\)",
    ):
        pipeline("t2i", backend="pixeldit", registry=_registry("pixeldit"))


def test_permissive_model_does_not_warn() -> None:
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        pipeline("t2i", backend="sd-v1-5", registry=_registry("sd-v1-5"))
    assert not warnings_record


def test_model_license_warning_is_emitted_once() -> None:
    registry = _registry("flux1-dev")
    with pytest.warns(ImagehubLicenseWarning) as warnings_record:
        pipeline("t2i", backend="flux1-dev", registry=registry)
        pipeline("t2i", backend="flux1-dev", registry=registry)
    assert len(warnings_record) == 1
