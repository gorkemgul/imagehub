import warnings
from typing import Any

import pytest

from imagehub import ImageHubLicenseWarning, ImagehubLicenseWarning, pipeline
from imagehub.backends.base import TextToImageBackend
from imagehub.registry import BackendRegistry
from imagehub.types import ImageResult, T2IRequest


class _Backend(TextToImageBackend):
    def __init__(self, **options: object) -> None:
        super().__init__()

    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        raise AssertionError("The test backend does not run inference")


def _registry(model_id: str) -> BackendRegistry:
    registry = BackendRegistry()
    registry.register("t2i", model_id, _Backend, model_id=model_id)
    return registry


def test_noncommercial_weights_warn_when_a_pipeline_is_created() -> None:
    with pytest.warns(ImageHubLicenseWarning, match="flux1-dev: FLUX.1-dev Non-Commercial License"):
        pipeline("t2i", backend="flux1-dev", registry=_registry("flux1-dev"))


def test_noncommercial_implementation_is_included_in_the_warning() -> None:
    with pytest.warns(
        ImageHubLicenseWarning,
        match=r"pixeldit: NVIDIA License .* \(weights\) and NSCLv1 \(implementation\)",
    ):
        pipeline("t2i", backend="pixeldit", registry=_registry("pixeldit"))


def test_permissive_model_does_not_warn() -> None:
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        pipeline("t2i", backend="sd-v1-5", registry=_registry("sd-v1-5"))
    assert not warnings_record


def _create_twice(registry: BackendRegistry) -> None:
    for _ in range(2):
        pipeline("t2i", backend="flux1-dev", registry=registry)


def test_default_filter_emits_model_license_warning_once_per_call_site() -> None:
    registry = _registry("flux1-dev")
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("default", ImageHubLicenseWarning)
        _create_twice(registry)
    assert len(warnings_record) == 1


def test_always_filter_emits_every_model_license_warning() -> None:
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always", ImageHubLicenseWarning)
        _create_twice(_registry("flux1-dev"))
    assert len(warnings_record) == 2


def test_ignore_filter_suppresses_model_license_warning() -> None:
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("ignore", ImageHubLicenseWarning)
        _create_twice(_registry("flux1-dev"))
    assert not warnings_record


def test_old_license_warning_spelling_is_an_alias() -> None:
    assert ImagehubLicenseWarning is ImageHubLicenseWarning
