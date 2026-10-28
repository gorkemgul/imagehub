"""imagehub: open-weight image generation architectures in native PyTorch."""

from imagehub.backends.base import Backend, TextToImageBackend
from imagehub.catalog import ModelSpec, available_models, model_info
from imagehub.exceptions import (
    BackendBusyError,
    BackendContractError,
    CheckpointError,
    DownloadError,
    ImageHubError,
    ImageHubLicenseWarning,
    ImagehubLicenseWarning,
    MissingDependencyError,
)
from imagehub.factory import pipeline
from imagehub.image import Image
from imagehub.pipelines import Pipeline, TextToImagePipeline
from imagehub.registry import BackendRegistry, available_backends, backend_registry
from imagehub.types import ImageResult, T2IRequest, Task

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "BackendBusyError",
    "BackendContractError",
    "BackendRegistry",
    "CheckpointError",
    "DownloadError",
    "Image",
    "ImageHubError",
    "ImageHubLicenseWarning",
    "ImagehubLicenseWarning",
    "ImageResult",
    "ModelSpec",
    "MissingDependencyError",
    "Pipeline",
    "T2IRequest",
    "Task",
    "TextToImageBackend",
    "TextToImagePipeline",
    "__version__",
    "available_backends",
    "available_models",
    "backend_registry",
    "model_info",
    "pipeline",
]
