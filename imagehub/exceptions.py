"""Errors callers can handle without depending on a particular model runtime."""


class ImageHubError(Exception):
    """Base class for library-specific errors."""


class MissingDependencyError(ImageHubError, ImportError):
    """An optional dependency is not installed."""


class BackendBusyError(ImageHubError, RuntimeError):
    """A backend was unloaded or re-entered while an inference was running."""


class BackendContractError(ImageHubError, TypeError):
    """An adapter returned a value that violates its task's contract."""


class CheckpointError(ImageHubError, ValueError):
    """A checkpoint is missing, corrupt, or does not match the model architecture."""


class DownloadError(ImageHubError, RuntimeError):
    """A model download failed verification or could not be completed safely."""
