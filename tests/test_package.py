import imagehub
from imagehub.exceptions import (
    BackendBusyError,
    BackendContractError,
    CheckpointError,
    DownloadError,
    ImageHubError,
    MissingDependencyError,
)


def test_version():
    assert imagehub.__version__ == "0.1.0"


def test_exception_hierarchy():
    assert issubclass(MissingDependencyError, ImageHubError)
    assert issubclass(MissingDependencyError, ImportError)
    assert issubclass(BackendBusyError, RuntimeError)
    assert issubclass(BackendContractError, TypeError)
    assert issubclass(CheckpointError, ValueError)
    assert issubclass(DownloadError, RuntimeError)
    for error in (BackendBusyError, BackendContractError, CheckpointError, DownloadError):
        assert issubclass(error, ImageHubError)
