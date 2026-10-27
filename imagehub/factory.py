from collections.abc import Mapping
from typing import Any, Literal, overload

from imagehub.backends.base import Backend
from imagehub.catalog import model_info
from imagehub.licensing import warn_if_noncommercial
from imagehub.pipelines import Pipeline, TextToImagePipeline
from imagehub.registry import BackendRegistry, backend_registry
from imagehub.types import Task

_PIPELINES: dict[Task, type[Pipeline[Any, Any, Any]]] = {Task.T2I: TextToImagePipeline}


@overload
def pipeline(
    task: Literal["t2i", Task.T2I],
    *,
    backend: str | Backend[Any, Any],
    options: Mapping[str, Any] | None = None,
    registry: BackendRegistry | None = None,
    **backend_options: Any,
) -> TextToImagePipeline: ...


@overload
def pipeline(
    task: Task | str,
    *,
    backend: str | Backend[Any, Any],
    options: Mapping[str, Any] | None = None,
    registry: BackendRegistry | None = None,
    **backend_options: Any,
) -> Pipeline[Any, Any, Any]: ...


def pipeline(
    task: Task | str,
    *,
    backend: str | Backend[Any, Any],
    options: Mapping[str, Any] | None = None,
    registry: BackendRegistry | None = None,
    **backend_options: Any,
) -> Pipeline[Any, Any, Any]:
    """Create a task pipeline from a registered name or an existing backend.

    ``backend_options`` configure model construction (``model``, ``device``, ``dtype``);
    ``options`` supplies request defaults. Construction never downloads or loads weights.
    """
    selected = Task(task)
    if isinstance(backend, str):
        selected_registry = registry or backend_registry
        model_id = selected_registry.model_id(selected, backend)
        if model_id is not None:
            warn_if_noncommercial(model_info(model_id))
        instance = selected_registry.create(selected, backend, **backend_options)
    else:
        if backend_options:
            raise ValueError("Model construction options cannot be used with an existing backend")
        instance = backend
    return _PIPELINES[selected](instance, default_options=options)
