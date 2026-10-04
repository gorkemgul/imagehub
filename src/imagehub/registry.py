"""Explicit, extensible backend selection with lazy architecture imports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from imagehub.catalog import MODEL_CATALOG
from imagehub.types import Task

BackendFactory = Callable[..., Any]


@dataclass(frozen=True)
class _Registration:
    factory: BackendFactory | str
    model_id: str | None = None


class BackendRegistry:
    def __init__(self) -> None:
        self._entries: dict[tuple[Task, str], _Registration] = {}

    def register(
        self,
        task: Task | str,
        name: str,
        factory: BackendFactory | str,
        *,
        replace: bool = False,
        model_id: str | None = None,
    ) -> None:
        """Register a callable or ``module:class`` import path; no model is loaded.

        ``model_id`` marks the entry as a catalog model alias: ``create`` then passes
        ``model_id=<id>`` to the factory.
        """
        if not name or name.strip() != name:
            raise ValueError("Backend name must be nonempty and have no surrounding whitespace")
        key = (Task(task), name)
        if key in self._entries and not replace:
            raise ValueError(f"Backend {name!r} is already registered for {key[0].value}")
        if isinstance(factory, str):
            if len(factory.split(":")) != 2 or not all(factory.split(":")):
                raise ValueError("Lazy backend paths must use 'module:class'")
        elif not callable(factory):
            raise TypeError("Backend factory must be callable or a 'module:class' string")
        self._entries[key] = _Registration(factory, model_id)

    def names(self, task: Task | str) -> tuple[str, ...]:
        selected = Task(task)
        return tuple(sorted(name for (kind, name) in self._entries if kind == selected))

    def _entry(self, task: Task, name: str) -> _Registration:
        entry = self._entries.get((task, name))
        if entry is None:
            available = ", ".join(self.names(task)) or "none"
            raise ValueError(f"Unknown {task.value} backend {name!r}. Available: {available}")
        return entry

    def factory(self, task: Task | str, name: str) -> BackendFactory | str:
        return self._entry(Task(task), name).factory

    def create(self, task: Task | str, name: str, **options: Any) -> Any:
        selected = Task(task)
        entry = self._entry(selected, name)
        factory = entry.factory
        if entry.model_id is not None:
            options = {"model_id": entry.model_id} | options
        # Imported here: backends depend on the registry's package, not the other way round.
        from imagehub.backends.base import Backend

        if isinstance(factory, str):
            module, attribute = factory.split(":")
            factory = getattr(import_module(module), attribute)
        backend = factory(**options)
        if not isinstance(backend, Backend) or backend.task != selected:
            raise TypeError(f"Factory {name!r} must return a {selected.value} Backend")
        return backend


backend_registry = BackendRegistry()
for _task, _name, _path in (
    (Task.T2I, "stable-diffusion", "imagehub.backends.stable_diffusion:StableDiffusionBackend"),
):
    backend_registry.register(_task, _name, _path)

# Published model IDs are also valid selectors. Their checkpoint files stay local
# and explicit, just like family selectors; listing a model never downloads it.
for _spec in MODEL_CATALOG.values():
    if _spec.id not in backend_registry.names(_spec.task):
        backend_registry.register(
            _spec.task,
            _spec.id,
            backend_registry.factory(_spec.task, _spec.family),
            model_id=_spec.id,
        )


def available_backends(task: Task | str | None = None) -> dict[str, tuple[str, ...]]:
    tasks = (Task(task),) if task is not None else tuple(Task)
    return {kind.value: backend_registry.names(kind) for kind in tasks}
