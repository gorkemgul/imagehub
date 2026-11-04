"""Experiment trackers for training metrics and sample images (optional packages)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPORTERS = ("wandb", "clearml")


class Reporter:
    """Receives scalars and images from the main training process."""

    def log(self, values: Mapping[str, float], step: int) -> None:
        raise NotImplementedError

    def log_images(self, name: str, images: Sequence[tuple[Path, str]], step: int) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError


def _require(module: str) -> Any:
    try:
        return __import__(module)
    except ImportError as error:
        raise ImportError(f'report_to="{module}" needs the {module} package') from error


class WandbReporter(Reporter):
    def __init__(self, run_name: str | None, config: Mapping[str, Any]) -> None:
        self.wandb = _require("wandb")
        self.run = self.wandb.init(project="imagehub", name=run_name, config=dict(config))

    def log(self, values: Mapping[str, float], step: int) -> None:
        self.run.log(dict(values), step=step)

    def log_images(self, name: str, images: Sequence[tuple[Path, str]], step: int) -> None:
        pictures = [self.wandb.Image(str(path), caption=caption) for path, caption in images]
        self.run.log({name: pictures}, step=step)

    def finish(self) -> None:
        self.run.finish()


class ClearMLReporter(Reporter):
    def __init__(self, run_name: str | None, config: Mapping[str, Any]) -> None:
        clearml = _require("clearml")
        self.task = clearml.Task.init(
            project_name="imagehub", task_name=run_name or "lora-training"
        )
        self.task.connect(dict(config), name="TrainingArguments")
        self.logger = self.task.get_logger()

    def log(self, values: Mapping[str, float], step: int) -> None:
        for name, value in values.items():
            self.logger.report_scalar(title=name, series=name, value=value, iteration=step)

    def log_images(self, name: str, images: Sequence[tuple[Path, str]], step: int) -> None:
        for index, (path, caption) in enumerate(images):
            self.logger.report_image(
                title=name, series=f"{index}: {caption}", iteration=step, local_path=str(path)
            )

    def finish(self) -> None:
        self.task.close()


def start_reporters(
    names: Sequence[str], run_name: str | None, config: Mapping[str, Any]
) -> list[Reporter]:
    """Start one reporter per name; all of them or none."""
    classes = {"wandb": WandbReporter, "clearml": ClearMLReporter}
    started: list[Reporter] = []
    try:
        for name in names:
            started.append(classes[name](run_name, config))
    except BaseException:
        for reporter in started:
            reporter.finish()
        raise
    return started
