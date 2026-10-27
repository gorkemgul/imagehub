"""Locate test fixtures, including optional files that cannot be redistributed."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
RESTRICTED_FIXTURES = FIXTURES / "_restricted"
_MANIFEST = json.loads((FIXTURES / "RESTRICTED.json").read_text(encoding="utf-8"))
_RESTRICTED_TARGETS = {entry["target"] for entry in _MANIFEST}


def restricted_fixture(relative: str | Path) -> Path:
    """Return a fetched restricted fixture, or skip the test with setup instructions."""
    path = RESTRICTED_FIXTURES / relative
    if not path.exists():
        pytest.skip(f"restricted fixture is unavailable: {relative}; run scripts/fetch_fixtures.py")
    return path


@dataclass(frozen=True)
class FixturePath(os.PathLike[str]):
    """A lazy fixture path that redirects only manifest-listed files."""

    relative: Path

    def __truediv__(self, child: str | os.PathLike[str]) -> "FixturePath":
        return FixturePath(self.relative / child)

    def _resolve(self) -> Path:
        regular = FIXTURES / self.relative
        if regular.exists():
            return regular
        relative = self.relative.as_posix()
        if relative in _RESTRICTED_TARGETS or any(
            target.startswith(f"{relative}/") for target in _RESTRICTED_TARGETS
        ):
            return restricted_fixture(self.relative)
        return regular

    def __fspath__(self) -> str:
        return os.fspath(self._resolve())

    def __str__(self) -> str:
        return str(self._resolve())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)


def fixture_path(relative: str | Path = ".") -> FixturePath:
    """Build a lazy path that resolves restricted files through their cache."""
    return FixturePath(Path(relative))
